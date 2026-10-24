"""
Evaluation: zero-shot vs few-shot, per category or all at once.

    python test/test.py                                         # all categories, zero + few, 10/label
    python test/test.py --cats cigarette_box --shots few --n 0  # one category, every test image
    python test/test.py --cats drink_bottle --labels good cap_half_open --n 30

Few-shot prepends one curated example per label, so a request carries up to 6 images:
serve with MAX_IMAGES>=6. Every SKU used as an example is held out of evaluation in both
modes, so zero- and few-shot are scored on identical images.

Products are split train/val/test (split_of). The trainers in train/ fine-tune only on train, and the
suite scores only test by default, so base and fine-tuned models meet the same unseen products.
--split all scores every test image, including products a fine-tuned model has seen.
"""

import argparse
import base64
import hashlib
import io
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(HERE))
from utils import api, model_id
from spec import SPEC, CATS, prompt

ROOT, OUT = BASE / "data/goodsad", BASE / "runs/goodsad"
URL = os.getenv("URL", "http://127.0.0.1:8000")
SHOTS = ("zero", "few")
MAX_SIDE = 2048  # NIM rejects request bodies over 10 MiB; six 2048 px JPEGs stay under ~7 MiB

@lru_cache(maxsize=32)
def uri(path, side):
    im = Image.open(path).convert("RGB")
    im.thumbnail((side, side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# Data: n images per label, fixed seed; SKUs (first 3 filename chars) used as shots are held out
def split_of(cat, sku):
    """Product split, 70/15/15 with hash; shot SKUs are train because prompts show them."""
    if sku in {Path(p).stem[:3] for _, p, _ in SPEC[cat]["shots"]}:
        return "train"
    bucket = int(hashlib.sha256(f"42:goodsad/{cat}/{sku}".encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "val" if bucket < 85 else "test"

def samples(cat, n=None, labels=None, seed=0, split="test"):
    held = {Path(p).stem[:3] for _, p, _ in SPEC[cat]["shots"]}
    rows = []
    for label in (l for l in SPEC[cat]["labels"] if not labels or l in labels):
        files = [p for p in sorted((ROOT / cat / "test" / label).glob("*.jpg"))
                 if p.stem[:3] not in held and split in ("all", split_of(cat, p.stem[:3]))]
        rows += [(label, p) for p in (random.Random(seed).sample(files, min(n, len(files))) if n else files)]
    return rows

def user(text, path):
    return {"role": "user", "content": [{"type": "text", "text": text},
                                        {"type": "image_url", "image_url": {"url": uri(Path(path), MAX_SIDE)}}]}

def messages(cat, path, few=False):
    p, msgs = prompt(cat), []
    for i, (label, shot, why) in enumerate(SPEC[cat]["shots"] if few else [], 1):
        msgs += [user(f"Example {i}.\n{p}", ROOT / cat / shot),
                 {"role": "assistant", "content": f"{why}\nAnswer: {label}"}]
    return msgs + [user(f"Image to inspect.\n{p}", path)]

# Inference
ANSWER = re.compile(r"answer\b\W*([a-z][a-z _-]*)", re.I)

def ask(url, model, msgs, think=False):
    # No max_tokens: each server caps output at its own context (vLLM 32k, NIM 262k); a fixed 65536 fails on vLLM.
    body = {"model": model, "messages": msgs, "temperature": 0, "chat_template_kwargs": {"enable_thinking": think}}
    t = time.perf_counter()
    choice = api(url.rstrip("/") + "/v1/chat/completions", body)["choices"][0]
    return dict(raw=choice["message"].get("content") or "", finish=choice.get("finish_reason"),
                sec=round(time.perf_counter() - t, 2))

def parse(text, labels):
    found = ANSWER.findall(re.sub(r"(?s)<think>.*?(?:</think>|$)", "", text or ""))
    got = re.sub(r"[\s-]+", "_", found[-1].strip().lower()) if found else ""
    return next((l for l in sorted(labels, key=len, reverse=True) if got.startswith(l)), None)


def run(cats=CATS, shots=SHOTS, n=10, labels=None, url=None, think=False, workers=8, seed=0, split="test"):
    cats, shots, url = [cats] if isinstance(cats, str) else cats, [shots] if isinstance(shots, str) else shots, url or URL
    model = model_id(url)
    jobs = [(c, label, p, s) for c in cats for label, p in samples(c, n, labels, seed, split) for s in shots]

    def one(job):
        cat, label, path, shot = job
        rec = dict(cat=cat, label=label, shot=shot, pred=None, path=str(path), split=split_of(cat, path.stem[:3]),
                   model=model, think=think, side=MAX_SIDE, raw=None, finish=None, sec=None, error=None)
        try:
            rec.update(ask(url, model, messages(cat, path, shot == "few"), think))
            rec["pred"] = parse(rec["raw"], SPEC[cat]["labels"])
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        return rec

    OUT.mkdir(parents=True, exist_ok=True)
    log, rows = OUT / f"{time.strftime('%Y%m%d-%H%M%S')}_{model.split('/')[-1]}.jsonl", []
    pool = ThreadPoolExecutor(workers)
    try:
        with log.open("w") as f:
            for fut in tqdm(as_completed([pool.submit(one, j) for j in jobs]), total=len(jobs)):
                rows.append(fut.result())
                f.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
    except KeyboardInterrupt:
        print("Interrupted; returning finished results.")
    finally:
        pool.shutdown(cancel_futures=True)
    print(f"{len(rows)}/{len(jobs)} results -> {log}")
    return pd.DataFrame(rows)

# Defect is the positive class; unparseable replies and errors count as wrong
def _div(a, b):
    return a / b if b else float("nan")

def _score(g):
    bad, flag = g.label.ne("good"), g.pred.notna() & g.pred.ne("good")
    tp, fp = (bad & flag).sum(), (~bad & flag).sum()
    recall, spec, prec = _div(tp, bad.sum()), _div((~bad & g.pred.eq("good")).sum(), (~bad).sum()), _div(tp, tp + fp)
    return pd.Series(dict(n=len(g), recall=recall, specificity=spec, bal_acc=(recall + spec) / 2, precision=prec,
                          f1=_div(2 * prec * recall, prec + recall), type_acc=g.pred.eq(g.label).mean(),
                          invalid=g.pred.isna().mean(), errors=g.error.notna().mean(), sec=g.sec.mean()))

def report(df):
    d = pd.concat([df, df.assign(cat="ALL")]) if df.cat.nunique() > 1 else df
    return d.groupby(["cat", "shot"]).apply(_score).round(3).astype({"n": int})

def breakdown(df):
    d = df.assign(detected=df.pred.notna() & df.pred.ne("good").eq(df.label.ne("good")), exact=df.pred.eq(df.label))
    return d.pivot_table(index=["cat", "label"], columns="shot", values=["detected", "exact"], aggfunc="mean").round(3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GoodsAD zero/few-shot defect suite")
    ap.add_argument("--cats", nargs="+", default=CATS, choices=CATS)
    ap.add_argument("--shots", nargs="+", default=SHOTS, choices=SHOTS)
    ap.add_argument("--labels", nargs="+", help="only these labels, e.g. good cap_half_open")
    ap.add_argument("--n", type=int, default=10, help="images per label; 0 = all")
    ap.add_argument("--split", default="test", choices=("test", "val", "all"),
                    help="products to score; all = every test image, including ones fine-tuning saw")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a, _ = ap.parse_known_args()
    df = run(a.cats, a.shots, a.n, a.labels, a.url, a.think, a.workers, split=a.split)
    print(report(df).to_string(), breakdown(df).to_string(), sep="\n\n")
