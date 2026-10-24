"""Write the GoodsAD train and val splits as Parquet conversations for Megatron-Bridge.

Prompts, labels and the product split come from test/test.py; images get the same preprocessing test.py applies
before sending them to a server (shrunk to 2048 px, JPEG quality 90), with `Answer: <label>` as the assistant turn.
Megatron-Bridge renders them with the model's chat template, which puts the same <think></think> before the answer
that inference prompts end with. The image sits inside the user turn, where Megatron-Bridge's Nemotron Omni collate
looks for it.

    python train/nemotron/prepare.py --out runs/train/nemotron-megatron/data
"""
import argparse
import base64
import json
import random
import runpy
from collections import namedtuple
from multiprocessing import Pool
from pathlib import Path

import datasets

suite = runpy.run_path(str(Path(__file__).resolve().parents[2] / "test/test.py"))
SPEC, ROOT, split_of = suite["SPEC"], suite["ROOT"], suite["split_of"]
Row = namedtuple("Row", "cat label path sku split")
TEXT = datasets.Value("string")
FEATURES = datasets.Features({
    "conversation": [{"role": TEXT, "content": [{"type": TEXT, "text": TEXT, "image": datasets.Image()}]}],
    **{k: TEXT for k in Row._fields},
})


def load_splits(good_ratio, limit, seed):
    rows, rng = [], random.Random(seed)
    for cat, spec in SPEC.items():
        for path in sorted((ROOT / cat).glob("*/*/*.jpg")):
            sku, label = path.stem[:3], path.parent.name
            split = split_of(cat, sku)
            if label in spec["labels"] and (split == "train" or path.parts[-3] == "test"):
                rows.append(Row(cat, label, str(path), sku, split))
    splits = {s: [r for r in rows if r.split == s] for s in ("train", "val", "test")}
    if good_ratio:
        train = []
        for cat in SPEC:
            good = [r for r in splits["train"] if r.cat == cat and r.label == "good"]
            bad = [r for r in splits["train"] if r.cat == cat and r.label != "good"]
            train += bad + rng.sample(good, min(len(good), round(good_ratio * len(bad))))
        splits["train"] = train
    return {s: rng.sample(v, min(limit, len(v))) if limit else v for s, v in splits.items()}


def image_bytes(path):
    """The JPEG test.py sends a server for this image: shrunk to MAX_SIDE and re-encoded at quality 90."""
    return base64.b64decode(suite["uri"](Path(path), suite["MAX_SIDE"]).split(",", 1)[1])


def conversation(row, jpeg):
    prompt = f"Image to inspect.\n{suite['prompt'](row.cat)}"
    return [{"role": "user", "content": [{"type": "image", "text": None, "image": {"bytes": jpeg, "path": None}},
                                         {"type": "text", "text": prompt, "image": None}]},
            {"role": "assistant", "content": [{"type": "text", "text": f"Answer: {row.label}", "image": None}]}]


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = load_splits(args.good_ratio, args.limit, args.seed)
    with Pool(args.workers) as pool:
        for name in ("train", "val"):
            rows = splits[name]
            jpegs = pool.map(image_bytes, [r.path for r in rows], chunksize=8)
            records = [dict(conversation=conversation(r, j), **r._asdict()) for r, j in zip(rows, jpegs)]
            datasets.Dataset.from_list(records, features=FEATURES).to_parquet(out / f"{name}.parquet")
    (out / "sizes.json").write_text(json.dumps({s: len(v) for s, v in splits.items()}) + "\n")
    print({s: len(v) for s, v in splits.items()}, "->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GoodsAD conversations for Megatron-Bridge")
    ap.add_argument("--out", required=True)
    ap.add_argument("--good-ratio", type=float, default=1.0, help="train good images per defect, per category; 0 = all")
    ap.add_argument("--limit", type=int, default=0, help="rows per split, for smoke tests")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=16)
    main(ap.parse_args())
