"""
Fine-tune Gemma 4 E2B: LoRA or full.

    python train/gemma/train.py --out runs/train/base --eval-only                        # baseline on the test SKUs
    torchrun --nproc_per_node 4 train/gemma/train.py --out runs/train/lora                # LoRA on the LM and vision tower
    torchrun --nproc_per_node 4 train/gemma/train.py --out runs/train/full --mode full --no-vision

Needs transformers>=5.5, torchvision, peft and accelerate (nvcr.io/nvidia/nemo:26.08 ships them all);
HF_HOME=.cache/hf reuses the model servers' weights. --out gets split.jsonl, Trainer checkpoints, model/ (merged bf16, servable by
vLLM), test_predictions.jsonl and test_metrics.csv.
"""
import argparse
import os
import random
import runpy
import shutil
import time
from collections import namedtuple
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from PIL import Image
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# cuDNN's attention backward fails on B300 in the NeMo 26.08 container (CUDNN_STATUS_BAD_PARAM); SDPA has other kernels.
torch.backends.cuda.enable_cudnn_sdp(False)
suite = runpy.run_path(str(Path(__file__).resolve().parents[2] / "test/test.py"))
SPEC, ROOT, split_of = suite["SPEC"], suite["ROOT"], suite["split_of"]
PROJ = r"(q|k|v|o|gate|up|down)_proj"
MAIN = os.environ.get("RANK", "0") == "0"
Row = namedtuple("Row", "cat label path sku split")


print_trainable_params = lambda model: print(f"trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M params")

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


def collate(rows, processor, prompts, train):
    answers = [f"Answer: {r.label}<turn|>" if train else "" for r in rows]
    batch = processor(text=[prompts[r.cat] + a for r, a in zip(rows, answers)],
                      images=[[Image.open(r.path).convert("RGB")] for r in rows],
                      add_special_tokens=False, return_tensors="pt")
    if not train:
        return batch
    # Left padding ends every row with its answer, so only the last k positions need logits.
    ids = [processor.tokenizer(a, add_special_tokens=False).input_ids for a in answers]
    k = max(map(len, ids)) + 1
    labels = batch["input_ids"][:, -k:].clone()
    for i, a in enumerate(ids):
        assert labels[i, -len(a):].tolist() == a, "answer tokens must end the sequence"
        labels[i, :-len(a)] = -100
    return {**batch, "labels": labels, "logits_to_keep": k}


def finetune(model, processor, prompts, splits, args):
    if args.mode == "lora":
        target = rf"model\.language_model\.layers\.\d+\.\w+\.{PROJ}"
        if args.vision:
            target += rf"|model\.vision_tower\.encoder\.layers\.\d+\.\w+\.{PROJ}\.linear"
        model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                                                 target_modules=target, task_type="CAUSAL_LM"))
    else:
        frozen = ("audio", "embed_tokens") + (() if args.vision else ("vision",))
        for name, p in model.named_parameters():
            p.requires_grad = not any(k in name for k in frozen)

    if MAIN:
        print_trainable_params(model)

    trainer = Trainer(
        model=model, train_dataset=splits["train"], eval_dataset=splits["val"],
        data_collator=partial(collate, processor=processor, prompts=prompts, train=True),
        args=TrainingArguments(
            output_dir=args.out, num_train_epochs=args.epochs, max_steps=args.max_steps,
            per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr or (2e-4 if args.mode == "lora" else 1e-5),
            lr_scheduler_type="cosine", warmup_steps=0.03, bf16=True,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, save_only_model=True,
            load_best_model_at_end=True, logging_steps=10, dataloader_num_workers=8,
            remove_unused_columns=False, label_names=["labels"], ddp_find_unused_parameters=False,
            seed=args.seed))

    trainer.train()
    model = trainer.model
    return (model.merge_and_unload() if args.mode == "lora" else model).to(torch.bfloat16)


def export(state, base, dst):
    """Write the base checkpoint with tuned tensors swapped in. transformers drops tensors it doesn't use
    (KV-shared layers' k/v projections) and re-serializes config.json; the vLLM image needs both as shipped."""
    src = Path(base) if Path(base).is_dir() else Path(snapshot_download(base))
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.safetensors"), dirs_exist_ok=True)
    written = set()
    for f in src.glob("*.safetensors"):
        tensors = load_file(f)
        written |= tensors.keys()
        save_file({k: state.get(k, v).to("cpu", v.dtype) for k, v in tensors.items()}, dst / f.name,
                  metadata={"format": "pt"})
    assert set(state) <= written | {"lm_head.weight"}, "model keys must match the base checkpoint's"


@torch.inference_mode()
def evaluate(model, processor, prompts, rows, batch_size):
    model.eval()
    records = []
    for i in tqdm(range(0, len(rows), batch_size), desc="test"):
        chunk, start = rows[i:i + batch_size], time.perf_counter()
        batch = collate(chunk, processor, prompts, train=False).to(model.device)
        ids = model.generate(**batch, max_new_tokens=128, do_sample=False)
        texts = processor.batch_decode(ids[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        sec = (time.perf_counter() - start) / len(chunk)
        records += [dict(r._asdict(), shot="zero", raw=t, pred=suite["parse"](t, SPEC[r.cat]["labels"]),
                         error=None, sec=sec) for r, t in zip(chunk, texts)]
    return pd.DataFrame(records)


def main(args):
    if args.eval_only and not MAIN:
        return
    out = Path(args.out)
    splits = load_splits(args.good_ratio, args.limit, args.seed)

    if MAIN:
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([r for v in splits.values() for r in v]).to_json(out / "split.jsonl", orient="records", lines=True)
        print({s: len(v) for s, v in splits.items()})

    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    processor.image_processor.max_soft_tokens = args.vision_tokens
    image = next((ROOT / "food_box/train/good").glob("*.jpg"))
    prompts = {c: processor.apply_chat_template(suite["messages"](c, image), tokenize=False,
                                                add_generation_prompt=True) for c in SPEC}
    fp32 = args.mode == "full" and not args.eval_only  # fp32 master weights; bf16 autocast does the math
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float32 if fp32 else torch.bfloat16)

    if args.eval_only:
        model.cuda()
    else:
        model = finetune(model, processor, prompts, splits, args)
        if not MAIN:
            return
        export(model.state_dict(), args.model, out / "model")

    df = evaluate(model, processor, prompts, splits["test"], args.batch)
    df.to_json(out / "test_predictions.jsonl", orient="records", lines=True)
    report = suite["report"](df)
    report.to_csv(out / "test_metrics.csv")
    print(report.to_string(), suite["breakdown"](df).to_string(), sep="\n\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fine-tune Gemma 4 on GoodsAD")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="google/gemma-4-E2B-it", help="HF id or local dir, e.g. <out>/model")
    ap.add_argument("--mode", choices=("lora", "full"), default="lora")
    ap.add_argument("--vision", action=argparse.BooleanOptionalAction, default=True, help="also tune the vision tower")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, help="default 2e-4 for lora, 1e-5 for full")
    ap.add_argument("--batch", type=int, default=8, help="per GPU")
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--good-ratio", type=float, default=1.0, help="train good images per defect, per category; 0 = all")
    ap.add_argument("--vision-tokens", type=int, default=560, help="max soft tokens per image; match the server's VISION_TOKENS")
    ap.add_argument("--eval-only", action="store_true", help="skip training; score --model on the test SKUs")
    ap.add_argument("--limit", type=int, default=0, help="rows per split, for smoke tests")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
