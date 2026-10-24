"""Merge a PEFT adapter exported from Megatron-Bridge into the base checkpoint, servable by vLLM.

    python train/nemotron/merge.py --adapter runs/train/nemotron-megatron/adapter --out runs/train/nemotron-megatron/model
"""
import argparse
import re
import shutil
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModel

BASE = Path(__file__).resolve().parents[2] / ".cache/models/nemotron-3-nano-omni-30b-a3b-reasoning-bf16"
# The adapter lists bare names (q_proj, up_proj, ...) that would also match every routed expert.
TARGET = r"language_model\.backbone\.layers\.\d+\.mixer\.((q|k|v|o)_proj|(in|out)_proj|shared_experts\.(up|down)_proj)"


def main(args):
    model = AutoModel.from_pretrained(BASE, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": 0})
    config = LoraConfig.from_pretrained(args.adapter)
    config.target_modules = TARGET
    model = PeftModel.from_pretrained(model, args.adapter, config=config).merge_and_unload()
    state = {f"{n}.weight": m.weight for n, m in model.named_modules() if re.fullmatch(TARGET, n)}

    # Copy the base checkpoint as shipped, with only the merged tensors swapped in.
    out = Path(args.out)
    shutil.copytree(BASE, out, ignore=shutil.ignore_patterns("*.safetensors"), dirs_exist_ok=True)
    for f in sorted(BASE.glob("*.safetensors")):
        tensors = load_file(f)
        save_file({k: state.pop(k, v).to("cpu", v.dtype) for k, v in tensors.items()}, out / f.name,
                  metadata={"format": "pt"})
    assert not state, "merged keys must match the base checkpoint's"
    print(f"merged checkpoint: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Merge a Nemotron LoRA adapter into the base checkpoint")
    ap.add_argument("--adapter", required=True, help="PEFT adapter dir from export_adapter.py")
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
