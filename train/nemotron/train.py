"""LoRA fine-tune Nemotron 3 Nano Omni with Megatron-Bridge.

Starts from NVIDIA's Omni LoRA recipe (nemotron_omni_cord_v2_peft_config: rank 16, alpha 32, Adam with betas 0.9 and
0.95, LR 1e-4 cosine to 1e-5, bf16) and changes only what this task needs:

- data: the Parquet conversations from prepare.py instead of CORD-v2, with a val split, sampled as real shuffled epochs.
- LoRA targets: the language model's attention, Mamba and shared-expert projections. The recipe's bare names would
  also put adapters on the vision encoder and both projectors, but this task only needs the LM.
- schedule: global batch 24, 3 epochs, 3% warmup, a checkpoint at the end of every epoch.
- routers truly frozen: no MoE load-balancing loss and no expert-bias updates, which Megatron applies outside PEFT.
"""
import argparse
import json
import math
from pathlib import Path

import megatron.bridge.recipes.nemotron_omni.h100.nemotron_omni as recipe
from megatron.bridge.data.datasets.direct_sft import DirectSFTDataset
from megatron.bridge.data.sources.hf import HFDatasetSourceConfig
from megatron.bridge.training.config import apply_environment_variables
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.nemotron_omni_step import forward_step
from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides

REPO = Path(__file__).resolve().parents[2]
HF_BASE = REPO / ".cache/models/nemotron-3-nano-omni-30b-a3b-reasoning-bf16"
LLM = "*language_model.decoder.layers.*."
TARGETS = [LLM + m for m in ("self_attention.linear_qkv", "self_attention.linear_proj", "mixer.in_proj",
                             "mixer.out_proj", "mlp.shared_experts.linear_fc1", "mlp.shared_experts.linear_fc2")]

# DirectSFTDataset repeats its rows up to the requested sample count (iterations x global batch), so the random sampler
# treats the whole run as one epoch and data-parallel ranks can read the same rows in lockstep, which leaves far fewer
# distinct images per batch. With its true length, the sampler reshuffles every epoch and the ranks read different rows.
DirectSFTDataset.__len__ = lambda self: len(self._base_examples)


def parquet(path):
    return HFDatasetSourceConfig(path_or_dataset="parquet", load_kwargs={"data_files": str(path)}, split="train")


def build(args):
    recipe._DEFAULT_HF_PATH = str(HF_BASE)
    cfg = recipe.nemotron_omni_cord_v2_peft_4gpu_h100_bf16_config()
    out, data = Path(args.out), Path(args.out) / "data"
    sizes = json.loads((data / "sizes.json").read_text())

    cfg.dataset.source = parquet(data / "train.parquet")
    cfg.dataset.validation_source = parquet(data / "val.parquet")
    cfg.dataset.test_source, cfg.dataset.do_test = None, False
    cfg.dataset.hf_processor_path = str(HF_BASE)
    cfg.dataset.trust_remote_code = True
    cfg.dataset.seq_length = cfg.model.seq_length = args.seq_length
    cfg.dataset.num_workers = 4
    cfg.dataset.data_sharding = False

    cfg.peft.target_modules = TARGETS
    cfg.peft.dim, cfg.peft.alpha, cfg.peft.dropout = args.lora_r, 2 * args.lora_r, 0.0
    cfg.peft.lora_A_init_method = args.lora_init

    cfg.model.tensor_model_parallel_size, cfg.model.sequence_parallel = 1, False
    cfg.model.expert_model_parallel_size = args.ep

    # The routers are frozen, so the 1e-4 load-balancing loss could only nudge hidden states through the adapters;
    # without it the objective is the plain answer-token cross-entropy.
    cfg.model.moe_aux_loss_coeff = 0.0
    cfg.model.moe_router_bias_update_rate = 0.0

    per_epoch = sizes["train"] // args.batch
    iters = per_epoch * args.epochs if args.max_steps < 0 else args.max_steps
    cfg.train.global_batch_size, cfg.train.micro_batch_size, cfg.train.train_iters = args.batch, 1, iters
    cfg.optimizer.lr, cfg.optimizer.min_lr = args.lr, args.min_lr
    cfg.scheduler.lr_warmup_iters, cfg.scheduler.lr_decay_iters = round(args.warmup * iters), iters
    cfg.validation.eval_interval = min(per_epoch, iters)
    cfg.validation.eval_iters = math.ceil(sizes["val"] / args.batch)
    cfg.rng.seed = args.seed

    cfg.checkpoint.pretrained_checkpoint = str(Path(args.base).resolve())
    cfg.checkpoint.finetune = True
    cfg.checkpoint.save = cfg.checkpoint.load = str(out.resolve() / "checkpoints")
    cfg.checkpoint.save_interval = min(per_epoch, iters)
    cfg.logger.log_interval = 1
    cfg.logger.tensorboard_dir = str(out.resolve() / "tb")
    return process_config_with_overrides(cfg, cli_overrides=args.overrides or None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Megatron-Bridge LoRA for Nemotron 3 Nano Omni on GoodsAD")
    ap.add_argument("--out", required=True, help="run dir; prepare.py's Parquet files must be in <out>/data")
    ap.add_argument("--base", required=True, help="Megatron checkpoint imported from the HF base model")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=float, default=0.03, help="share of steps")
    ap.add_argument("--batch", type=int, default=24, help="global batch: images per optimizer step")
    ap.add_argument("--lora-r", type=int, default=16, help="alpha = 2 x rank")
    ap.add_argument("--lora-init", choices=("xavier", "kaiming"), default="xavier",
                    help="LoRA A init: Megatron-Bridge's xavier-normal, or kaiming = PEFT's kaiming-uniform")
    ap.add_argument("--ep", type=int, default=2, help="expert parallel size; must divide 128 and the GPU count")
    ap.add_argument("--seq-length", type=int, default=4096)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("overrides", nargs="*", help="extra ConfigContainer overrides (not peft.*), e.g. model.recompute_granularity=full")
    args = ap.parse_args()
    cfg = build(args)
    apply_environment_variables(cfg)
    finetune(config=cfg, forward_step_func=forward_step)
