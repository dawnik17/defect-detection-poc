## LoRA fine-tune Nemotron 3 Nano Omni with Megatron-Bridge

Starts from NVIDIA's Omni LoRA recipe (nemotron_omni_cord_v2_peft_config: rank 16, alpha 32, Adam with betas 0.9 and
0.95, LR 1e-4 cosine to 1e-5, bf16) and changes only what this task needs:

- data: the Parquet conversations from prepare.py instead of CORD-v2, with a val split, sampled as real shuffled epochs.
- LoRA targets: the language model's attention, Mamba and shared-expert projections. The recipe's bare names would
  also put adapters on the vision encoder and both projectors, but this task only needs the LM.
- schedule: global batch 24, 3 epochs, 3% warmup, a checkpoint at the end of every epoch.
- routers truly frozen: no MoE load-balancing loss and no expert-bias updates, which Megatron applies outside PEFT.

```bash
# once: import the HF base model into a Megatron checkpoint
python -c "import torch; from megatron.bridge import AutoBridge; AutoBridge.import_ckpt(
    '.cache/models/nemotron-3-nano-omni-30b-a3b-reasoning-bf16', '.cache/megatron/nemotron-3-nano-omni-30b-a3b-reasoning-bf16',
    trust_remote_code=True, torch_dtype=torch.bfloat16)"

# prepare data
python train/nemotron/prepare.py --out runs/train/nemotron-megatron/data

# train
torchrun --nproc_per_node 8 train/nemotron/train.py --out runs/train/nemotron-megatron \
    --base .cache/megatron/nemotron-3-nano-omni-30b-a3b-reasoning-bf16

# export the last checkpoint to a PEFT adapter (--ep and --nproc_per_node = train.py --ep), then merge it
torchrun --nproc_per_node 2 /opt/Megatron-Bridge/examples/conversion/adapter/export_adapter.py \
    --hf-model-path .cache/models/nemotron-3-nano-omni-30b-a3b-reasoning-bf16 --trust-remote-code \
    --lora-checkpoint runs/train/nemotron-megatron/checkpoints/iter_<N> \
    --output runs/train/nemotron-megatron/adapter --ep 2 --dtype bfloat16
python train/nemotron/merge.py --adapter runs/train/nemotron-megatron/adapter --out runs/train/nemotron-megatron/model
```
