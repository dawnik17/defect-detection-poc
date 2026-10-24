# Defect Detection POC

Retail packaging-defect classification on the GoodsAD dataset with vision-language models (Gemma 4, Nemotron 3 Nano Omni), covering evaluation, fine-tuning, serving, benchmarking, profiling and kernel optimization.

## Components

```
test/              zero-shot vs few-shot evaluation suite: prompts, labels, product split and metrics
train/gemma/       Gemma 4 E2B fine-tuning (LoRA or full) with Hugging Face Transformers
train/nemotron/    Nemotron 3 Nano Omni LoRA fine-tuning with Megatron-Bridge
serve/             vLLM and NIM inference servers in Docker
perf/aiperf/       throughput and latency benchmarking with AIPerf
perf/nsys/         server profiling with Nsight Systems
kernels/           vLLM plugin with a faster Gemma 4 vision encoder
```

## NVIDIA tools

| Tool | Used for | Link |
|---|---|---|
| Nemotron 3 Nano Omni | multimodal model under evaluation and fine-tuning | [Hugging Face](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16) |
| NIM | optimized inference containers (`serve/`) | [docs](https://docs.nvidia.com/nim/) |
| NeMo Framework | training container (`nvcr.io/nvidia/nemo:26.08`) | [GitHub](https://github.com/NVIDIA/NeMo), [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo) |
| Megatron-Bridge | Nemotron LoRA training and checkpoint conversion (`train/nemotron/`) | [GitHub](https://github.com/NVIDIA-NeMo/Megatron-Bridge) |
| AIPerf | load generation and latency/throughput metrics (`perf/aiperf/`) | [GitHub](https://github.com/ai-dynamo/aiperf) |
| Nsight Systems | system-wide GPU/CPU tracing of the server (`perf/nsys/`) | [docs](https://developer.nvidia.com/nsight-systems) |
| cuDNN | attention kernel for the vision encoder (`kernels/`) | [docs](https://developer.nvidia.com/cudnn) |

Other libraries: [vLLM](https://github.com/vllm-project/vllm), [Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft).

Run every command from this folder.
