# Custom kernels

```bash
docker build -t vllm-custom-kernels:0.31.0 kernels      # again after any change here
IMAGE=vllm-custom-kernels:0.31.0 MODEL=runs/train/lora/model GPU_IDS=0 perf/aiperf/run.sh <run> e2b \
  --max-num-seqs 128 --max-num-batched-tokens 8192 --api-server-count 4 --kv-sharing-fast-prefill
```

The image is the vLLM 0.31 image plus `pip install` of this package and `VLLM_PLUGINS=custom_kernels`
(`kernels/Dockerfile`). `CUSTOM_VISION` selects the encoder: `compile` (default) runs the layers under torch.compile,
`eager` does not. `CUSTOM_ATTENTION=cudnn` lets the encoder's attention use cuDNN, which vLLM switches off for every
model; on Blackwell its kernel is faster than FlashAttention 2. `perf/server.sh` passes both to the container when set.
The same image works with `serve/serve.sh`'s flags.
