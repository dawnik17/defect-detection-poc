# Serving performance

- **Benchmarking with AIPerf** (`aiperf/`): throughput and latency over a sweep of load.
- **Profiling with Nsight Systems** (`nsys/`): every CUDA call and GPU kernel inside the server under that same load,
broken down into vLLM's engine steps.

Both tools start the server with `server.sh`, so a profile sees exactly the configuration a benchmark measured.
Results go to `runs/perf/<run>/`. Run every command from the repo root.

```
perf/
  server.sh     start a vLLM server in Docker for one model; both tools use it
  Dockerfile    the serving image: vLLM 0.31.0 plus Nsight Systems
  aiperf/       benchmarking with AIPerf
  nsys/         profiling with Nsight Systems
```

### One time setup

```bash
docker build -t vllm-prof:0.31.0 perf                  # the serving image
docker build -t perf-aiperf:0.13.0 perf/aiperf         # the AIPerf client

python perf/aiperf/payloads.py requests                        # freeze the 419 GoodsAD test requests
python perf/aiperf/payloads.py images                          # their images, for synthetic use cases

hf download nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 --revision a5fba9ed825b7a641606d97c8f189ce01e3a7cf0 \
  --local-dir .cache/models/nemotron-3-nano-omni-30b-a3b-reasoning-bf16   # for nemotron only: 62 GB
```

### Custom Kernels

Everything above is stock vLLM. `kernels/` holds a vLLM plugin that lives in its own image, built on the serving image. Build that image, pass it to either tool as `IMAGE`, and compare with the stock run of the same model.

```bash
docker build -t vllm-custom-kernels:0.31.0 kernels    # once, and again after any change in kernels/
IMAGE=vllm-custom-kernels:0.31.0 GPU_IDS=0 perf/aiperf/run.sh e2b-kernels e2b
IMAGE=vllm-custom-kernels:0.31.0 GPU_IDS=1 perf/nsys/profile.sh nsys-e2b-kernels e2b
```
