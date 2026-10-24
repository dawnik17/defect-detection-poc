# Profiling with Nsight Systems

Start the server under Nsight Systems, warm it up, record a fixed load window, export reports and analysis

**Run**

```bash
GPU_IDS=0 perf/nsys/profile.sh nsys-e2b e2b                      # -> runs/perf/nsys-e2b/nsys/
GPU_IDS=1 perf/nsys/profile.sh nsys-nemotron nemotron

GPU_IDS=5 perf/nsys/profile.sh nsys-e2b-tuned e2b --max-num-seqs 128 --max-num-batched-tokens 8192 \
  --api-server-count 4 --kv-sharing-fast-prefill

CONCURRENCY=1 GPU_IDS=2 perf/nsys/profile.sh nsys-e2b-c1 e2b     # one request at a time

python perf/nsys/steps.py runs/perf/nsys-e2b/nsys/server.sqlite  # rerun analysis on an existing trace
```
