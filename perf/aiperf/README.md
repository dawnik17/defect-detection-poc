# Benchmarking with AIPerf

**Run**

```bash
GPU_IDS=0 perf/aiperf/run.sh e2b e2b                             # serve, sweep, stop
GPU_IDS=1 perf/aiperf/run.sh e4b e4b                             # one model per GPU, side by side
GPU_IDS=2 perf/aiperf/run.sh nemotron nemotron
GPU_IDS=3 perf/aiperf/run.sh e2b-seqs128 e2b --max-num-seqs 128  # flags after the model, override its flags
```

`run.sh <run name> <model>` writes to `runs/perf/<run name>/`
