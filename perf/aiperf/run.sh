#!/usr/bin/env bash
# Benchmark one model end to end: start its server, load it with AIPerf, stop it.
#   GPU_IDS=0 perf/aiperf/run.sh <run name> <model> [extra vLLM flags...]     results in runs/perf/<run name>/
#   GPU_IDS=0 perf/aiperf/run.sh e2b e2b
#   GPU_IDS=1 perf/aiperf/run.sh e2b-seqs128 e2b --max-num-seqs 128
set -euo pipefail
perf=$(realpath "$(dirname "$0")/..")   # this perf folder
cd "$perf/.."                           # the repo root

name=$1 model=$2; shift 2               # everything after the model is extra vLLM flags
run=runs/perf/$name container=perf-$name
gpus=${GPU_IDS:-0}
port=${PORT:-$((8100 + ${gpus%%,*}))}   # 8100 + the first GPU id, so runs on different GPUs get different ports
url=http://127.0.0.1:$port
mkdir -p "$run"

# On exit, even after an error or Ctrl+C: keep the server's log, then remove the server.
trap 'docker logs "$container" > "$run/server.log" 2>&1; docker rm -f "$container" > /dev/null 2>&1' EXIT

# 1. Start the server, and record exactly how it runs: image, flags and GPUs.
NAME=$container GPU_IDS=$gpus PORT=$port "$perf/server.sh" "$model" "$@"
docker inspect "$container" > "$run/container.json"

# 2. Load it with AIPerf: every use case at every concurrency.
URL=$url GPU_IDS=$gpus "$perf/aiperf/bench.sh" "$run"
