#!/usr/bin/env bash
# Profile one model with Nsight Systems while AIPerf loads it; the trace and its reports go to runs/perf/<run>/nsys/.
#   GPU_IDS=0 perf/nsys/profile.sh <run name> <model> [extra vLLM flags...]
#   GPU_IDS=0 perf/nsys/profile.sh nsys-e2b e2b
#   CONCURRENCY=1 GPU_IDS=0 perf/nsys/profile.sh nsys-e2b-c1 e2b      one request at a time - deafult is 16
set -euo pipefail
perf=$(realpath "$(dirname "$0")/..")   # this perf folder
cd "$perf/.."                           # the repo root

name=$1 model=$2; shift 2               # everything after the model are vLLM flags
run=runs/perf/$name container=perf-$name
gpus=${GPU_IDS:-0}
port=${PORT:-$((8100 + ${gpus%%,*}))}   # runs on different GPUs get different ports
url=http://127.0.0.1:$port
mkdir -p "$run"

# On exit, even after an error or Ctrl+C: keep the server's log, then remove the server.
trap 'docker logs "$container" > "$run/server.log" 2>&1; docker rm -f "$container" > /dev/null 2>&1' EXIT

# 1. Start the server under Nsight Systems, which records nothing yet.
NAME=$container GPU_IDS=$gpus PORT=$port NSYS_DIR=$run/nsys "$perf/server.sh" "$model" "$@"
docker inspect "$container" > "$run/container.json"

# 2. Warm the server up with AIPerf load (not recorded), then record a fixed window of the same load.
export URL=$url GPU_IDS=$gpus CONCURRENCY=${CONCURRENCY:-16}
REQUESTS=64 "$perf/aiperf/bench.sh" "$run/warmup"
curl -fsS -X POST "$url/start_profile" > /dev/null
REQUESTS=${REQUESTS:-160} "$perf/aiperf/bench.sh" "$run"
curl -fs -X POST "$url/stop_profile" > /dev/null || true   # Nsight Systems stops the server here, so it may not reply
docker wait "$container" > /dev/null                       # ... and writes the trace before the container exits

# 3. Export the trace to SQLite plus summary tables (stats_*.csv), then break it into engine steps.
docker run --rm -v "$PWD:$PWD" -w "$PWD" --entrypoint nsys vllm-prof:0.31.0 stats -q --force-export=true \
  --format csv --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,nvtx_sum,osrt_sum \
  --output "$run/nsys/stats" "$run/nsys/server.nsys-rep" > "$run/nsys/stats.log" 2>&1
python "$perf/nsys/steps.py" "$run/nsys/server.sqlite" | tee "$run/nsys/steps.txt"
