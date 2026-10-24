#!/usr/bin/env bash
# Start a vLLM server in Docker for one model, and return once it answers requests.
#   GPU_IDS=0 PORT=8100 perf/server.sh e2b                          Gemma 4 E2B with the production flags below
#   GPU_IDS=0 PORT=8100 perf/server.sh e2b --max-num-seqs 128       extra vLLM flags override those flags
#   MODEL=runs/train/lora/model perf/server.sh e2b                  a fine-tuned checkpoint, with E2B's flags
#   docker logs -f perf                                             follow the server's log
#   docker rm -f perf                                               stop the server
set -euo pipefail
cd "$(dirname "$0")/.."   # the repo root

model=$1; shift           # everything after the model is extra vLLM flags
name=${NAME:-perf} gpus=${GPU_IDS:-0} port=${PORT:-8100} image=${IMAGE:-vllm-prof:0.31.0}

# vLLM keeps the last value of a repeated flag, so the extra flags win.
flags=(--served-model-name "$model" --max-model-len 32768 --max-num-seqs 128 --max-num-batched-tokens 8192 --kv-sharing-fast-prefill
       --limit-mm-per-prompt '{"image":6,"audio":0,"video":0}' --no-enable-prefix-caching --mm-processor-cache-gb 0)
case $model in
  e2b)      weights=google/gemma-4-E2B-it
            flags+=(--mm-processor-kwargs '{"max_soft_tokens":560}') ;;
  e4b)      weights=google/gemma-4-E4B-it
            flags+=(--mm-processor-kwargs '{"max_soft_tokens":560}') ;;
  nemotron) weights=$PWD/.cache/models/nemotron-3-nano-omni-30b-a3b-reasoning-bf16
            flags+=(--trust-remote-code --reasoning-parser nemotron_v3) ;;
  *)        echo "Unknown model '$model': use e2b, e4b or nemotron" >&2; exit 1 ;;
esac
weights=${MODEL:-$weights}
command=(vllm serve "$weights" --host 0.0.0.0 --port 8000 "${flags[@]}" "$@")

# Profiling (nsys/profile.sh sets NSYS_DIR): run the server under Nsight Systems. It records nothing until vLLM's
# /start_profile endpoint is called; at /stop_profile it stops the server and writes $NSYS_DIR/server.nsys-rep.
if [[ -n ${NSYS_DIR:-} ]]; then
  mkdir -p "$NSYS_DIR"
  command=(nsys profile -o "$NSYS_DIR/server" "--trace=cuda,nvtx,osrt" --cuda-graph-trace=node
           --trace-fork-before-exec=true --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown
           "${command[@]}" --profiler-config '{"profiler":"cuda"}')
fi

# The container sees the repo at the same path as the host does, and keeps model downloads and compile caches in
# .cache/, so a restart neither downloads nor compiles again. The inner quotes let GPU_IDS be a list such as 0,1;
# --entrypoint "" runs our command instead of the image's own "vllm serve". CUSTOM_VISION and CUSTOM_ATTENTION
# are the kernels image's switches (kernels/README.md), passed on only when set.
docker run -d --name "$name" --gpus "\"device=$gpus\"" --ipc=host -p "127.0.0.1:$port:8000" \
  -v "$PWD:$PWD" -w "$PWD" -e HF_TOKEN -e HF_HOME="$PWD/.cache/hf" -e CUSTOM_VISION -e CUSTOM_ATTENTION \
  -e VLLM_CACHE_ROOT="$PWD/.cache/perf/vllm" -e TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/perf/inductor" \
  --entrypoint "" "$image" "${command[@]}" > /dev/null

echo "Starting $model in container $name on GPU $gpus; loading and compiling take 1-5 minutes"
until curl -fs "http://127.0.0.1:$port/v1/models" > /dev/null; do
  if [[ $(docker inspect -f '{{.State.Running}}' "$name") != true ]]; then
    docker logs --tail 30 "$name"; echo "The server stopped during startup; its log is above" >&2; exit 1
  fi
  sleep 5
done
echo "READY http://127.0.0.1:$port/v1"
