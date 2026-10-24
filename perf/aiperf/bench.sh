#!/usr/bin/env bash
# Load a running vLLM server with NVIDIA AIPerf, as in the NIM benchmarking guide: every use case at every
# concurrency, one AIPerf run per point, results in <out dir>/<use case>/CON<concurrency>/.
#   URL=http://127.0.0.1:8100 GPU_IDS=0 perf/aiperf/bench.sh runs/perf/<run>
#   URL=http://127.0.0.1:8100 USE_CASES="goodsad-zero img6-out384" CONCURRENCY="1 8" perf/aiperf/bench.sh <out dir>
set -euo pipefail
cd "$(dirname "$0")/../.."   # the repo root
out=$1 url=${URL:-http://127.0.0.1:8100}
mkdir -p "$out"

# Ask the server what it serves: the model name requests must use, and the checkpoint it loaded (a Hugging Face id
# such as google/gemma-4-E2B-it, or a local directory), whose tokenizer AIPerf uses for synthetic prompts.
model=$(curl -fs "$url/v1/models" | jq -r '.data[0].id')
tokenizer=$(curl -fs "$url/v1/models" | jq -r '.data[0].root')

# AIPerf runs in its own container (Dockerfile here), on the server's host so the network adds no latency. It sees
# the repo at the same path as the host does, and the server's GPUs, whose telemetry it records.
aiperf() {
  docker run --rm --network host --gpus "\"device=${GPU_IDS:-0}\"" -v "$PWD:$PWD" -w "$PWD" \
    -e HF_HOME="$PWD/.cache/hf" perf-aiperf:0.13.0 "$@"
}
max() { (( $1 > $2 )) && echo "$1" || echo "$2"; }

# Real requests: replay a file written by payloads.py, byte for byte, in file order.
replay=(--custom-dataset-type raw-payload --dataset-sampling-strategy sequential)
# Synthetic requests: 200 random text tokens plus GoodsAD images (AIPerf resizes them to 2048 x 2048 and sends JPEG),
# and exactly --osl output tokens (min_tokens with ignore_eos).
synthetic=(--isl 200 --image-source runs/perf/images --image-source-sampling shuffle-cycle
           --image-width-mean 2048 --image-height-mean 2048 --image-format jpeg
           --extra-inputs ignore_eos:true --extra-inputs '{"chat_template_kwargs":{"enable_thinking":false}}')

for case in ${USE_CASES:-goodsad-zero}; do
  case $case in
    goodsad-zero*) args=(--input-file "runs/perf/payloads/$case.jsonl" "${replay[@]}") ;;
    img1-out128)   args=("${synthetic[@]}" --image-batch-size 1 --osl 128 --extra-inputs min_tokens:128) ;;
    img6-out384)   args=("${synthetic[@]}" --image-batch-size 6 --osl 384 --extra-inputs min_tokens:384) ;;
    *)             echo "Unknown use case '$case'" >&2; exit 1 ;;
  esac

  for c in ${CONCURRENCY:-1 2 4 8 16 32 64 128}; do
    requests=${REQUESTS:-$(max 100 $((8 * c)))}   # at least 100 requests per point, for a stable p95
    log=$out/${case}_CON$c.log
    echo "$case, concurrency $c: $requests requests (log: $log)"
    aiperf profile -m "$model" --tokenizer "$tokenizer" -u "$url" --endpoint-type chat --streaming \
      --use-server-token-count "${args[@]}" --concurrency "$c" --request-count "$requests" \
      --warmup-request-count "$(max 4 "$c")" --num-profile-runs "${RUNS:-1}" --random-seed 42 \
      --gpu-telemetry pynvml --ui none --artifact-dir "$out/$case/CON$c" \
      > "$log" 2>&1 || echo "  AIPerf failed, see $log"
    sleep 5   # let the server go idle before the next point
  done
done
