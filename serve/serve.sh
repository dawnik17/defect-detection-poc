#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

name=${NAME:-poc}
if [[ ${1:-} == stop ]]; then
  docker rm -f "$name" 2>/dev/null || true
  exit
fi

nim_image=
case ${1:-} in
  e2b) model=google/gemma-4-E2B-it; image=vllm/vllm-openai:gemma4; seqs=4 ;;
  e4b) model=google/gemma-4-E4B-it; image=vllm/vllm-openai:gemma4; seqs=4 ;;
  12b) model=google/gemma-4-12B-it; image=vllm/vllm-openai:gemma4-unified-cu129; seqs=4 ;;
  31b) nim_image=nvcr.io/nim/google/gemma-4-31b-it:latest ;;
  nemotron) nim_image=nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:latest ;;
  nemotron-vllm) model=.cache/models/nemotron-3-nano-omni-30b-a3b-reasoning-bf16; image=vllm/vllm-openai:v0.31.0; seqs=4 ;;
  *) echo 'Usage: serve/serve.sh {e2b|e4b|12b|31b|nemotron|nemotron-vllm|stop}'; exit 2 ;;
esac

gpus=${GPU_IDS:-0}; port=${PORT:-8000}; tp=${TP:-1}
images=${MAX_IMAGES:-6}
opts=(--name "$name" --gpus "\"device=$gpus\"" --ipc=host
      -p "127.0.0.1:$port:8000")

if [[ -n $nim_image ]]; then
  : "${NGC_API_KEY:?Export NGC_API_KEY for NIM}"

  cache=${NIM_CACHE_DIR:-${LOCAL_NIM_CACHE:-$HOME/.cache/nim}}
  mkdir -p "$cache"
  cache=$(realpath "$cache")
  chmod -R a+rwX "$cache"
  printf '%s' "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin

  opts+=(--user "${NIM_USER:-0:0}")

  docker run -d "${opts[@]}" --shm-size=32GB -e NGC_API_KEY \
    -e "NIM_TENSOR_PARALLEL_SIZE=$tp" \
    -e "NIM_MAX_IMAGES_PER_PROMPT=$images" \
    -v "$cache:/opt/nim/.cache" "${NIM_IMAGE:-$nim_image}"
else
  cache=${CACHE_DIR:-$PWD/.cache}/hf
  mkdir -p "$cache"
  cache=$(realpath "$cache")
  flags=(--dtype bfloat16 --tensor-parallel-size "$tp"
         --max-model-len "${MAX_LEN:-32768}" --max-num-seqs "${MAX_SEQS:-$seqs}"
         --gpu-memory-utilization "${GPU_MEMORY:-0.90}"
         --limit-mm-per-prompt "{\"image\":$images,\"audio\":0,\"video\":0}"
         --no-enable-prefix-caching --mm-processor-cache-gb 0)
  if [[ $1 == e2b || $1 == e4b ]]; then
    flags+=(--mm-processor-kwargs "{\"max_soft_tokens\":${VISION_TOKENS:-560}}")
  fi
  if [[ $1 == nemotron-vllm ]]; then  # local checkpoint under .cache/models, mounted at the same path
    opts+=(-v "$(realpath .cache):$(realpath .cache):ro"); model=$(realpath "$model")
    flags+=(--trust-remote-code --reasoning-parser nemotron_v3)
  fi
  if [[ -n ${MODEL:-} ]]; then  # local checkpoint, e.g. runs/train/lora/model; served as $NAME
    opts+=(-v "$(realpath "$MODEL"):/model:ro"); model=/model; flags+=(--served-model-name "$name")
  fi

  docker run -d "${opts[@]}" -e HF_TOKEN \
    -v "$cache:/root/.cache/huggingface" "${VLLM_IMAGE:-$image}" \
    --model "$model" --host 0.0.0.0 --port 8000 "${flags[@]}"
fi

echo "Container started. Follow model loading: docker logs -f $name"
echo "API after loading: http://127.0.0.1:$port/v1"
