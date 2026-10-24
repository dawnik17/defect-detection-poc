# The serving image, vllm-prof:0.31.0: vLLM 0.31.0 plus Nsight Systems
FROM vllm/vllm-openai:v0.31.0
RUN apt-get update \
 && apt-get install -y --no-install-recommends nsight-systems-2026.3.2 \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /opt/nvidia/nsight-compute/2026.3.1/ncu /usr/local/bin/ncu
