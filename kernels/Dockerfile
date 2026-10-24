FROM vllm-prof:0.31.0
COPY pyproject.toml /opt/custom-kernels/
COPY custom_kernels /opt/custom-kernels/custom_kernels
RUN pip install --no-cache-dir --no-deps --no-build-isolation /opt/custom-kernels
ENV VLLM_PLUGINS=custom_kernels
