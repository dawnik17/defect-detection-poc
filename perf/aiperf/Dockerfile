# The AIPerf client image that bench.sh runs, perf-aiperf:0.13.0. requirements.txt pins AIPerf 0.13.0 and
# every dependency, so each build is the same.
FROM python:3.12-slim
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
ENTRYPOINT ["aiperf"]
