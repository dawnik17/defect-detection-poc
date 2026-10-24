# Serve

```bash
serve/serve.sh e2b                                # detached container poc, GPU 0, port 8000
docker logs -f poc                                # follow loading; Ctrl+C stops following, not the server
curl -fsS http://127.0.0.1:8000/v1/models         # answers once the server is up
serve/serve.sh stop                               # remove the container named $NAME (default: poc)
```

Pick the name, GPUs and port for each extra server, and pass the same `NAME` to stop it:

```bash
NAME=e4b GPU_IDS=1 PORT=8001 serve/serve.sh e4b
NAME=31b GPU_IDS=4,5 TP=2 PORT=8003 NIM_CACHE_DIR=.cache/nim serve/serve.sh 31b
NAME=e2b-lora MODEL=runs/train/lora/model GPU_IDS=6 PORT=8004 serve/serve.sh e2b
NAME=e4b serve/serve.sh stop
```
