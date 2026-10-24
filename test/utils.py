"""OpenAI-compatible API calls for test.py."""
import json
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def api(url, body=None, timeout=300):
    headers = {"Content-Type": "application/json"}
    if os.getenv("API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["API_KEY"]
    request = Request(url, json.dumps(body).encode() if body is not None else None, headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read(4096).decode(errors='replace')}") from None


def model_id(url):
    models = api(url.rstrip("/") + "/v1/models")["data"]
    if len(models) != 1:
        raise ValueError("Endpoint has multiple models; specify --model.")
    return models[0]["id"]

