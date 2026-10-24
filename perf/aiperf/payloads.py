"""
Freeze the requests and images AIPerf sends, built with test/test.py's own prompt and image code.

    python perf/aiperf/payloads.py requests             # the 419 zero-shot test requests, as test.py sends them
    python perf/aiperf/payloads.py requests --side 768  # the same requests, images resized by the client to 768 px
    python perf/aiperf/payloads.py images               # their images as JPEG files, FOR SYNTHETIC USEC ASE

"""
import argparse
import base64
import importlib.util
import json
import random
from pathlib import Path

# test/test.py, loaded by its path: `import test` would load Python's own test package.
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("goodsad_suite", ROOT / "test" / "test.py")
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


def test_images():
    """(category, label, path) of the 419 held-out test images that test.py scores."""
    return [(cat, label, path) for cat in suite.CATS for label, path in suite.samples(cat, n=0)]


def requests(side):
    suite.MAX_SIDE = side
    bodies = [{"messages": suite.messages(cat, path), "temperature": 0,
               "max_tokens": 128,  # a cap only: replies end after about 5 to 50 tokens
               "stream": True, "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": False}}
              for cat, label, path in test_images()]
    # A fixed shuffle gives every window of requests the same mix of categories; AIPerf replays them in order.
    random.Random(0).shuffle(bodies)

    name = "goodsad-zero" if side == 2048 else f"goodsad-zero-{side}"
    out = ROOT / "runs/perf/payloads" / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for body in bodies:
            f.write(json.dumps(body, ensure_ascii=False) + "\n")
    print(f"{len(bodies)} requests, {out.stat().st_size / 2**20:.0f} MiB -> {out}")


# Used for synthetic use cases
def images():
    out = ROOT / "runs/perf/images"
    out.mkdir(parents=True, exist_ok=True)
    items = test_images()
    for cat, label, path in items:
        data = suite.uri(Path(path), suite.MAX_SIDE).split(",", 1)[1]  # test.py's JPEG, base64 after the comma
        (out / f"{cat}-{label}-{Path(path).stem}.jpg").write_bytes(base64.b64decode(data))
    print(f"{len(items)} images at {suite.MAX_SIDE} px -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("requests", help="request bodies for AIPerf's raw-payload replay")
    r.add_argument("--side", type=int, default=2048, help="longest image side the client sends (test.py: 2048)")
    sub.add_parser("images", help="the request images as JPEG files")
    args = ap.parse_args()
    if args.command == "requests":
        requests(args.side)
    else:
        images()
