import hashlib
import urllib.request

RUN_SPEC = {
    "experiment_id": "REC-pooled-s1903-6k",
    "model": "pooled",
    "seed": 1903,
}
WORKER_URL = (
    "https://raw.githubusercontent.com/CameronBadman/Hippocampus-1.0/"
    "611dd9ab11b2a48cc69e9338f5321a73f24ac1d2/"
    "scripts/colab_spider_v0_2_recurrence_worker.py"
)
EXPECTED_SHA256 = (
    "09a520066331553d8597b789f29838e16f56f7ef851787a92a7610e3b67cef60"
)
source = urllib.request.urlopen(WORKER_URL, timeout=60).read()
if hashlib.sha256(source).hexdigest() != EXPECTED_SHA256:
    raise RuntimeError("recurrence worker hash mismatch")
exec(compile(source, WORKER_URL, "exec"), globals())
