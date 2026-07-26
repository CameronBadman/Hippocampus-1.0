import hashlib
import urllib.request

RUN_SPEC = {
    "experiment_id": "REC-pooled-s1802-6k",
    "model": "pooled",
    "seed": 1802,
}
WORKER_URL = (
    "https://raw.githubusercontent.com/CameronBadman/Hippocampus-1.0/"
    "35c013299fca94a5675056f97bd7569554ff9c6e/"
    "scripts/colab_spider_v0_2_recurrence_worker.py"
)
EXPECTED_SHA256 = (
    "6e1a99a7eea44a0f1b8484ba3d13d08e76b57b1d0a91e681999e5b4ef4affdac"
)
source = urllib.request.urlopen(WORKER_URL, timeout=60).read()
if hashlib.sha256(source).hexdigest() != EXPECTED_SHA256:
    raise RuntimeError("recurrence worker hash mismatch")
exec(compile(source, WORKER_URL, "exec"), globals())
