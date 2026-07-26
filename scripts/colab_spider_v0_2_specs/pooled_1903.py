import hashlib
import urllib.request

RUN_SPEC = {
    "experiment_id": "REC-pooled-s1903-6k",
    "model": "pooled",
    "seed": 1903,
}
WORKER_URL = (
    "https://raw.githubusercontent.com/CameronBadman/Hippocampus-1.0/"
    "33b7e63ac3a5082b69a07e4a262c17cebef5164a/"
    "scripts/colab_spider_v0_2_recurrence_worker.py"
)
EXPECTED_SHA256 = (
    "8c0759ff0687f2e13fbd9476d8005ab777d16e33eb1b1dae213198a81dad4274"
)
source = urllib.request.urlopen(WORKER_URL, timeout=60).read()
if hashlib.sha256(source).hexdigest() != EXPECTED_SHA256:
    raise RuntimeError("recurrence worker hash mismatch")
exec(compile(source, WORKER_URL, "exec"), globals())
