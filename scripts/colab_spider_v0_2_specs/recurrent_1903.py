import hashlib
import urllib.request

RUN_SPEC = {
    "experiment_id": "REC-recurrent-s1903-6k",
    "model": "recurrent",
    "seed": 1903,
}
WORKER_URL = (
    "https://raw.githubusercontent.com/CameronBadman/Hippocampus-1.0/"
    "c4814ed7ac3073724b8cc23d33b25dacae6c46a6/"
    "scripts/colab_spider_v0_2_recurrence_worker.py"
)
EXPECTED_SHA256 = (
    "0bb62d1491f75bf604c0c8353dd93472dcc5cd91df10e9e151e4d929094ed9c8"
)
source = urllib.request.urlopen(WORKER_URL, timeout=60).read()
if hashlib.sha256(source).hexdigest() != EXPECTED_SHA256:
    raise RuntimeError("recurrence worker hash mismatch")
exec(compile(source, WORKER_URL, "exec"), globals())
