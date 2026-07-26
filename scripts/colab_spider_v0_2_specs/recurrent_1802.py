import hashlib
import urllib.request

RUN_SPEC = {
    "experiment_id": "REC-recurrent-s1802-6k",
    "model": "recurrent",
    "seed": 1802,
}
WORKER_URL = (
    "https://raw.githubusercontent.com/CameronBadman/Hippocampus-1.0/"
    "44dc478bb582224268e4f94a19b99f4681579b86/"
    "scripts/colab_spider_v0_2_recurrence_worker.py"
)
EXPECTED_SHA256 = (
    "1091d6672109a89f573992a959b10417579327da6278e4376ee76e813ce273e4"
)
source = urllib.request.urlopen(WORKER_URL, timeout=60).read()
if hashlib.sha256(source).hexdigest() != EXPECTED_SHA256:
    raise RuntimeError("recurrence worker hash mismatch")
exec(compile(source, WORKER_URL, "exec"), globals())
