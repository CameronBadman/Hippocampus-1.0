"""Select the frozen pooled seed-1802 Colab replication."""

from pathlib import Path


Path("/content/SPIDER_RUN_SPEC.json").write_text(
    '{"experiment_id":"L-E005-pooled-s1802-5k",'
    '"model_family":"pooled","seed":1802}\n'
)
