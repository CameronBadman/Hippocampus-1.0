"""Select the frozen recurrent seed-1701 Colab replication."""

from pathlib import Path


Path("/content/SPIDER_RUN_SPEC.json").write_text(
    '{"experiment_id":"L-E004-recurrent-s1701-5k",'
    '"model_family":"recurrent","seed":1701}\n'
)
