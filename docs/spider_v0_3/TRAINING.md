# Spider v0.3 training protocol

Spider v0.3 isolates evidence selection first, then termination. The v0.2
source, six accepted runs, reports, tags, and Drive folder are immutable. No
v0/v0.2 sealed case is available to these commands.

## Evidence matrix

The registered development data has 512 training cases, 64 grouped
calibration cases, and 64 disjoint grouped evaluation cases. Its aggregate
SHA-256 is
`0ed8e27ec44f3773f76b79f1947526f33ba233556b7db91fef04dcb647e5409d`.
Calibration chooses one global evidence threshold; development evaluation is
read only after that choice.

The three evidence arms are:

- E0: preserved class-balanced evidence BCE and set-recall objective.
- E1: E0 plus a multi-positive hard-negative ranking loss.
- E2: E1 plus a separate loss over mined structurally plausible negatives.

The screen is E0/E1/E2 for 1,000 steps at seeds 1701, 1802, and 1903. E0 and
the best experimental arm that passes the registered matched-seed gate advance
to 6,000 total steps. Full runs resume the exact 1,000-step model, optimiser,
RNG, data-order, and rollout state.

Run locally with:

```bash
.venv/bin/python scripts/run_spider_v0_3_autoresearch.py \
  --phase all \
  --output-root artifacts/spider_v0_3/evidence
```

## One-session A100 execution

The production research run uses one named Better Colab session and exactly
one A100. First verify the client:

```bash
better-colab doctor --format json
better-colab capabilities --format json
```

After committing and pushing the exact source, allocate and launch:

```bash
better-colab session ensure spider-v03-evidence-a100 \
  --gpu A100 --format json

.venv/bin/python scripts/render_spider_v0_3_colab_launch.py |
  better-colab execution start \
    --session spider-v03-evidence-a100 \
    --idempotency-key spider-v03-evidence-$(git rev-parse --short=12 HEAD) \
    --detach --format json
```

Record the returned execution ID and observe it in bounded calls:

```bash
better-colab execution wait EXECUTION_ID \
  --timeout 60 --max-bytes 65536 --format json
```

A wait timeout is observational and does not cancel training. Do not submit
changed source under the same idempotency key.

The worker mounts Google Drive and mirrors every file only after it is stable
across two observations. Final shutdown forces one last verified copy. The
same single A100 session runs the evidence matrix and then the T0/T1/T2
termination matrix. The destination is:

```text
My Drive/Hippocampus-1.0/Spider-v0.3-Evidence/<source-commit-prefix>/
```

Each copied checkpoint is hashed locally and on Drive. The final
`GOOGLE_DRIVE_CHECKPOINTS.json` records every path, byte count, and SHA-256.
This is a new folder; the v0.2 folder is never read or written.

## Interrupted-session continuation

Launch the same pushed commit with the same generated cell in a replacement
A100 session. The worker restores the commit-keyed Drive mirror. Completed
runs are reused only after source-hash validation. An incomplete run is moved
to a numbered `recovery/` directory, never overwritten, and resumed from its
latest `checkpoint.pt`. If no checkpoint exists, its partial record remains
archived and that one run restarts.

Before stopping a session, observe or cancel its active execution deliberately,
then run:

```bash
better-colab session stop spider-v03-evidence-a100 --format json
```

## Advancement and termination

The evidence change advances only with at least `+0.05` recall or `+0.03`
exact-set accuracy on two of three matched seeds, precision loss no worse than
`0.02`, and no unexplained scorer-coverage loss.

After freezing the winning evidence checkpoint and threshold, termination
training uses direct oracle-state labels for evidence sufficiency, useful work
remaining, answer support, and unknown reason. Traversal and evidence scoring
remain frozen for T0–T2. T2 adds a per-hypothesis NULL action; NULL kills one
branch and never implies global termination.
