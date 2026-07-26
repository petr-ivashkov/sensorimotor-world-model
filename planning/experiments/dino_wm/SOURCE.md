# DINO-WM source provenance

The model uses the DINO-WM classes shipped by the locked dependency:

- package: `stable-worldmodel==0.0.6`
- module: `stable_worldmodel.wm.prejepa`
- installed module SHA-256:
  `58269b7d363c7ef552fc20d05f359bf630c2a4630d55b578ef8a70d76f2b2d14`

The training objective and defaults were verified against the maintained
reference implementation:

- repository: `https://github.com/galilai-group/stable-worldmodel`
- inspected commit: `0baacc8118c5f262aeff942da75cf191e71d3c5f`
- script: `scripts/train/prejepa.py`
- config: `scripts/train/config/prejepa.yaml`

The maintained implementation cites the original DINO-WM code:

- repository: `https://github.com/gaoyuezhou/dino_wm`
- inspected commit: `0a9492fa12044b852ae9e001cc74604b79c8bb0c`

The active objective is the shipped teacher-forcing MSE between the predicted
and target non-action embeddings. DINOv2-Small is frozen. The predictor,
action encoder, and state encoder are trainable.

The Hugging Face backbone is pinned to revision
`ed25f3a31f01632728cabb09d1542f84ab7b0056`.

Adaptations are limited to the controlled comparison:

- `H=1` rather than DINO-WM's native `H=3`;
- the shared 10-epoch, effective-batch-256 optimization protocol, implemented
  as eight mean-reduced micro-batches of 32;
- the shared optimizer-update-indexed learning-rate, validation, and logging
  schedules;
- the same full HDF5 training split and held-out validation split;
- the same CEM planner, tasks, seeds, and evaluation budget as `planning_eval`;
- repository-standard strict checkpoints.

TwoRoom and Push-T use the standard proprioception and action encoders. Reacher
uses its corresponding `observation` vector. OGBench-Cube follows the maintained
action-only DINO-WM configuration because its planning interface does not expose
a matching goal observation.
