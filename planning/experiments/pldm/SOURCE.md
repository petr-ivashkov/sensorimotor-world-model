# PLDM source provenance

The model components and loss implementation under `vendor/` are copied from:

- repository: `https://github.com/galilai-group/stable-worldmodel`
- commit: `0baacc8118c5f262aeff942da75cf191e71d3c5f`
- files:
  - `stable_worldmodel/wm/pldm/module.py`
  - `stable_worldmodel/wm/pldm/pldm.py`
  - `stable_worldmodel/wm/loss.py`

The port cites the original PLDM implementation:

- repository: `https://github.com/vladisai/PLDM`
- inspected commit: `1bd7e564ecd961205bc18b23067b19e9ca24ac90`

The training adapter preserves the shipped PLDM model and default objective. It
replaces dataset/checkpoint plumbing, sets predictor history to `H=1`, and uses
the shared 10-epoch, effective-batch-256 optimization protocol to match every
learned method in the main planning figure.

Default active loss:

```text
prediction + 18 std + 0.7 std_t + 12 cov + 0.2 temporal_alignment
```

The shipped default explicitly sets `cov_t=0`, `IDM=0`, `SIGReg=0`, and temporal
straightening to disabled. Its native predictor history is `H=3`; this experiment
uses the configurable implementation at `H=1`.
