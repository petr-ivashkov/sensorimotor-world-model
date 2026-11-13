# Planning

The core code is `jepa.py` (world model), `module.py` (networks), `train.py`
(training), and `eval.py` (CEM planning). Configurations cover TwoRoom, Reacher,
Push-T, and OGBench-Cube. Use the root [setup](../README.md#setup); training and
simulation use Linux with a CUDA GPU. Commands below start in `planning/`.

## Pretrained models

The root [quickstart](../README.md#pretrained-models-and-analysis) downloads
seed-0 embeddings for SMWM, SIGReg, and Forward-only in all four environments,
and the five-seed planning results for all six methods. The notebooks use
these files directly. Checkpoints live in
[`petr-ivashkov/smwm`](https://huggingface.co/petr-ivashkov/smwm), under
`<environment>/<method>/seed<0–4>/`, with `config.yaml`, `weights.pt`, and
`metrics.json`. Use another path in the quickstart's `allow_patterns` to select
a model, or omit the filter to download all checkpoints.

To encode the small TwoRoom image samples with the downloaded checkpoint:

```bash
python scripts/export_embeddings.py \
  --run-dir outputs/pretrained/tworoom/smwm/seed0 \
  --train-data outputs/samples/tworoom/train.h5 \
  --eval-data outputs/samples/tworoom/eval.h5 \
  --output-dir outputs/demo_embeddings --batch-size 32 --device cpu
```

Set `EXPORT_ROOT = PLANNING / "outputs/demo_embeddings"` in the embedding or
probe notebook to inspect this export. The samples contain 1,000 training and
200 evaluation images. For the full 25,000/5,000-image analysis, download
`images/<environment>/*` from the
[analysis dataset](https://huggingface.co/datasets/petr-ivashkov/smwm-analysis)
and pass those files instead. Use `load_jepa_from_run` in `eval.py` to load any
released model; planning evaluation also accepts these checkpoint directories.

## Data

Download the expert datasets below, decompress them, and place the HDF5 files
in `data/external/` using the source names in the last column.

| Environment | Download | Source name |
|---|---|---|
| TwoRoom | [tworoom.tar.zst](https://huggingface.co/datasets/quentinll/lewm-tworooms/resolve/6903a2de048b13819d812da0b4dd661290bc01e4/tworoom.tar.zst) | `tworoom.h5` |
| Reacher | [reacher.tar.zst](https://huggingface.co/datasets/quentinll/lewm-reacher/resolve/e70a080d0d04c6072123c9ebd343acf7fff28dbf/reacher.tar.zst) | `reacher.h5` |
| Push-T | [pusht_expert_train.h5.zst](https://huggingface.co/datasets/quentinll/lewm-pusht/resolve/655cd446b9929369d7d406001da85c15d1457850/pusht_expert_train.h5.zst) | `pusht_expert.h5` |
| OGBench-Cube | [cube_single_expert.tar.zst](https://huggingface.co/datasets/quentinll/lewm-cube/resolve/02a19a67a0dc8c9d6215f89c19e0a597691e152a/cube_single_expert.tar.zst) | `cube_single_expert.h5` |

For `.tar.zst`, use `tar --zstd -xf archive.tar.zst`; for Push-T, use
`zstd -d pusht_expert_train.h5.zst -o pusht_expert.h5`. Then prepare the
episode-level 90/10 split (seed 3072):

```bash
python scripts/make_episode_splits.py
```

Set `EXTERNAL_DATA_ROOT` to use another data directory. Only downloaded
environments are processed; existing split files are preserved.

## Performance comparison

```bash
python train.py data=tworoom method=smwm seed=0
python eval.py env=tworoom run_dir=outputs/train/tworoom_smwm_seed0
```

Use `data=` / `env=` with `tworoom`, `reacher`, `pusht`, or `ogbcube`.
Methods are `smwm`, `sigreg`, and `forward_only`; [PLDM and DINO-WM](baselines/README.md)
have separate training adapters and use the same evaluation script.
Random needs no checkpoint: `python eval.py env=tworoom policy=random repeat=0`.

For the paper comparison, use training seeds 0–4 for each learned method and
Random repeats 0–4. Training uses 10 epochs, effective batch size 256, history 1,
inverse weight 10 for SMWM, and SIGReg weight 0.09. Evaluation uses 100 tasks,
goal offset 25, and a 50-step budget. CEM uses 300 samples, 30 iterations, and
30 elites, with a horizon of five blocks of five actions. Task and policy seeds
are fixed in the environment configs.

Checkpoints, resolved settings, and loss logs go to `outputs/train/`; episode
outcomes go to `outputs/eval/`. The notebook
[`planning_results.ipynb`](analysis/planning_results.ipynb) reports mean ± sample
STD across runs and plots the six-method comparison. Outputs stay outside Git.

## Embeddings and physical probes

Prepare the analysis samples and export embeddings for seed 0 of each
environment/method (SMWM, SIGReg, and Forward-only):

```bash
python scripts/create_randomized_transition_datasets.py --train-n 25000 --eval-n 5000
python scripts/export_embeddings.py --run-dir outputs/train/tworoom_smwm_seed0
```

Open [`embeddings.ipynb`](analysis/embeddings.ipynb) for PCA geometry and
[`physical_probes.ipynb`](analysis/physical_probes.ipynb) for linear/MLP decoding.
Both fit on training embeddings; probes report evaluation R². Exports live
under `outputs/embeddings/`; plots and tables go to `outputs/analysis/`.

Other paper settings can be explored by overriding `loss.inverse.weight`,
`loss.sigreg.weight`, `wm.history_size`, `eval.goal_offset_steps`, and
`eval.eval_budget`. Use distinct training `subdir` and evaluation `output_dir`
values for separate runs.

Adapted from [LeWorldModel](https://github.com/lucas-maes/le-wm), using
`stable-worldmodel` and `stable-pretraining`. See the
[third-party notices](../THIRD_PARTY_NOTICES.md).
