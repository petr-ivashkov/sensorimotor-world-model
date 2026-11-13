# Toy worlds

Small CNN-based SMWMs on dot and sprite worlds. Install the root
[environment](../README.md#setup), then run from the repository root:

```bash
python toy/train.py --config toy/config/dot.yaml --smoke
```

Omit `--smoke` for the full configuration. Other examples are
[`structured.yaml`](config/structured.yaml) (coupled, independent, and random dots)
and [`sprite.yaml`](config/sprite.yaml) (translation and rotation).
Configs contain the dataset, network, loss, and optimizer settings.
Samples are generated on demand; set `dataset.num_workers` to parallelize rendering.
Results go to `toy/outputs/<run_name>/`; use a new `output.run_name` for a new run.

For frozen-encoder image decoding:

```bash
python toy/train.py --config toy/config/sprite.yaml --smoke
python toy/train_decoder.py --config toy/config/decoder.yaml --smoke
```

Open [`analyze.ipynb`](analyze.ipynb) for loss curves, embedding geometry, and
reconstructions. Set `RUN` to the desired output directory name.
To vary the experiments, edit `training.lambda`, dot `dataset.groups`, or sprite
`dataset.control` (`none`, `x`, `xy`, `xyt`). Training/evaluation splits and the
encoder, forward/inverse models, and decoder are shared across settings.
