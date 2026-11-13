# Toy experiments

A CNN encoder, forward model, and inverse model are trained jointly with
`L = L_fwd + λ·L_inv`. These experiments isolate the relation between inverse dynamics,
controllable degrees of freedom, and the geometry of the learned representation.

Use the root [setup](../README.md#setup). Commands below run from the repository root.
Everything here runs on CPU; a GPU is used automatically if available.

## Single dot

A colored dot moves on a `64×64` canvas under horizontal and vertical displacements.
The experiment tests whether a 64-dimensional encoder recovers the underlying 2D
state space from pixels and actions alone. The analysis compares the PCA spectrum,
a grid of physical positions, and the corresponding latent projection.

```bash
python toy/train.py --config toy/config/dot.yaml --smoke
```

Open [analyze.ipynb](analyze.ipynb) and select `RUN = "dot_smoke"`.
The smoke run uses one epoch on a small dataset. Omit `--smoke` for the full experiment
and select `RUN = "dot"` after training.

To study collapse without inverse dynamics, set `training.lambda` to zero in a copy
of the configuration and give it a different `output.run_name`.

## Structured dots

We vary the controllable degrees of freedom and add uncontrolled distractors.
Independently controlled dots contribute two action coordinates each; a coupled pair
shares one 2D displacement. A randomly moving dot adds visual variation without adding
an action coordinate.

The default configuration combines a coupled pair, two independent dots, and a random
distractor. Edit `dataset.groups` to study these settings separately.

```bash
python toy/train.py --config toy/config/structured.yaml
```

Compare the PCA spectra across configurations to assess whether the representation
tracks controllable structure rather than the number of visible objects.

## Sprite and reconstruction

A triangular sprite translates and rotates. We vary which pose changes are exposed
as actions while keeping the visual dynamics fixed: uncontrolled coordinates continue
to move randomly. After training, we freeze the encoder and train a decoder to visualize
which aspects of position and orientation the representation retains.

```bash
python toy/train.py --config toy/config/sprite.yaml --smoke
python toy/train_decoder.py --config toy/config/decoder.yaml --smoke
```

Select `RUN = "sprite_smoke"` in [analyze.ipynb](analyze.ipynb). Omit `--smoke` for the
full configurations. The control settings are `none`, `x`, `xy`, and `xyt`, selected
with `dataset.control`. Use separate run names for comparisons and set `encoder_run`
in the decoder configuration to the corresponding encoder.

The notebook shows representation geometry, loss curves, and available reconstructions.
Training and evaluation use separate samples. The configurations contain the remaining
model and optimization settings.
