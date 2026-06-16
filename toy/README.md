# Causal World Modelling in Dot World

JEPA-style latent world model — CNN encoder `f_θ`, forward model `g_φ`, inverse
model `h_ψ`, trained jointly with `L = L_fwd + λ·L_inv`. The inverse loss is the
anti-collapse signal.

## Install

The environment is defined once at the repo root (see the top-level [`README.md`](../README.md)):

```bash
uv sync                 # from the repo root; creates ../.venv (Python ≥ 3.13)
source ../.venv/bin/activate
```

Everything here runs on CPU (a GPU is used automatically if present).

## Reproduce

Train, then open the experiment's notebook and **Run All**. Checkpoints land in `experiments/<exp>/results/<run_name>/`.

```bash
# single_dot
python train.py --config experiments/single_dot/config.yaml
python train.py --config experiments/single_dot/config_lambda0.yaml

# structured_dots
for c in 2_independent 1_coupled_pair 1_dot_1_random combined; do
  python train.py --config experiments/structured_dots/config_$c.yaml
done

# sprite_decoder — all encoders first, then all decoders
for c in none x xy xyt; do python train.py --config experiments/sprite_decoder/config_$c.yaml; done
for c in none x xy xyt; do python train_decoder.py --config experiments/sprite_decoder/config_decoder_$c.yaml; done
```