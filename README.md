# Sensorimotor World Model

Learning latent world models whose representations capture the **controllable**
structure of an environment. A JEPA-style encoder is trained jointly with a forward
model and an **inverse-dynamics** model — `L = L_fwd + λ·L_inv` — where the inverse
loss is the anti-collapse signal that ties the latent space to the degrees of
freedom an agent can actually act on.

**Links:** [Paper](#) · [arXiv](#) · [Project page](https://github.com/petr-ivashkov/sensorimotor-world-model.github.io) <!-- TODO: fill paper / arXiv links -->

## Repository map

| Subproject | What it is |
|---|---|
| [`toy/`](toy/) | Dot-world / sprite experiments that isolate the inverse-dynamics anti-collapse mechanism. Light, CPU-friendly. |
| [`planning/`](planning/) | Planning experiments on four environments (TwoRoom, Reacher, Push-T, OGBench-Cube). Built on [LeWorldModel](https://github.com/lucas-maes/le-wm); needs a CUDA GPU. |

A single environment at the repo root covers both subprojects:

```bash
uv sync                      # creates ./.venv (Python ≥ 3.13)
source .venv/bin/activate
```

> On macOS/arm64 the `toy/` half installs and runs as-is; the `planning/` half pulls
> simulator dependencies that build only on Linux + CUDA (where the experiments are run).

## Two ways in

- **Just try the method** (laptop, no GPU) → [`toy/`](toy/README.md):
  ```bash
  cd toy && python train.py --config experiments/single_dot/config.yaml
  ```
- **Reproduce the paper** (CUDA GPU + datasets) → [`planning/`](planning/README.md),
  which is the single place documenting full reproduction.

## License

Released under the MIT License ([`LICENSE`](LICENSE)). The `planning/` subproject derives
from LeWorldModel — see [`planning/README.md`](planning/README.md#credit) for attribution.
