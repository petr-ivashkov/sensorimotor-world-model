# Third-party notices

The project is released under the [MIT License](LICENSE). The following copied or
adapted components retain their upstream copyright and MIT license notices.

## LeWorldModel

The planning encoder/predictor, world-model training and evaluation pipeline,
utilities, and Hydra configuration layout derive from
[LeWorldModel](https://github.com/lucas-maes/le-wm) by Lucas Maes.

- Copyright (c) 2026 Lucas Maes.
- License copy: [planning/LICENSE.LeWorldModel](planning/LICENSE.LeWorldModel).

## stable-worldmodel and PLDM

The code under [`planning/baselines/pldm/vendor/`](planning/baselines/pldm/vendor/)
is copied from the [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel)
PLDM port. Its source commit and copied files are recorded in
[baseline notes](planning/baselines/README.md).

- Copyright (c) 2026 GalilAI-group.
- License copy: [vendor/LICENSE](planning/baselines/pldm/vendor/LICENSE).

The port is based on [PLDM](https://github.com/vladisai/PLDM) by Vlad Sobal and
collaborators. Its upstream provenance is also recorded in `SOURCE.md`.

- Copyright (c) 2025 Vlad Sobal.
- License copy: [vendor/LICENSE.PLDM](planning/baselines/pldm/vendor/LICENSE.PLDM).

## Installed implementations and pretrained models

The DINO-WM baseline imports the `stable-worldmodel` implementation and uses a
frozen DINOv2 backbone. The implementation, model revision, and adaptations are
recorded in [baseline notes](planning/baselines/README.md).

Installed dependencies, downloaded datasets, and pretrained weights retain their
respective upstream licenses, available in their package metadata and model or
dataset cards.
