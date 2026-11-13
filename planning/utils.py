"""Input transforms shared by the training scripts."""

import numpy as np
import torch


class ResizeCompat:
    """Version-tolerant image resize transform for stable_pretraining pipelines."""

    def __init__(self, size: int, source: str = "image", target: str = "image"):
        import stable_pretraining as spt
        from torchvision.transforms import v2

        class _ResizeTransform(spt.data.transforms.Transform):
            def __init__(self, resize, source, target):
                super().__init__()
                self.resize = resize
                self.source = source
                self.target = target

            def __call__(self, x):
                self.nested_set(x, self.resize(self.nested_get(x, self.source)), self.target)
                return x

        self.transform = _ResizeTransform(v2.Resize(size), source, target)

    def __call__(self, x):
        return self.transform(x)


def get_column_normalizer(dataset, source: str, target: str):
    import stable_pretraining as spt
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        return ((x - mean) / std).float()

    normalizer = spt.data.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    return normalizer


def disable_external_callbacks():
    # The pinned training stack discovers optional plugin callbacks by default.
    # Use the explicit checkpoint/logging setup in these scripts instead.
    import lightning.fabric.utilities.registry as registry
    import lightning.pytorch.trainer.connectors.callback_connector as connector

    registry._load_external_callbacks = lambda _group: []
    connector._load_external_callbacks = lambda _group: []
