"""Small CPU regressions for deterministic initialization and toy data splits."""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "toy"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "smwm-test-matplotlib"))

import train as toy_train
import train_decoder


class SeedDataset:
    """Expose a lazy dataset's per-sample seed through the real loader pipeline."""

    def __init__(self, _config, *, num_samples, seed):
        self.size = num_samples
        self.seed = seed

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        sample = torch.tensor([self.seed + index], dtype=torch.int64)
        return sample, sample, sample, sample


class ToySplitTests(unittest.TestCase):
    def test_model_and_decoder_loaders_use_the_same_disjoint_samples(self):
        config = {"train_samples": 1004, "eval_samples": 4}
        model_train, model_eval = toy_train.build_loaders(config, None, SeedDataset, 42, 8)
        decoder_train, decoder_eval = train_decoder.build_loaders(config, None, SeedDataset, 42, 8)
        def seeds(loader):
            return set(torch.cat([batch[0] for batch in loader]).flatten().tolist())
        training_seeds, evaluation_seeds = seeds(model_train), seeds(model_eval)
        self.assertFalse(training_seeds & evaluation_seeds)
        self.assertEqual(training_seeds, seeds(decoder_train))
        self.assertEqual(evaluation_seeds, seeds(decoder_eval))

    def test_full_loaders_do_not_preload_every_image(self):
        class UnrenderedDataset(SeedDataset):
            def __getitem__(self, index):
                raise AssertionError("Dataset rendered during loader construction")
        config = {"train_samples": 250_000, "eval_samples": 1000}
        train, evaluation = toy_train.build_loaders(config, None, UnrenderedDataset, 42, 256)
        self.assertEqual(len(train.dataset), 250_000)
        self.assertEqual(len(evaluation.dataset), 1000)

    def test_explicit_overlapping_seed_range_is_rejected(self):
        config = {"train_samples": 250_000, "eval_samples": 1000, "eval_seed": 42 + 999}
        with self.assertRaisesRegex(ValueError, "overlap"):
            toy_train.evaluation_seed(config, 42)

    def test_explicit_disjoint_seed_is_preserved(self):
        config = {"train_samples": 250_000, "eval_samples": 1000, "eval_seed": 1_000_000}
        self.assertEqual(toy_train.evaluation_seed(config, 42), 1_000_000)

    def test_real_datasets_no_longer_repeat_the_old_training_indices(self):
        config_paths = [
            "toy/config/dot.yaml",
            "toy/config/sprite.yaml",
        ]
        for relative_path in config_paths:
            with self.subTest(config=relative_path):
                config = toy_train.load_config(REPO_ROOT / relative_path)
                dataset_config = config["dataset"]
                seed = config["training"]["seed"]
                world, dataset_class, _ = toy_train.build_world(dataset_config)
                train = dataset_class(world, num_samples=dataset_config["train_samples"], seed=seed)
                evaluation = dataset_class(
                    world,
                    num_samples=dataset_config["eval_samples"],
                    seed=toy_train.evaluation_seed(dataset_config, seed),
                )
                for index in (0, 1, 999):
                    self.assertFalse(all(torch.equal(a, b) for a, b in zip(evaluation[index], train[index + 999])))


class ResultTests(unittest.TestCase):
    def test_error_bars_are_sample_std_of_seed_success_rates(self):
        path = REPO_ROOT / "planning/analysis/results.py"
        spec = importlib.util.spec_from_file_location("planning_results", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            for seed in range(5):
                record = dict(
                    env="tworoom", environment="TwoRoom", method="smwm", inverse_weight=10,
                    sigreg_weight=0, history=1, goal_offset=25, eval_budget=50, num_eval=100,
                    task_seed=42025, train_seed=seed, repeat=0, seed=52025 + seed,
                    metrics={"episode_successes": [seed == 4] * 100,
                             "success_rate": 100 if seed == 4 else 0},
                )
                out = Path(directory) / str(seed)
                out.mkdir()
                (out / "metrics.json").write_text(json.dumps(record))
            runs = module.load_results(directory)
            result = module.summarize(runs).iloc[0]
            self.assertEqual(result["mean"], 20)
            self.assertEqual(result["n"], 5)
            self.assertAlmostEqual(result["std"], math.sqrt(2000))
            self.assertTrue(math.isnan(module.summarize(runs.iloc[:1]).iloc[0]["std"]))
            with self.assertRaisesRegex(ValueError, "Repeated seed"):
                module.summarize(pd.concat([runs, runs.iloc[:1]]))


@unittest.skipUnless(importlib.util.find_spec("stable_pretraining"), "planning dependencies are not installed")
class PlanningSeedTests(unittest.TestCase):
    def test_released_weights_load_without_training_state_and_reject_missing_parameters(self):
        script = r'''
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(sys.argv[1]) / "planning"))
import eval as evaluation

def build(_cfg):
    return torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.BatchNorm1d(3))

torch.manual_seed(7)
state = build(None).state_dict()
inputs = torch.randn(4, 2)
with tempfile.TemporaryDirectory() as directory, patch.object(evaluation, "build_jepa", side_effect=build):
    directory = Path(directory)
    (directory / "checkpoints").mkdir()
    OmegaConf.save(OmegaConf.create({"method": "smwm", "seed": 0}), directory / "config.yaml")
    checkpoint = directory / "checkpoints/last.ckpt"
    torch.save({"state_dict": {"model." + key: value for key, value in state.items()},
                "optimizer_states": [{"unused_training_state": 1}]}, checkpoint)
    original, _ = evaluation.load_jepa_from_run(directory, device="cpu")
    torch.save(state, directory / "weights.pt")
    checkpoint.unlink()
    released, _ = evaluation.load_jepa_from_run(directory, device="cpu")
    assert torch.equal(original(inputs), released(inputs))
    assert all(not parameter.requires_grad for parameter in released.parameters())
    broken = dict(state)
    broken.pop("0.weight")
    torch.save(broken, directory / "weights.pt")
    try:
        evaluation.load_jepa_from_run(directory, device="cpu")
    except RuntimeError:
        pass
    else:
        raise AssertionError("An incomplete release was loaded with random parameters")

# A released DINO-WM must also load its embedded backbone configuration offline.
from transformers import Dinov2Config
from baselines.dino_wm.model import build_dino_wm

config = OmegaConf.create({
    "method": "dino_wm", "seed": 0,
    "backbone": {"config": Dinov2Config(
        hidden_size=24, num_hidden_layers=1, num_attention_heads=2,
        image_size=28, patch_size=14, architectures=["Dinov2Model"],
    ).to_dict(), "interpolate_pos_encoding": True},
    "wm": {"encoding": {"action": 10}, "history_size": 1, "num_preds": 1},
    "extra_dims": {"action": 10}, "image_size": 28, "patch_size": 14,
    "predictor": {"depth": 1, "heads": 2, "mlp_dim": 64, "dim_head": 8,
                  "dropout": 0., "emb_dropout": 0.},
})
with tempfile.TemporaryDirectory() as directory, \
     patch("baselines.dino_wm.model.AutoModel.from_pretrained", side_effect=AssertionError("network")), \
     patch("baselines.dino_wm.model.AutoConfig.from_pretrained", side_effect=AssertionError("network")):
    directory = Path(directory)
    original = build_dino_wm(config, pretrained=False).eval()
    OmegaConf.save(config, directory / "config.yaml")
    torch.save(original.state_dict(), directory / "weights.pt")
    released, _ = evaluation.load_jepa_from_run(directory, device="cpu")
    pixels = torch.randn(2, 3, 28, 28)
    with torch.no_grad():
        expected = original.backbone(pixel_values=pixels).last_hidden_state
        actual = released.backbone(pixel_values=pixels).last_hidden_state
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    from types import SimpleNamespace
    from baselines.dino_wm.train import dinowm_forward
    module = SimpleNamespace(model=released, log_dict=lambda *args, **kwargs: None)
    for batch_size in [1, 2]:
        batch = {"pixels": torch.randn(batch_size, 2, 3, 28, 28),
                 "action": torch.randn(batch_size, 2, 10)}
        with torch.no_grad():
            output = dinowm_forward(module, batch, "validate", config)
        assert torch.isfinite(output["loss"])
'''
        result = subprocess.run(
            [sys.executable, "-c", script, str(REPO_ROOT)],
            capture_output=True, text=True, timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_training_entrypoints_seed_before_dataset_and_model_construction(self):
        # Run script entrypoints in a fresh process: toy.datasets and the
        # planning dependency Hugging Face datasets have different import roots.
        script = r'''
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch
import torch
from omegaconf import OmegaConf

class StopBeforeData(Exception):
    pass

root = Path(sys.argv[1])
sys.path.insert(0, str(root / "planning"))
for relative_path, name in [
    ("planning/train.py", "main"),
    ("planning/baselines/pldm/train.py", "pldm"),
    ("planning/baselines/dino_wm/train.py", "dino"),
]:
    path = root / relative_path
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(f"seed_check_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = OmegaConf.create({
        "seed": 42, "wm": {"history_size": 1},
        "data": {"dataset": {"num_steps": 2}},
    })
    observed = []
    def capture_rng(*args, **kwargs):
        observed.append(torch.rand(8))
        raise StopBeforeData
    expected = torch.rand(8, generator=torch.Generator().manual_seed(config.seed))
    torch.manual_seed(9876)
    run = getattr(module.run, "__wrapped__", module.run)
    target = (patch.object(module.swm.data, "HDF5Dataset", side_effect=capture_rng)
              if name == "main" else patch.object(module, "build_datasets", side_effect=capture_rng))
    try:
        with target:
            run(config)
    except StopBeforeData:
        pass
    else:
        raise AssertionError(f"{name}: dataset interception was not reached")
    assert len(observed) == 1 and torch.equal(observed[0], expected), name
'''
        result = subprocess.run(
            [sys.executable, "-c", script, str(REPO_ROOT)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
