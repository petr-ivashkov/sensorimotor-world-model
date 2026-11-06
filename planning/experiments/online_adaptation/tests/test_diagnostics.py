"""Analytic diagnostics checks; no environments, checkpoints, or GPU required."""

import copy
import math
import unittest

import torch
from torch import nn

from planning.experiments.online_adaptation.diagnostics import (
    circular_mae,
    evaluate_diagnostics,
    linear_cka,
    physical_targets,
)


class AnalyticModel(nn.Module):
    """Six physical features and known noncommuting affine transitions."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.predictor = nn.Identity()
        self.predictor.num_frames = 1
        self.action_encoder = nn.Identity()
        self.inverse_model = nn.Identity()
        self.prediction_bias = 0.0
        self.actions_seen = []

    def encode(self, info):
        return {"emb": info["pixels"][..., 0, 0].float() * self.scale}

    def predict(self, embeddings, actions):
        self.actions_seen.append(actions.detach().clone())
        result = embeddings * actions[..., :1] + actions[..., 1:7] * self.scale
        bias = result.new_tensor([1, 1, 1, 1, 0, 0]) * self.prediction_bias
        return result + bias

    def predict_action(self, initial, future):
        return initial.new_zeros((*initial.shape[:-1], 10))


def make_batch(seed, episode_start, count=40, steps=6):
    generator = torch.Generator().manual_seed(seed)
    state = torch.randn(count, steps, 7, generator=generator)
    state[:, 0, 4] = torch.linspace(-math.pi + 0.02, math.pi - 0.02, count)
    actions = torch.zeros(count, steps, 10)
    scales = (0.8, 1.2, 0.9, 1.1, 0.7)
    for step in range(steps - 1):
        scale = scales[step]
        delta = torch.tensor([step + 0.3, -0.2, 0.4, step * 0.5])
        state[:, step + 1] = state[:, step]
        state[:, step + 1, :4] = state[:, step, :4] * scale + delta
        actions[:, step, 0] = scale
        actions[:, step, 1:5] = delta
        actions[:, step, 5:7] = physical_targets(state[:, step])[:, 4:] * (1 - scale)
    actions[:, -1] = float("nan")  # Writer's terminal padding must be ignored.
    pixels = physical_targets(state)[..., None, None]
    return {
        "pixels": pixels,
        "action": actions,
        "state": state,
        "ep_idx": torch.arange(episode_start, episode_start + count),
    }


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.fit = make_batch(7, 0)
        self.test = make_batch(9, 100)

    def evaluate(self, model=None, fit=None, test=None, **kwargs):
        return evaluate_diagnostics(
            model or AnalyticModel(), [fit or self.fit], [test or self.test],
            device="cpu", ridge_alpha=0, **kwargs,
        )

    def test_cka_rotation_scale_translation_invariance_and_constant_space(self):
        generator = torch.Generator().manual_seed(18)
        features = torch.randn(30, 8, generator=generator)
        rotation = torch.linalg.qr(torch.randn(8, 8, generator=generator)).Q
        transformed = 4 * features @ rotation + 17
        self.assertAlmostEqual(linear_cka(features, transformed), 1, places=10)
        self.assertTrue(math.isnan(linear_cka(features, torch.ones_like(features))))

    def test_angle_wrapping_across_pi(self):
        predicted = torch.tensor([math.pi - 0.01])
        target = torch.tensor([-math.pi + 0.01])
        self.assertAlmostEqual(circular_mae(predicted, target), 0.02, places=6)

    def test_exact_prediction_physical_probes_and_outgoing_action_order(self):
        model = AnalyticModel()
        metrics, reference = self.evaluate(model)
        self.assertEqual(reference["pixels"].shape[1], 1)
        self.assertEqual(metrics["probe.fit_samples"], 40)
        self.assertEqual(metrics["probe.fit_episode_count"], 40)
        for horizon in (1, 3, 5):
            self.assertLess(metrics[f"horizon_{horizon}.latent_mse"], 1e-11)
            self.assertLess(metrics[f"horizon_{horizon}.prediction.block_position_rmse"], 1e-5)
            self.assertLess(metrics[f"horizon_{horizon}.prediction.block_angle_mae_rad"], 1e-5)
        self.assertGreater(metrics["horizon_5.persistence.block_position_rmse"], 1)
        for step, actual in enumerate(model.actions_seen):
            torch.testing.assert_close(actual, self.test["action"][:, step:step + 1])
        expected_mse = self.test["action"][:, 0].square().mean()
        self.assertAlmostEqual(metrics["inverse.action_mse"], float(expected_mse), places=6)

    def test_rollout_uses_its_previous_prediction(self):
        model = AnalyticModel()
        model.prediction_bias = 0.1
        metrics, _ = self.evaluate(model)
        error = 0.0
        for step, scale in enumerate((0.8, 1.2, 0.9, 1.1, 0.7), start=1):
            error = error * scale + model.prediction_bias
            if step in (1, 3, 5):
                self.assertAlmostEqual(metrics[f"horizon_{step}.latent_mse"], error**2 * 4 / 6, places=6)

    def test_reference_is_fixed_and_coordinate_drift_detects_scaling(self):
        model = AnalyticModel()
        metrics0, reference = self.evaluate(model)
        baseline = copy.deepcopy(reference)
        self.assertAlmostEqual(metrics0["representation.relative_coordinate_drift"], 0)
        model.scale.data.fill_(2)
        # A changed loader must not replace the stored anchor images/order.
        other_test = make_batch(91, 1000, count=11)
        metrics1, returned = self.evaluate(model, test=other_test, reference=reference)
        self.assertIs(returned, reference)
        self.assertEqual(metrics1["representation.anchor_count"], 40)
        self.assertAlmostEqual(metrics1["representation.linear_cka"], 1, places=10)
        self.assertAlmostEqual(metrics1["representation.relative_coordinate_drift"], 1, places=7)
        for key in ("pixels", "state", "embeddings"):
            torch.testing.assert_close(reference[key], baseline[key])
        self.assertAlmostEqual(
            metrics1["representation.std_mean"], 2 * metrics0["representation.std_mean"], places=6,
        )

    def test_probe_never_fits_future_fit_frames_or_test_labels(self):
        base, _ = self.evaluate()
        altered_fit = copy.deepcopy(self.fit)
        altered_fit["state"][:, 1:, :4] += 100000
        metrics, _ = self.evaluate(fit=altered_fit)
        self.assertAlmostEqual(metrics["observed.block_position_rmse"], base["observed.block_position_rmse"])
        altered_test = copy.deepcopy(self.test)
        altered_test["state"][:, :, :4] += 100
        metrics, _ = self.evaluate(test=altered_test)
        self.assertAlmostEqual(metrics["observed.block_position_rmse"], math.sqrt(2) * 100, places=4)

    def test_episode_overlap_rejected_and_modes_restored_on_failure(self):
        model = AnalyticModel()
        model.train()
        model.predictor.eval()  # Preserve frozen-module modes, too.
        with self.assertRaisesRegex(ValueError, "episode IDs overlap"):
            self.evaluate(model, test=self.fit)
        self.assertTrue(model.training)
        self.assertFalse(model.predictor.training)
        self.assertTrue(model.action_encoder.training)
        self.assertIsNone(model.scale.grad)

    def test_modes_restored_after_success_and_no_gradients_created(self):
        model = AnalyticModel()
        model.train()
        model.predictor.eval()
        self.evaluate(model)
        self.assertTrue(model.training)
        self.assertFalse(model.predictor.training)
        self.assertIsNone(model.scale.grad)

    def test_insufficient_windows_or_nonfinite_consumed_actions_rejected(self):
        short = make_batch(9, 100, steps=2)
        with self.assertRaisesRegex(ValueError, "at least 6 observations"):
            self.evaluate(test=short)
        invalid = copy.deepcopy(self.test)
        invalid["action"][:, 2] = float("nan")
        with self.assertRaisesRegex(ValueError, "must be finite"):
            self.evaluate(test=invalid)


if __name__ == "__main__":
    unittest.main()
