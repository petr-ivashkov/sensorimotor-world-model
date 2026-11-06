"""Focused CPU checks for online update and interruption behavior."""

import copy
import io
import random
import unittest

import numpy as np
import torch

from planning.experiments.online_adaptation.adaptation import OnlineLearner


class TinyModel(torch.nn.Module):
    def __init__(self, *, batchnorm=False, dropout=0.0):
        super().__init__()
        self.encoder = torch.nn.Linear(2, 3)
        self.action_encoder = torch.nn.Linear(2, 3)
        self.predictor = torch.nn.Linear(6, 3)
        self.inverse = torch.nn.Linear(6, 2)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm = torch.nn.BatchNorm1d(3) if batchnorm else torch.nn.Identity()

    def encode(self, batch):
        return {"emb": self.encoder(batch["pixels"]),
                "act_emb": self.action_encoder(batch["action"])}

    def predict(self, embeddings, action_embeddings):
        prediction = self.predictor(torch.cat((embeddings, action_embeddings), -1))
        return self.dropout(self.norm(prediction.flatten(0, 1))).reshape_as(prediction)

    def predict_action(self, current, target):
        return self.inverse(torch.cat((current, target), -1))


def config(history=1, inverse=10.0):
    return {"wm": {"history_size": history},
            "loss": {"inverse": {"weight": inverse}, "sigreg": {"weight": 0.0}}}


def batch(batch_size=4, time=2):
    return {"pixels": torch.arange(batch_size * time * 2).reshape(batch_size, time, 2).float() / 10,
            "action": torch.linspace(-0.5, 0.5, batch_size * time * 2).reshape(batch_size, time, 2)}


class OnlineLearnerTest(unittest.TestCase):
    def learner(self, model=None, **kwargs):
        return OnlineLearner(model or TinyModel(), config(), device="cpu", **kwargs)

    def test_source_loss_and_target_gradient_match(self):
        model = TinyModel()
        learner = self.learner(model=model)
        data = batch()
        output = model.encode(data)
        embeddings = output["emb"]
        expected_forward = (model.predict(embeddings[:, :1], output["act_emb"][:, :1])
                            - embeddings[:, 1:2]).square().mean()
        expected_inverse = (model.predict_action(embeddings[:, :-1], embeddings[:, 1:])
                            - data["action"][:, :-1]).square().mean()
        expected = expected_forward + 10 * expected_inverse
        expected.backward()
        expected_gradient = model.encoder.weight.grad.clone()
        model.zero_grad()
        losses = learner._losses(dict(data))
        losses["loss"].backward()
        torch.testing.assert_close(losses["forward_loss"], expected_forward)
        torch.testing.assert_close(losses["inverse_loss"], expected_inverse)
        torch.testing.assert_close(model.encoder.weight.grad, expected_gradient)

    def test_updates_reenable_pretrained_weights_and_keep_optimizer(self):
        model = TinyModel().eval().requires_grad_(False)
        learner = self.learner(model=model)
        before = model.encoder.weight.detach().clone()
        first = learner.update([batch()], 2)
        second = learner.update([batch()], 3)
        self.assertEqual(first["steps"], 2)
        self.assertEqual(second["update_steps"], 5)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertFalse(torch.equal(before, model.encoder.weight))
        self.assertTrue(all(int(value["step"]) == 5 for value in learner.optimizer.state.values()))
        self.assertEqual(learner.precision, "float32")

    def test_resume_preserves_optimizer_rng_and_future_updates(self):
        learner = self.learner(model=TinyModel(dropout=0.3), seed=31)
        learner.update([batch()], 2)
        serialized = io.BytesIO()
        torch.save(learner.state_dict(), serialized)
        serialized.seek(0)
        snapshot = torch.load(serialized, map_location="cpu", weights_only=False)
        expected_random = (random.random(), np.random.rand(), torch.rand(4))
        expected_metrics = learner.update([batch()], 3)
        expected_parameters = copy.deepcopy(learner.model.state_dict())
        restored = self.learner(model=TinyModel(dropout=0.3), seed=99)
        restored.load_state_dict(snapshot)
        self.assertEqual(random.random(), expected_random[0])
        self.assertEqual(np.random.rand(), expected_random[1])
        torch.testing.assert_close(torch.rand(4), expected_random[2], rtol=0, atol=0)
        actual_metrics = restored.update([batch()], 3)
        self.assertEqual(expected_metrics, actual_metrics)
        for key, value in restored.model.state_dict().items():
            torch.testing.assert_close(value, expected_parameters[key], rtol=0, atol=0)
        # Snapshots must not drift as the original learner keeps training.
        self.assertEqual(snapshot["update_steps"], 2)
        self.assertEqual(int(next(iter(snapshot["optimizer_state"]["state"].values()))["step"]), 2)
        self.assertTrue(all(key.startswith("model.") for key in snapshot["state_dict"]))

    def test_strict_restore_rejects_wrong_objective_and_source_checkpoint(self):
        learner = self.learner()
        state = learner.state_dict()
        incompatible = OnlineLearner(TinyModel(), config(inverse=1.0), device="cpu")
        with self.assertRaisesRegex(ValueError, "settings"):
            incompatible.load_state_dict(state)
        with self.assertRaisesRegex(ValueError, "keys"):
            learner.load_state_dict({"state_dict": state["state_dict"]})
        del state["state_dict"]["model.encoder.weight"]
        with self.assertRaises(RuntimeError):
            learner.load_state_dict(state)

    def test_empty_loader_and_invalid_steps_fail(self):
        learner = self.learner()
        with self.assertRaisesRegex(ValueError, "empty"):
            learner.update([], 1)
        for steps in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                learner.update([batch()], steps)
        self.assertEqual(learner.update([], 0), {"steps": 0, "update_steps": 0})

    def test_batchnorm_small_batch_and_history_contract(self):
        learner = self.learner(model=TinyModel(batchnorm=True))
        with self.assertRaisesRegex(ValueError, "BatchNorm"):
            learner.update([batch(batch_size=1)], 1)
        history_learner = OnlineLearner(TinyModel(batchnorm=True), config(history=2), device="cpu")
        with self.assertRaisesRegex(ValueError, "at least 3"):
            history_learner.update([batch()], 1)
        result = history_learner.update([batch(batch_size=1, time=3)], 1)
        self.assertEqual(result["steps"], 1)

    def test_nonfinite_loss_does_not_update_parameters(self):
        learner = self.learner()
        before = copy.deepcopy(learner.model.state_dict())
        data = batch()
        data["pixels"].fill_(float("nan"))
        with self.assertRaises(FloatingPointError):
            learner.update([data], 1)
        self.assertEqual(learner.update_steps, 0)
        for key, value in learner.model.state_dict().items():
            torch.testing.assert_close(value, before[key])

    def test_zero_inverse_weight_omits_inverse_objective(self):
        learner = OnlineLearner(TinyModel(), config(inverse=0.0), device="cpu")
        result = learner.update([batch()], 1)
        self.assertNotIn("inverse_loss", result)
        self.assertEqual(result["loss"], result["forward_loss"])
        self.assertIsNone(learner.model.inverse.weight.grad)

    def test_optional_sigreg_uses_source_settings_and_restores(self):
        source_cfg = config()
        source_cfg["loss"]["sigreg"] = {"weight": 0.09, "kwargs": {"knots": 3, "num_proj": 4}}
        learner = OnlineLearner(TinyModel(), source_cfg, device="cpu")
        result = learner.update([batch()], 1)
        self.assertGreaterEqual(result["sigreg_loss"], 0.0)
        self.assertAlmostEqual(result["loss"], result["forward_loss"]
                               + 10 * result["inverse_loss"] + 0.09 * result["sigreg_loss"], places=5)
        restored = OnlineLearner(TinyModel(), source_cfg, device="cpu")
        restored.load_state_dict(learner.state_dict())
        self.assertEqual(restored.update_steps, 1)
        torch.testing.assert_close(restored.sigreg.weights, learner.sigreg.weights)


if __name__ == "__main__":
    unittest.main()
