"""Actual two-body simulation, collection, and diagnostic-state contracts."""
import contextlib
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import gymnasium as gym
import h5py
import numpy as np
import pymunk
import torch

from planning.experiments.online_adaptation.two_block_env import PushTwoBlock
from planning.experiments.online_adaptation.collection_policy import MultiBlockWeakPolicy
from planning.experiments.online_adaptation.diagnostics import physical_targets, _physical_errors, evaluate_diagnostics
from planning.experiments.online_adaptation.environments import collect_round, calibrate_shift
from planning.experiments.online_adaptation.run import _training_collate, load_config, validate_config
from planning.experiments.online_adaptation.adaptation import OnlineLearner
from planning.experiments.online_adaptation.tests.test_adaptation import TinyModel, config, batch


@contextlib.contextmanager
def compatible_collision_api():
    # Only the test process adapts the local Pymunk 6 callback name. Production
    # uses native Pymunk 7 on the cluster; shared installed packages are untouched.
    if hasattr(pymunk.Space, "on_collision"):
        yield
        return
    class TestSpace(pymunk.Space):
        def on_collision(self, first, second, *, post_solve):
            self.add_collision_handler(first, second).post_solve = post_solve
    with patch.object(pymunk, "Space", TestSpace):
        yield


class TwoBlockTests(unittest.TestCase):
    def test_reset_step_render_and_full_state_restore(self):
        with compatible_collision_api():
            env = PushTwoBlock()
            try:
                obs, info = env.reset(seed=42)
                self.assertEqual(obs["state"].shape, (10,))
                self.assertEqual(obs["proprio"].shape, (4,))
                self.assertEqual(info["goal_state"].shape, (10,))
                self.assertEqual(env.render().shape, (224, 224, 3))
                self.assertTrue(env._second_pose_is_valid())
                for body in env.movable_blocks:
                    self.assertEqual(body.body_type, pymunk.Body.DYNAMIC)
                    self.assertEqual(body.mass, 1)
                    self.assertEqual(len(body.shapes), 2)
                state = obs["state"].copy()
                initial = env.render().copy()
                repeated, _ = env.reset(seed=42)
                np.testing.assert_allclose(repeated["state"], state)
                np.testing.assert_array_equal(env.render(), initial)
                env._set_state(state)
                np.testing.assert_allclose(env._get_obs(), state, atol=1e-5)
                stepped, *_ = env.step(np.array([.3, -.2]))
                self.assertTrue(np.isfinite(stepped["state"]).all())
                np.testing.assert_allclose(stepped["proprio"], np.r_[stepped["state"][:2], stepped["state"][-2:]])
            finally:
                env.close()

    def test_pushing_transfers_motion_through_two_dynamic_bodies(self):
        with compatible_collision_api():
            env = PushTwoBlock()
            try:
                env.reset(seed=3)
                state = np.array([80, 175, 180, 160, 0, 320, 160, 0, 0, 0.])
                env._set_state(state)
                start = np.array([env.block.position.x, env.block2.position.x])
                contacts = []
                for _ in range(30):
                    *_, info = env.step(np.array([1., 0.]))
                    contacts.append(info["block_block_contact"])
                end = np.array([env.block.position.x, env.block2.position.x])
                self.assertGreater(end[0]-start[0], 10)
                self.assertGreater(end[1]-start[1], 10)
                self.assertTrue(any(contacts))
            finally:
                env.close()

    def test_collector_switches_between_both_actual_bodies(self):
        with compatible_collision_api():
            env = PushTwoBlock()
            try:
                env.reset(seed=4)
                env.spec = gym.envs.registration.EnvSpec("swm/PushT-TwoBlock-v0")
                policy = MultiBlockWeakPolicy(seed=5, target_switch_steps=50)
                vector = types.SimpleNamespace(
                    unwrapped=types.SimpleNamespace(envs=[env]), spec=env.spec,
                    action_space=gym.spaces.Box(-1, 1, shape=(1, 2), dtype=np.float32),
                )
                policy.set_env(vector)
                policy.get_action({"step_idx": [[0]]})
                first = env.collector_target_block
                action = policy.get_action({"step_idx": [[50]]})
                self.assertEqual(env.collector_target_block, 1-first)
                self.assertTrue(np.all(np.abs(action) <= 1))
            finally:
                env.close()

    def test_recorded_dataset_has_both_states_actions_contacts_and_motion(self):
        with compatible_collision_api(), tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "two.h5"
            summary = collect_round(path, {"env_name": "swm/PushT-TwoBlock-v0", "max_episode_steps": 200},
                                    seed=42, episodes=2, policy_config={"policy": "weak"})
            self.assertEqual(summary["raw_transitions"], 400)
            self.assertGreater(summary["block_motion_mean"], 0)
            self.assertGreater(summary["block2_motion_mean"], 0)
            self.assertGreater(summary["agent_block1_contact_fraction"], 0)
            self.assertGreater(summary["agent_block2_contact_fraction"], 0)
            with h5py.File(path) as handle:
                self.assertEqual(handle["state"].shape[-1], 10)
                self.assertEqual(handle["action"].shape[-1], 2)
                self.assertEqual(set(np.unique(handle["collector_target_block"])), {0, 1})

    def test_calibration_accepts_different_state_sizes(self):
        with compatible_collision_api():
            metrics = calibrate_shift({"env_name": "swm/PushT-v1"},
                                      {"env_name": "swm/PushT-TwoBlock-v0"}, seed=42, steps=20)
            self.assertEqual(metrics["source_state_dim"], 7)
            self.assertEqual(metrics["target_state_dim"], 10)
            self.assertTrue(np.isfinite(metrics["initial_state_difference"]))

    def test_physical_targets_and_errors_include_secondary_pose(self):
        raw = torch.tensor([[10., 20, 30, 40, 0, 50, 60, np.pi/2, 70, 80]])
        target = physical_targets(raw)
        torch.testing.assert_close(target[:, 6:8], raw[:, 5:7])
        torch.testing.assert_close(target[:, 8:], torch.tensor([[1., 0.]]), atol=1e-6, rtol=0)
        decoded = target.clone()
        decoded[:, 6] += 3
        decoded[:, 7] += 4
        metrics = _physical_errors(decoded, target)
        self.assertEqual(metrics["block_position_rmse"], 0)
        self.assertEqual(metrics["block2_position_rmse"], 5)
        self.assertAlmostEqual(metrics["all_blocks_position_rmse"], np.sqrt(12.5), places=6)

    def test_full_rollout_diagnostics_measure_both_blocks(self):
        class IdentityDynamics(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.predictor = torch.nn.Identity()
                self.predictor.num_frames = 1
                self.action_encoder = torch.nn.Identity()
            def encode(self, data):
                return {"emb": data["pixels"][..., 0, 0]}
            def predict(self, latent, action):
                return latent + action
        def data(seed, start):
            rng = torch.Generator().manual_seed(seed)
            initial = torch.randn(64, 10, generator=rng)
            initial[:, [4, 7]] *= 2
            state = initial[:, None].repeat(1, 6, 1)
            for step in range(6):
                state[:, step, 5:7] += step * torch.tensor([3., 4.])
            features = physical_targets(state)
            actions = torch.zeros(64, 6, 10)
            actions[:, :-1] = features[:, 1:]-features[:, :-1]
            return {"pixels": features[..., None, None], "state": state,
                    "action": actions, "ep_idx": torch.arange(start, start+64)}
        metrics, _ = evaluate_diagnostics(IdentityDynamics(), [data(1, 0)], [data(2, 100)],
                                         ridge_alpha=0, device="cpu")
        self.assertLess(metrics["horizon_5.prediction.block2_position_rmse"], 1e-4)
        self.assertGreater(metrics["horizon_5.persistence.block2_position_rmse"], 24)
        self.assertLess(metrics["horizon_5.prediction.block_position_rmse"], 1e-4)

    def test_source_and_two_block_training_mix_without_state_leakage(self):
        samples = [{"pixels": torch.zeros(2, 3, 2, 2), "action": torch.ones(2, 10),
                    "state": torch.zeros(2, size)} for size in (7, 10)]
        result = _training_collate(samples)
        self.assertNotIn("state", result)
        self.assertEqual(result["action"].shape, (2, 2, 10))

    def test_time_budget_finishes_an_update_then_preserves_optimizer(self):
        learner = OnlineLearner(TinyModel(), config(), device="cpu")
        with patch("planning.experiments.online_adaptation.adaptation.time.monotonic", side_effect=[0, 1]):
            metrics = learner.update([batch()], 10000, max_seconds=.5)
        self.assertEqual(metrics["steps"], 1)
        self.assertEqual(learner.update_steps, 1)
        for state in learner.optimizer.state.values():
            self.assertEqual(int(state["step"]), 1)

    def test_both_overnight_configs_resolve_and_validate(self):
        directory = Path(__file__).resolve().parents[1]
        for name, env in (("overnight_u.yaml", "swm/PushT-U-v0"),
                          ("overnight_two_block.yaml", "swm/PushT-TwoBlock-v0")):
            cfg = load_config(directory / name)
            validate_config(cfg)
            self.assertEqual(cfg["target_change"]["env_name"], env)
            self.assertEqual(cfg["rounds"]*cfg["update_seconds_per_round"], 9000)
            self.assertEqual(cfg["target_change"]["reset_options"]["variation_values"]["block.shape"], 2)


if __name__ == "__main__":
    unittest.main()
