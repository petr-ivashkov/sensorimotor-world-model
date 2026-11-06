"""Behavioral tests of simulator boundaries without changing the local venv."""

from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import h5py
import numpy as np

from planning.experiments.online_adaptation import environments as envs


def write_fixture(path):
    states = np.zeros((6, 7), dtype=np.float64)
    states[:, 2] = [0, 1, 2, 100, 101, 102]
    states[:, 4] = [2 * np.pi - 0.1, 0.1, 0.2, 0, 0.1, 0.2]
    actions = np.array([[0.1, 0.2], [0.3, 0.4], [np.nan, np.nan]] * 2, dtype=np.float32)
    with h5py.File(path, "w") as f:
        f["state"] = states
        f["action"] = actions
        f["ep_len"] = [3, 3]
        f["ep_offset"] = [0, 3]
        f["n_contacts"] = [0, 1, 0, 0, 1, 1]


class EnvironmentTests(unittest.TestCase):
    def test_world_forwards_physics_and_preserves_config(self):
        factory = Mock()
        cfg = {"env_name": "swm/PushT-v1", "block_cog": [0, 0],
               "image_shape": [96, 96], "reset_options": {"state": [1, 2]},
               "history_size": 6, "frame_skip": 1, "max_episode_steps": 80}
        with patch.dict("sys.modules", {"stable_worldmodel": types.SimpleNamespace(World=factory)}):
            envs.make_world(cfg, seed=13, num_envs=4)
        factory.assert_called_once_with(
            env_name="swm/PushT-v1", num_envs=4, image_shape=(96, 96),
            seed=13, history_size=6, frame_skip=1, max_episode_steps=80, block_cog=[0, 0],
        )
        self.assertEqual(cfg["reset_options"], {"state": [1, 2]})

    def test_fixed_statistics_roundtrip_and_source_dtype(self):
        processor = envs.FixedActionProcessor({"mean": [0.1, 0.2], "scale": [0.3, 0.4]})
        raw = np.array([[0.9, -0.8]], dtype=np.float64)
        normalized = processor.transform(raw)
        expected = ((raw - np.array([0.1, 0.2])) / np.array([0.3, 0.4])).astype(np.float32)
        np.testing.assert_array_equal(normalized, expected)
        self.assertEqual(normalized.dtype, np.float32)
        np.testing.assert_allclose(processor.inverse_transform(normalized), raw, atol=1e-7)
        with self.assertRaises(ValueError):
            envs.FixedActionProcessor({"mean": [0, 0], "scale": [1, 0]})

    def test_counts_do_not_join_episodes_and_contacts_follow_action(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "round.h5"
            write_fixture(path)
            stats = envs.summarize_experience(path)
            self.assertEqual(stats["raw_transitions"], 4)
            self.assertEqual(stats["frames"], 6)
            self.assertEqual(stats["contact_transitions"], 3)
            self.assertEqual(stats["contact_fraction"], 0.75)
            self.assertEqual(stats["block_motion_mean"], 1)
            self.assertAlmostEqual(stats["block_angle_change_mean"], 0.125)
            with h5py.File(path, "a") as f:
                f["action"][-1] = [0, 0]
            with self.assertRaisesRegex(ValueError, "terminal-NaN"):
                envs.summarize_experience(path)

    def test_collection_rejects_existing_file_before_loading_simulator(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "round.h5"
            path.write_bytes(b"original")
            with patch.object(envs, "make_world") as factory:
                with self.assertRaises(FileExistsError):
                    envs.collect_round(path, {}, seed=1, episodes=2, policy_config={})
                factory.assert_not_called()
            self.assertEqual(path.read_bytes(), b"original")

    def test_collection_forces_raw_steps_and_closes_world(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "round.h5"
            world = Mock()
            world.record_dataset.side_effect = lambda **kw: write_fixture(
                Path(kw["cache_dir"]) / f"{kw['dataset_name']}.h5"
            )
            with patch.object(envs, "make_world", return_value=world) as factory, \
                 patch.object(envs, "_collector_policy", return_value=(object(), "weak")):
                stats = envs.collect_round(
                    path, {"history_size": 3, "frame_skip": 5, "block_cog": [0, 0]},
                    seed=11, episodes=2, policy_config={"type": "weak"},
                )
            self.assertEqual(factory.call_args.args[0]["history_size"], 1)
            self.assertEqual(factory.call_args.args[0]["frame_skip"], 1)
            self.assertEqual(stats["raw_transitions"], 4)
            self.assertEqual(world.record_dataset.call_args.kwargs["episodes"], 2)
            world.close.assert_called_once()

    def test_failed_recording_closes_world(self):
        world = Mock()
        world.record_dataset.side_effect = RuntimeError("simulator failure")
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(envs, "make_world", return_value=world), \
             patch.object(envs, "_collector_policy", return_value=(object(), "weak")):
            with self.assertRaisesRegex(RuntimeError, "simulator failure"):
                envs.collect_round(Path(folder) / "round.h5", {}, seed=1, episodes=1, policy_config={})
        world.close.assert_called_once()

    def test_incomplete_shard_is_not_published(self):
        world = Mock()
        world.record_dataset.side_effect = lambda **kw: write_fixture(
            Path(kw["cache_dir"]) / f"{kw['dataset_name']}.h5"
        )
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(envs, "make_world", return_value=world), \
             patch.object(envs, "_collector_policy", return_value=(object(), "weak")):
            path = Path(folder) / "round.h5"
            with self.assertRaisesRegex(ValueError, "episode count"):
                envs.collect_round(path, {}, seed=1, episodes=3, policy_config={})
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_policy_recreates_on_new_episode_and_clips_executed_action(self):
        inners = []

        def factory():
            inner = Mock()
            inner.get_action.return_value = np.array([[2.0, -2.0]])
            inner.seed = None
            inners.append(inner)
            return inner

        space = types.SimpleNamespace(low=np.array([[-1, -1]]), high=np.array([[1, 1]]))
        policy = envs._BoundedPolicy(factory, reset_each_episode=True)
        policy.set_env(types.SimpleNamespace(action_space=space))
        first = policy.get_action({"step_idx": np.array([[0]])})
        policy.get_action({"step_idx": np.array([[1]])})
        policy.get_action({"step_idx": np.array([[0]])})
        self.assertEqual(len(inners), 3)
        np.testing.assert_array_equal(first, [[1, -1]])
        self.assertEqual(inners[1].get_action.call_count, 2)
        self.assertEqual(inners[2].get_action.call_count, 1)

    def test_cem_scores_same_bounded_action_tokens_that_execute(self):
        import torch

        processor = envs.FixedActionProcessor({"mean": [0.2, -0.4], "scale": [0.1, 0.3]})
        model = Mock()
        model.get_cost.return_value = torch.tensor([[1.]])
        candidates = torch.tensor([[[[100., -100., -100., 100.],
                                      [0., 0., 0., 0.]]]])
        original = candidates.clone()
        wrapper = envs._BoundedCostModel(model, processor)
        wrapper.get_cost({"pixels": "fixture"}, candidates)
        scored = model.get_cost.call_args.args[1]
        executed_raw = np.clip(processor.inverse_transform(candidates.numpy().reshape(-1, 2)), -1, 1)
        expected = processor.transform(executed_raw).reshape(candidates.shape)
        np.testing.assert_allclose(scored.numpy(), expected, atol=1e-6)
        torch.testing.assert_close(candidates, original)

    def test_canonical_random_policy_config_does_not_select_weak(self):
        factory = Mock()
        factory.return_value.seed = None
        fake_policy = types.SimpleNamespace(RandomPolicy=factory)
        with patch.dict("sys.modules", {"stable_worldmodel.policy": fake_policy}):
            policy, kind = envs._collector_policy(
                {"policy": "random", "dist_constraint": 100}, seed=3,
                model=None, process=None, plan_config=None, solver_config=None,
            )
            policy.set_env(object())
        self.assertEqual(kind, "random")
        factory.assert_called_once_with(seed=3)

    def test_cem_uses_model_device_and_bounded_cost_adapter(self):
        solver = Mock()
        fake_swm = types.SimpleNamespace(
            solver=types.SimpleNamespace(CEMSolver=solver),
            policy=types.SimpleNamespace(WorldModelPolicy=Mock()),
            PlanConfig=lambda **kwargs: kwargs,
        )
        transforms = types.SimpleNamespace(v2=types.SimpleNamespace(ToImage=lambda: "to_image"))
        model = types.SimpleNamespace(parameters=lambda: iter([types.SimpleNamespace(device="cuda:2")]))
        processor = envs.FixedActionProcessor({"mean": [0, 0], "scale": [1, 1]})
        with patch.dict("sys.modules", {"stable_worldmodel": fake_swm, "torchvision.transforms": transforms}):
            envs._make_cem_policy(model, {"action": processor}, {"horizon": 2}, {}, seed=1)
        self.assertEqual(solver.call_args.kwargs["device"], "cuda:2")
        self.assertIsInstance(solver.call_args.kwargs["model"], envs._BoundedCostModel)
        self.assertIs(solver.call_args.kwargs["model"].model, model)

    def test_collection_is_inference_only_and_preserves_batchnorm(self):
        import torch

        model = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 2))
        before = {key: value.clone() for key, value in model.state_dict().items()}
        observations = []

        def record(**kwargs):
            observations.append((model.training, torch.is_inference_mode_enabled(),
                                 model(torch.ones(2, 2)).requires_grad))
            write_fixture(Path(kwargs["cache_dir"]) / f"{kwargs['dataset_name']}.h5")

        world = Mock()
        world.record_dataset.side_effect = record
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(envs, "make_world", return_value=world), \
             patch.object(envs, "_collector_policy", return_value=(object(), "weak")):
            envs.collect_round(Path(folder) / "round.h5", {}, seed=1, episodes=2,
                               policy_config={}, model=model)
        self.assertEqual(observations, [(False, True, False)])
        self.assertTrue(model.training)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key])

    def test_calibration_replays_same_actions_and_closes_both_worlds(self):
        class FakeWorld:
            def __init__(self, speed):
                self.speed = speed
                self.state = np.zeros(7)
                self.infos = {"state": self.state.reshape(1, 1, 7), "n_contacts": [[1]]}
                self.envs = types.SimpleNamespace(step=self.step)
                self.actions = []
                self.closed = False

            def reset(self, **kwargs):
                pass

            def set_policy(self, policy):
                self.policy = policy

            def step(self, actions):
                self.actions.append(actions.copy())
                self.state[2] += self.speed
                return None, None, np.array([False]), np.array([False]), self.infos

            def close(self):
                self.closed = True

        source, target = FakeWorld(1), FakeWorld(2)
        policy = Mock()
        policy.get_action.return_value = np.array([[0.5, -0.5]])
        with patch.object(envs, "make_world", side_effect=[source, target]), \
             patch.object(envs, "_collector_policy", return_value=(policy, "weak")):
            metrics = envs.calibrate_shift({}, {"block_cog": [0, 0]}, seed=12, steps=3)
        self.assertEqual(metrics["steps"], 3)
        self.assertEqual(metrics["final_block_position_divergence"], 3)
        self.assertTrue(metrics["measurable_shift"])
        self.assertTrue(metrics["has_contacts"])
        self.assertTrue(source.closed and target.closed)
        np.testing.assert_array_equal(source.actions, target.actions)

    def test_frozen_tasks_reject_invalid_goal_before_simulation(self):
        dataset = types.SimpleNamespace(lengths=np.array([10, 20]))
        with self.assertRaisesRegex(ValueError, "beyond"):
            envs._validate_tasks(dataset, {
                "episode_ids": [0], "start_steps": [8], "goal_offset_steps": 3,
            })
        episodes, starts, offset = envs._validate_tasks(dataset, {
            "episode_ids": [1], "start_steps": [17], "goal_offset_steps": 3,
        })
        self.assertEqual(episodes.tolist(), [1])
        self.assertEqual(starts.tolist(), [17])
        self.assertEqual(offset, 3)

    def test_planning_uses_frozen_tasks_and_exact_exclusive_end_goal(self):
        class FakeTensor:
            def __init__(self, value):
                self.value = value

            def numpy(self):
                return self.value

        dataset = Mock()
        dataset.lengths = np.array([20])
        dataset.h5_file = Mock()
        goal = np.array([1, 2, 3, 4, 0.2, 0, 0])
        dataset._load_slice.return_value = {"state": [FakeTensor(goal)]}
        world = Mock()
        world.evaluate_from_dataset.return_value = {"success_rate": 100., "episode_successes": np.array([True])}
        world.infos = {"state": goal.reshape(1, 1, 7)}
        model = Mock(training=True)
        fake_swm = types.SimpleNamespace(data=types.SimpleNamespace(HDF5Dataset=Mock(return_value=dataset)))
        with patch.dict("sys.modules", {"stable_worldmodel": fake_swm}), \
             patch.object(envs, "make_world", return_value=world), \
             patch.object(envs, "_make_cem_policy", return_value=Mock()):
            metrics = envs.evaluate_planning(
                model, Path("/tmp/test.h5"), {}, {"mean": [0, 0], "scale": [1, 1]},
                {"episode_ids": [0], "start_steps": [4], "goal_offset_steps": 5},
                seed=1, plan_config={"history_len": 1}, solver_config={},
                eval_budget=10, device="cpu",
            )
        dataset._load_slice.assert_called_once_with(0, 8, 9)
        self.assertEqual(metrics["actual_goal_offset_steps"], 4)
        self.assertEqual(metrics["final_block_position_error"], 0)
        self.assertEqual(metrics["episode_successes"], [True])
        self.assertEqual(world.evaluate_from_dataset.call_args.kwargs["start_steps"], [4])
        world.close.assert_called_once()
        dataset.h5_file.close.assert_called_once()
        model.train.assert_called_once_with(True)


if __name__ == "__main__":
    unittest.main()
