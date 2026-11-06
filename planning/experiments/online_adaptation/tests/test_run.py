"""Run the real local learning/data/diagnostic path with simulator boundaries mocked."""

import contextlib
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from planning.experiments.online_adaptation import run


class TinyImageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(3, 4)
        self.projector = torch.nn.Tanh()
        self.action_encoder = torch.nn.Linear(10, 4)
        self.predictor = torch.nn.Linear(8, 4)
        self.predictor.num_frames = 1
        self.inverse_model = torch.nn.Linear(8, 10)

    def encode(self, batch):
        features = batch["pixels"].float().mean((-1, -2)) / 255
        result = {"emb": self.projector(self.encoder(features))}
        if "action" in batch:
            result["act_emb"] = self.action_encoder(batch["action"])
        return result

    def predict(self, latent, action_embedding):
        return self.predictor(torch.cat((latent, action_embedding), -1))

    def predict_action(self, current, target):
        return self.inverse_model(torch.cat((current, target), -1))


def synthetic_shard(path, episodes=4, length=12):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = episodes * length
    index = np.arange(rows, dtype=np.float64)
    colors = np.stack(((index * 5) % 255, (index * 13 + 7) % 255,
                       (index * 3 + 31) % 255), axis=1).astype(np.uint8)
    pixels = np.broadcast_to(colors[:, None, None, :], (rows, 2, 2, 3))
    actions = np.stack((30 + index % 13, 40 + index % 17), axis=1)
    actions[np.arange(1, episodes + 1) * length - 1] = np.nan
    state = np.stack((40 + index, 50 + index * 0.5, 80 + index * 0.3,
                      90 + index * 0.7, index * 0.03, index * 0, index * 0), axis=1)
    with h5py.File(path, "w") as handle:
        handle["pixels"] = pixels
        handle["action"] = actions
        handle["state"] = state
        handle["ep_len"] = np.full(episodes, length, dtype=np.int64)
        handle["ep_offset"] = np.arange(episodes) * length
        handle["ep_idx"] = np.repeat(np.arange(episodes), length)
        handle["step_idx"] = np.tile(np.arange(length), episodes)
    return {"path": str(path), "episodes": episodes,
            "raw_transitions": episodes * (length - 1), "frames": rows}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RunIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.source_run = self.root / "source_run"
        (self.source_run / "checkpoints").mkdir(parents=True)
        self.source_train = self.root / "source_train.h5"
        self.source_eval = self.root / "source_eval.h5"
        synthetic_shard(self.source_train)
        synthetic_shard(self.source_eval)
        self.source_cfg = OmegaConf.create({
            "wm": {"history_size": 1, "action_dim": 2},
            "data": {"dataset": {"frameskip": 5, "num_steps": 2}},
            "loss": {"inverse": {"weight": 10.0}, "sigreg": {"weight": 0.0}},
        })
        OmegaConf.save(self.source_cfg, self.source_run / "config.yaml")
        torch.save({"source_sentinel": True}, self.source_run / "checkpoints/last.ckpt")
        self.source_files = [self.source_train, self.source_eval, self.source_run / "config.yaml",
                             self.source_run / "checkpoints/last.ckpt"]
        self.before = {path: digest(path) for path in self.source_files}
        self.report = {"problems": [], "source_train": {"path": str(self.source_train)},
                       "source_eval": {"path": str(self.source_eval)}, "fixture": "local-only"}
        self.cfg = {
            "seed": 31, "initialization": "full", "rounds": 1,
            "episodes_per_round": 2, "updates_per_round": 2,
            "source_env": {"env_name": "fixture", "max_episode_steps": 12},
            "target_change": {"block_cog": [20, 45]},
            "calibration": {"steps": 10, "min_mean_position_divergence": 1.0},
            "collection": {"policy": "weak", "dist_constraint": 100},
            "replay": {"recent_rounds": None, "source_fraction": 0.0},
            "training": {"batch_size": 2, "num_workers": 0, "learning_rate": 0.01,
                         "weight_decay": 0.001, "precision": "float32", "gradient_clip": 1.0},
            "diagnostics": {"holdout_episodes": 4, "max_clips": 8, "batch_size": 2,
                            "horizons": [1], "ridge_alpha": 1.0},
            "planning": {"enabled": True, "num_tasks": 2, "goal_offset_steps": 5,
                         "eval_budget": 5, "plan": {}, "solver": {}},
        }
        stack = self.enterContext(contextlib.ExitStack())
        self.preflight = stack.enter_context(mock.patch.object(
            run, "preflight", return_value=(self.report, self.source_cfg)))
        self.model_factory = stack.enter_context(mock.patch.object(
            run, "_load_model", side_effect=lambda *args: TinyImageModel().eval().requires_grad_(False)))
        self.calibration = stack.enter_context(mock.patch.object(
            run, "calibrate_shift", return_value={"mean_block_position_divergence": 2.0,
                                                   "has_contacts": True}))
        self.collector = stack.enter_context(mock.patch.object(
            run, "collect_round", side_effect=lambda path, env, **kw: synthetic_shard(path, kw["episodes"])))
        self.planner = stack.enter_context(mock.patch.object(
            run, "evaluate_planning", return_value={"success_rate": 0.25, "fixture": True}))

    def execute(self, output, **kwargs):
        return run.run_experiment(self.cfg, self.source_run, output,
                                  source_data=self.source_train, source_eval=self.source_eval,
                                  device="cpu", **kwargs)

    def checkpoint(self, output, round_index):
        return torch.load(output / f"snapshots/round_{round_index:03d}/checkpoints/last.ckpt",
                          map_location="cpu", weights_only=False)["state_dict"]

    def assert_sources_unchanged(self):
        self.assertEqual(self.before, {path: digest(path) for path in self.source_files})

    def test_one_round_updates_model_measures_drift_and_preserves_sources(self):
        output = self.root / "online"
        self.assertEqual(self.execute(output), output)
        zero = json.loads((output / "metrics/round_000.json").read_text())
        one = json.loads((output / "metrics/round_001.json").read_text())
        self.assertEqual((zero["round"], zero["updates"], zero["experience_transitions"]), (0, 0, 0))
        self.assertEqual((one["round"], one["updates"], one["experience_transitions"]), (1, 2, 22))
        self.assertEqual(one["training"]["steps"], 2)
        self.assertEqual(one["training"]["update_steps"], 2)
        self.assertTrue(np.isfinite(one["training"]["loss"]))
        baseline, adapted = self.checkpoint(output, 0), self.checkpoint(output, 1)
        self.assertFalse(torch.equal(baseline["model.encoder.weight"], adapted["model.encoder.weight"]))
        self.assertGreater(one["target"]["diagnostics"]["representation.relative_coordinate_drift"], 0)
        self.assertEqual(zero["target"]["diagnostics"]["representation.relative_coordinate_drift"], 0)
        self.assertIn("horizon_1.prediction.block_position_rmse", one["target"]["diagnostics"])
        self.assertIn("inverse.action_mse", one["source"]["diagnostics"])
        self.assertEqual(self.collector.call_count, 2)  # fixed holdout, then fresh experience
        self.assertEqual(self.planner.call_count, 4)  # source/target before and after update
        self.calibration.assert_called_once()
        self.assert_sources_unchanged()
        learner_state = torch.load(output / "online_state.pt", map_location="cpu", weights_only=False)
        self.assertEqual(learner_state["round"], 1)
        self.assertEqual(learner_state["learner"]["update_steps"], 2)
        shard = output / "data/round_001/experience.h5"
        manifest = json.loads((shard.parent / "manifest.json").read_text())
        self.assertEqual(manifest["sha256"], digest(shard))
        self.assertFalse(list(output.rglob("*.tmp")))

    def test_planning_swmr_reader_opens_after_diagnostic_readers(self):
        # stable-worldmodel opens HDF5 with swmr=True; diagnostics use ordinary
        # immutable readers. HDF5 rejects simultaneous opens with different flags.
        def planning_reader(model, dataset_path, *args, **kwargs):
            with h5py.File(dataset_path, "r", swmr=True) as handle:
                self.assertGreater(len(handle["pixels"]), 0)
            return {"success_rate": 0.0}

        self.planner.side_effect = planning_reader
        output = self.root / "online"
        self.execute(output)
        self.assertEqual(self.planner.call_count, 4)
        self.assertTrue((output / "metrics/round_001.json").is_file())
        self.assert_sources_unchanged()

    def test_completed_resume_does_not_collect_update_or_rewrite_artifacts(self):
        output = self.root / "online"
        self.execute(output)
        before = {path: digest(path) for path in output.rglob("*") if path.is_file()}
        self.collector.reset_mock()
        self.planner.reset_mock()
        with mock.patch.object(run.OnlineLearner, "update", side_effect=AssertionError("duplicate update")):
            self.execute(output, resume=True)
        self.collector.assert_not_called()
        self.planner.assert_not_called()
        self.assertEqual(before, {path: digest(path) for path in output.rglob("*") if path.is_file()})
        self.assert_sources_unchanged()

    def test_zero_round_screen_never_trains_or_collects_training_experience(self):
        self.cfg["rounds"] = 0
        output = self.root / "frozen_screen"
        with mock.patch.object(run.OnlineLearner, "update", side_effect=AssertionError("zero-shot trained")):
            self.execute(output)
        result = json.loads((output / "metrics/round_000.json").read_text())
        self.assertEqual(result["updates"], 0)
        self.assertEqual(result["experience_transitions"], 0)
        self.assertEqual(self.collector.call_count, 1)  # target holdout only
        self.assertFalse((output / "data/round_001").exists())
        self.assertFalse((output / "metrics/round_001.json").exists())
        self.assert_sources_unchanged()

    def test_matched_source_control_uses_same_collector_and_seed(self):
        self.cfg["rounds"] = 0
        self.cfg["source_evaluation"] = "matched_collection"
        output = self.root / "matched_screen"
        self.execute(output)
        self.assertEqual(self.collector.call_count, 2)
        target_call, source_call = self.collector.call_args_list
        self.assertEqual(target_call.kwargs["seed"], source_call.kwargs["seed"])
        self.assertEqual(target_call.kwargs["policy_config"], source_call.kwargs["policy_config"])
        self.assertEqual(target_call.kwargs["episodes"], source_call.kwargs["episodes"])
        self.assertIn("block_cog", target_call.args[1])
        self.assertNotIn("block_cog", source_call.args[1])
        tasks = json.loads((output / "evaluation_tasks.json").read_text())
        self.assertEqual(tasks["source_split"], tasks["target_split"])
        self.assertIn("source_holdout", tasks["datasets"]["source"])
        self.assertIn("target_holdout", tasks["datasets"]["target"])
        self.assert_sources_unchanged()

    def test_two_rounds_keep_optimizer_momentum_with_source_replay(self):
        self.cfg["rounds"] = 2
        self.cfg["replay"] = {"recent_rounds": 1, "source_fraction": 0.25}
        output = self.root / "online"
        self.execute(output)
        final = torch.load(output / "online_state.pt", map_location="cpu", weights_only=False)
        self.assertEqual(final["round"], 2)
        self.assertEqual(final["experience_transitions"], 44)
        self.assertEqual(final["learner"]["update_steps"], 4)
        for state in final["learner"]["optimizer_state"]["state"].values():
            self.assertEqual(int(state["step"]), 4)
        self.assertEqual(self.collector.call_count, 3)
        self.assertEqual(self.planner.call_count, 6)
        self.assert_sources_unchanged()

    def test_resume_after_committed_collection_reuses_shard_and_matches_uninterrupted_model(self):
        expected_output = self.root / "uninterrupted"
        self.execute(expected_output)
        expected = self.checkpoint(expected_output, 1)
        interrupted = self.root / "interrupted"
        with mock.patch.object(run.OnlineLearner, "update", side_effect=RuntimeError("simulated interruption")):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                self.execute(interrupted)
        shard = interrupted / "data/round_001/experience.h5"
        self.assertTrue(shard.is_file())
        shard_before = digest(shard)
        checkpoint = torch.load(interrupted / "online_state.pt", map_location="cpu", weights_only=False)
        self.assertEqual(checkpoint["round"], 0)
        self.assertEqual(checkpoint["learner"]["update_steps"], 0)
        self.collector.reset_mock()
        self.execute(interrupted, resume=True)
        self.collector.assert_not_called()
        self.assertEqual(digest(shard), shard_before)
        for key, value in self.checkpoint(interrupted, 1).items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        self.assert_sources_unchanged()

    def test_two_block_state_runs_with_source_replay_and_separate_diagnostics(self):
        def two_block_shard(path, env, **kwargs):
            result = synthetic_shard(path, kwargs["episodes"])
            with h5py.File(path, "r+") as handle:
                primary = handle["state"][:]
                secondary = np.stack((primary[:, 2]+60, primary[:, 3]+50, primary[:, 4]+.2), -1)
                del handle["state"]
                handle["state"] = np.concatenate((primary[:, :5], secondary, primary[:, -2:]), -1)
            return result
        self.collector.side_effect = two_block_shard
        self.cfg["replay"]["source_fraction"] = .25
        self.cfg["planning"]["enabled"] = False
        output = self.root / "two_block"
        self.execute(output)
        result = json.loads((output / "metrics/round_001.json").read_text())
        self.assertIn("horizon_1.prediction.block2_position_rmse", result["target"]["diagnostics"])
        self.assertNotIn("observed.block2_position_rmse", result["source"]["diagnostics"])
        self.assertEqual(result["updates"], 2)
        self.assert_sources_unchanged()

    def test_reuse_rejects_corrupted_immutable_shard(self):
        output = self.root / "online"
        self.execute(output)
        shard = output / "data/target_holdout/experience.h5"
        with h5py.File(shard, "r+") as handle:
            handle["action"][0, 0] += 1
        self.collector.reset_mock()
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            self.execute(output, resume=True)
        self.collector.assert_not_called()
        self.assert_sources_unchanged()

    def test_nonempty_output_or_changed_config_requires_new_run(self):
        output = self.root / "online"
        self.execute(output)
        with self.assertRaises(FileExistsError):
            self.execute(output)
        self.cfg = copy.deepcopy(self.cfg)
        self.cfg["training"]["learning_rate"] *= 2
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            self.execute(output, resume=True)
        self.assert_sources_unchanged()


if __name__ == "__main__":
    unittest.main()
