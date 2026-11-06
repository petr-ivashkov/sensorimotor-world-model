"""Synthetic HDF5 checks for episode-safe online adaptation data."""

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import torch

from planning.experiments.online_adaptation.data import (
    ExperienceDataset,
    action_statistics,
    episode_metadata,
    fixed_tasks,
    split_episodes,
)


def write_shard(path, lengths=(8, 7, 3), invalid_action_row=None):
    lengths = np.asarray(lengths, dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths[:-1])))
    rows = int(lengths.sum())
    row = np.arange(rows, dtype=np.float64)
    actions = np.stack((row + 1, 3 * row + 2), axis=1)
    actions[offsets + lengths - 1] = np.nan
    if invalid_action_row is not None:
        actions[invalid_action_row, 1] = np.nan
    with h5py.File(path, "w") as handle:
        handle["ep_len"] = lengths
        handle["ep_offset"] = offsets
        handle["action"] = actions
        handle["state"] = np.repeat(row[:, None], 7, axis=1)
        handle["pixels"] = np.broadcast_to(
            np.arange(rows, dtype=np.uint8)[:, None, None, None], (rows, 2, 2, 3)
        )
        handle["ep_idx"] = np.repeat(np.arange(len(lengths)), lengths)
        handle["step_idx"] = np.concatenate([np.arange(n) for n in lengths])
    return actions


def numpy_collate(items):
    # Exercise actual worker HDF5 access without requiring torch_shm_manager,
    # which the local execution sandbox does not permit launching.
    return {key: np.stack([item[key].numpy() for item in items]) for key in items[0]}


class ExperienceDataTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "experience.h5"
        self.actions = write_shard(self.path)
        self.statistics = {"mean": [2.5, 7.0], "scale": [2.0, 3.0]}

    def dataset(self, **kwargs):
        dataset = ExperienceDataset(self.path, self.statistics, **kwargs)
        self.addCleanup(dataset.close)
        return dataset

    def test_source_statistics_match_original_training_float64_sample_std(self):
        result = action_statistics(self.path)
        original = torch.from_numpy(self.actions)
        original = original[~torch.isnan(original).any(dim=1)]
        torch.testing.assert_close(torch.tensor(result["mean"], dtype=torch.float64),
                                   original.mean(0), rtol=0, atol=0)
        torch.testing.assert_close(torch.tensor(result["scale"], dtype=torch.float64),
                                   original.std(0), rtol=0, atol=0)
        self.assertEqual(result["ddof"], 1)
        self.assertEqual(result["count"], len(original))
        # Population normalization would silently change every source action.
        self.assertFalse(np.array_equal(result["scale"], original.std(0, correction=0).numpy()))

    def test_normalize_raw_actions_before_grouping_and_stride_pixels(self):
        dataset = self.dataset(frameskip=2, num_steps=3, episodes=[0])
        item = dataset[1]
        expected_raw = self.actions[1:5]
        expected_actions = ((expected_raw - self.statistics["mean"])
                            / self.statistics["scale"]).astype(np.float32).reshape(2, 4)
        torch.testing.assert_close(item["action"][:-1], torch.from_numpy(expected_actions), rtol=0, atol=0)
        torch.testing.assert_close(item["action"][-1], torch.zeros(4))
        self.assertEqual(item["pixels"].shape, (3, 3, 2, 2))
        self.assertEqual(item["pixels"][:, 0, 0, 0].tolist(), [1, 3, 5])
        self.assertEqual(item["state"][:, 0].tolist(), [1, 3, 5])
        self.assertEqual(int(item["ep_idx"]), 0)
        self.assertEqual(int(item["step_idx"]), 1)

    def test_clips_never_cross_episodes_and_terminal_nan_is_unsupervised(self):
        dataset = self.dataset(frameskip=2, num_steps=3)
        self.assertEqual(dataset.clips, [(0, 0), (0, 1), (0, 2), (0, 3),
                                         (1, 0), (1, 1), (1, 2)])
        for index, (episode, start) in enumerate(dataset.clips):
            item = dataset[index]
            rows = item["state"][:, 0].numpy().astype(int)
            self.assertTrue((rows >= dataset.offsets[episode]).all())
            self.assertTrue((rows < dataset.offsets[episode] + dataset.lengths[episode]).all())
            self.assertTrue(torch.isfinite(item["action"]).all())
            self.assertEqual(rows[-1], dataset.offsets[episode] + start + 4)
        # The last observation may be terminal: its NaN outgoing action is padding.
        endpoint = dataset[3]
        self.assertEqual(endpoint["state"][-1, 0].item(), 7)
        self.assertTrue(np.isnan(self.actions[7]).all())
        self.assertTrue(torch.equal(endpoint["action"][-1], torch.zeros(4)))

    def test_nonfinite_executed_actions_remove_only_affected_clips(self):
        write_shard(self.path, invalid_action_row=2)
        dataset = self.dataset(frameskip=2, num_steps=3)
        self.assertEqual(dataset.clips, [(0, 3), (1, 0), (1, 1), (1, 2)])
        self.assertTrue(all(torch.isfinite(dataset[i]["action"]).all() for i in range(len(dataset))))

    def test_short_episodes_and_deterministic_clip_subsampling(self):
        short = self.dataset(frameskip=2, num_steps=3, episodes=[2])
        self.assertEqual(len(short), 0)
        left = self.dataset(frameskip=2, num_steps=3, max_clips=3, sample_seed=17)
        right = self.dataset(frameskip=2, num_steps=3, max_clips=3, sample_seed=17)
        self.assertEqual(len(left), 3)
        self.assertEqual(left.clips, right.clips)
        self.assertEqual(len(set(left.clips)), 3)

    def test_episode_split_is_disjoint_complete_and_reproducible(self):
        split = split_episodes(self.path, seed=91)
        self.assertEqual(split, split_episodes(self.path, seed=91))
        self.assertFalse(set(split["fit"]) & set(split["test"]))
        self.assertEqual(set(split["fit"]) | set(split["test"]), {0, 1, 2})
        for fraction in (0, 1, -0.5):
            with self.assertRaises(ValueError):
                split_episodes(self.path, seed=91, fit_fraction=fraction)
        single = Path(self.temporary.name) / "single.h5"
        write_shard(single, lengths=(8,))
        with self.assertRaises(ValueError):
            split_episodes(single, seed=0)

    def test_fixed_tasks_stay_in_selected_episodes_and_use_pinned_goal_endpoint(self):
        result = fixed_tasks(self.path, episodes=[0, 1], count=2, goal_offset_steps=7, seed=10)
        self.assertEqual(result, fixed_tasks(self.path, episodes=[0, 1], count=2,
                                             goal_offset_steps=7, seed=10))
        self.assertEqual(set(result["episode_ids"]), {0, 1})
        for episode, start in zip(result["episode_ids"], result["start_steps"]):
            self.assertGreaterEqual(start, 0)
            # evaluate_from_dataset loads [start:start+offset] and takes last frame.
            self.assertLess(start + result["goal_offset_steps"] - 1, [8, 7, 3][episode])
        with self.assertRaises(ValueError):
            fixed_tasks(self.path, episodes=[2], count=1, goal_offset_steps=4, seed=0)

    def test_episode_and_task_validation_reject_invalid_or_repeated_ids(self):
        for episodes in ([-1], [3], [0, 0]):
            with self.subTest(episodes=episodes):
                with self.assertRaises(ValueError):
                    self.dataset(episodes=episodes)
                with self.assertRaises(ValueError):
                    fixed_tasks(self.path, episodes=episodes, count=1, goal_offset_steps=2, seed=0)
        for count, offset in ((0, 2), (1, 1)):
            with self.assertRaises(ValueError):
                fixed_tasks(self.path, episodes=[0], count=count, goal_offset_steps=offset, seed=0)

    def test_fixed_tasks_require_actual_object_motion(self):
        result = fixed_tasks(self.path, episodes=[0, 1], count=2,
                             goal_offset_steps=3, seed=5, min_block_displacement=2.5)
        with h5py.File(self.path, "r") as handle:
            for ep, start in zip(result["episode_ids"], result["start_steps"]):
                offset = int(handle["ep_offset"][ep])
                initial = handle["state"][offset+start, 2:4]
                goal = handle["state"][offset+start+2, 2:4]
                self.assertGreaterEqual(np.linalg.norm(goal-initial), 2.5)
        for minimum in (3.0, float("nan"), -1.0):
            with self.assertRaises(ValueError):
                fixed_tasks(self.path, episodes=[0, 1], count=1,
                            goal_offset_steps=3, seed=5, min_block_displacement=minimum)

    def test_invalid_episode_metadata_is_rejected(self):
        with h5py.File(self.path, "r+") as handle:
            handle["ep_offset"][1] = 9
        with self.assertRaisesRegex(ValueError, "contiguous"):
            episode_metadata(self.path)

    def test_open_dataset_can_be_pickled_without_sharing_hdf5_handle(self):
        dataset = self.dataset(frameskip=2, num_steps=3)
        expected = dataset[0]
        restored = pickle.loads(pickle.dumps(dataset))
        self.addCleanup(restored.close)
        self.assertIsNone(restored._handle)
        actual = restored[0]
        self.assertIsNot(restored._handle, dataset._handle)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])

    def test_pid_change_reopens_inherited_parent_hdf5_handle(self):
        dataset = self.dataset(frameskip=2, num_steps=3)
        expected = dataset[0]
        parent_handle = dataset._handle
        simulated_child_pid = dataset._pid + 1
        with mock.patch("planning.experiments.online_adaptation.data.os.getpid",
                        return_value=simulated_child_pid):
            actual = dataset[0]
        self.assertEqual(dataset._pid, simulated_child_pid)
        self.assertIsNot(dataset._handle, parent_handle)
        self.assertFalse(parent_handle.id.valid)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])

    def test_spawn_loader_workers_read_same_episode_safe_clips(self):
        dataset = self.dataset(frameskip=2, num_steps=3)
        # Open a parent handle first, as representation evaluation can do.
        expected = [dataset[index] for index in range(len(dataset))]
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=2,
                                            multiprocessing_context="spawn", shuffle=False,
                                            collate_fn=numpy_collate)
        batches = list(loader)
        for key in expected[0]:
            actual = torch.from_numpy(np.concatenate([item[key] for item in batches]))
            torch.testing.assert_close(actual, torch.stack([item[key] for item in expected]))


if __name__ == "__main__":
    unittest.main()
