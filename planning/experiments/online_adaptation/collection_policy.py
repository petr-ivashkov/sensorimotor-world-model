"""The upstream weak collector, visiting each movable block in turn.

For a single block this delegates to the original policy unchanged. With two,
the first target is randomized each episode and changes every 50 raw controls.
The target only constrains the same random displacement as upstream; no world
model, reward, uncertainty or desired goal pose selects these actions.
"""
import numpy as np
from stable_worldmodel.envs.pusht import WeakPolicy


class MultiBlockWeakPolicy(WeakPolicy):
    def __init__(self, *args, target_switch_steps=50, **kwargs):
        if int(target_switch_steps) < 1:
            raise ValueError("target_switch_steps must be positive")
        self.target_switch_steps = int(target_switch_steps)
        self._first_targets = {}
        super().__init__(*args, **kwargs)

    def get_action(self, info_dict, **kwargs):
        base = self.env.unwrapped
        envs = [e.unwrapped for e in base.envs] if hasattr(base, "envs") else [base]
        if all(not hasattr(env, "movable_blocks") for env in envs):
            return super().get_action(info_dict, **kwargs)
        steps = np.asarray(info_dict.get("step_idx", np.zeros(len(envs)))).reshape(len(envs), -1)[:, -1]
        actions = np.zeros(self.env.action_space.shape, dtype=np.float32)
        for index, env in enumerate(envs):
            blocks = getattr(env, "movable_blocks", (env.block,))
            if steps[index] == 0 or index not in self._first_targets:
                self._first_targets[index] = int(self.rng.integers(len(blocks)))
            target = (self._first_targets[index] + int(steps[index]) // self.target_switch_steps) % len(blocks)
            env.collector_target_block = target
            point = np.asarray(env.agent.position) + env.action_scale * self.rng.uniform(-1, 1, 2)
            center = np.asarray(blocks[target].position)
            point = np.clip(point, center-self.dist_constraint, center+self.dist_constraint)
            actions[index] = np.clip((point-np.asarray(env.agent.position))/env.action_scale, -1, 1)
        return actions
