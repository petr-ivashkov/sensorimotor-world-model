"""Original Push-T plus a second independently movable, coral-colored T.

Both bodies use the upstream T constructor, mass, material and collision rules.
Actions remain 2D and proprioception remains 4D. Diagnostic state is now
[agent_x, agent_y, T1_x, T1_y, T1_angle, T2_x, T2_y, T2_angle, agent_vx, agent_vy].
The extra state is never an input to world-model training.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces
from stable_worldmodel.envs.pusht.env import PushT

TWO_BLOCK_ENV_ID = "swm/PushT-TwoBlock-v0"
ENTRY_POINT = "planning.experiments.online_adaptation.two_block_env:PushTwoBlock"


class PushTwoBlock(PushT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_name = "PushT-TwoBlock"
        # The last two entries stay agent velocity, as required by upstream
        # reset/step/proprioception. Only physical-state diagnostics change.
        self.observation_space.spaces["state"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64,
        )
        self.block2 = None
        self._agent_contacts = np.zeros(2, dtype=np.int64)
        self._block_contact = 0
        self.collector_target_block = 0

    def _setup(self):
        super()._setup()
        self.block2 = self.add_tee(position=(256, 256), angle=0., color="LightCoral")
        self._agent_contacts[:] = 0
        self._block_contact = 0
        self.collector_target_block = 0

    @property
    def movable_blocks(self):
        return (self.block, self.block2)

    def _get_obs(self):
        primary = super()._get_obs()
        return np.concatenate((primary[:5], [*self.block2.position,
                               self.block2.angle % (2*np.pi)], primary[-2:]))

    def _second_pose_is_valid(self):
        # Reindex after teleporting; cached Pymunk bounding boxes otherwise
        # describe the previous pose. Use actual polygons, including rotation.
        for body in (self.agent, self.block, self.block2):
            self.space.reindex_shapes_for_body(body)
        for shape in self.block2.shapes:
            bb = shape.bb
            if min(bb.left, bb.bottom) < 10 or max(bb.right, bb.top) > 502:
                return False
            # Pymunk 7.3 shapes_collide can assert for disjoint polygons with
            # overlapping AABBs. The space query only returns real overlaps.
            if any(hit.shape.body in (self.agent, self.block)
                   for hit in self.space.shape_query(shape)):
                return False
        return True

    def _set_state(self, state):
        state = np.asarray(state, dtype=np.float64)
        if state.shape not in {(7,), (10,)} or not np.isfinite(state).all():
            raise ValueError("Two-block reset needs a finite 7D primary or full 10D state")
        self.agent.position = tuple(state[:2])
        self.agent.velocity = tuple(state[-2:])
        # Keep upstream assignment order for its off-origin T center of mass.
        self.block.angle = float(state[4])
        self.block.position = tuple(state[2:4])
        self.block2.velocity = (0, 0)
        self.block2.angular_velocity = 0
        if state.size == 10:
            self.block2.angle = float(state[7])
            self.block2.position = tuple(state[5:7])
        else:
            # Preserve the original primary/agent reset distribution. Reject
            # secondary poses intersecting them or extending through a wall.
            for _ in range(1000):
                position = tuple(self.rng.uniform(100, 400, size=2))
                self.block2.angle = float(self.rng.uniform(-2*np.pi, 2*np.pi))
                self.block2.position = position
                if self._second_pose_is_valid():
                    break
            else:
                raise RuntimeError("Could not sample a collision-free second T")
        for body in (self.agent, self.block, self.block2):
            self.space.reindex_shapes_for_body(body)
        self.space.step(self.dt)

    def _set_goal_state(self, goal_state):
        goal_state = np.asarray(goal_state, dtype=np.float64)
        if goal_state.size == 7:
            goal_state = np.concatenate((goal_state[:5], [*self.block2.position,
                                         self.block2.angle % (2*np.pi)], goal_state[-2:]))
        self.goal_state = goal_state.copy()

    def eval_state(self, goal_state, cur_state):
        goal_state, cur_state = np.asarray(goal_state), np.asarray(cur_state)
        success, _ = super().eval_state(
            np.r_[goal_state[:5], goal_state[-2:]], np.r_[cur_state[:5], cur_state[-2:]],
        )
        angle = abs(np.arctan2(np.sin(goal_state[7]-cur_state[7]),
                               np.cos(goal_state[7]-cur_state[7])))
        success = bool(success and np.linalg.norm(goal_state[5:7]-cur_state[5:7]) < 20
                       and angle < np.pi/9)
        return success, float(np.linalg.norm(goal_state-cur_state))

    def step(self, action):
        self._agent_contacts[:] = 0
        self._block_contact = 0
        return super().step(action)

    def _handle_collision(self, arbiter, space, data):
        super()._handle_collision(arbiter, space, data)
        bodies = {shape.body for shape in arbiter.shapes}
        for index, block in enumerate(self.movable_blocks):
            if self.agent in bodies and block in bodies:
                self._agent_contacts[index] = 1
        if self.block in bodies and self.block2 in bodies:
            self._block_contact = 1

    def _get_info(self):
        info = super()._get_info()
        info.update(block2_pose=np.array([*self.block2.position, self.block2.angle]),
                    agent_block1_contact=int(self._agent_contacts[0]),
                    agent_block2_contact=int(self._agent_contacts[1]),
                    block_block_contact=int(self._block_contact),
                    collector_target_block=int(self.collector_target_block))
        return info


def register_two_block_environment():
    import gymnasium as gym
    existing = gym.envs.registry.get(TWO_BLOCK_ENV_ID)
    if existing is None:
        gym.register(id=TWO_BLOCK_ENV_ID, entry_point=ENTRY_POINT)
    elif existing.entry_point != ENTRY_POINT:
        raise ValueError(f"Environment ID {TWO_BLOCK_ENV_ID!r} belongs to another implementation")
    return TWO_BLOCK_ENV_ID
