"""
structured_dot_world.py
=======================
Extended Dot World dataset with structured motion types.

Four motion modes per dot group:

  INDEPENDENT  – each dot has its own (dx, dy) action entry.
  STATIC       – dots are placed once and never move.
  RANDOM       – dots move randomly each step; not in the action vector.
  COUPLED      – dot pairs share a single (dx, dy) action entry.
                 Two dots are displaced identically.

The action vector contains only controllable entries (INDEPENDENT + COUPLED).
STATIC and RANDOM motion is invisible to the action interface, so the
effective controllable dimensionality equals action_dim.

Controllable dimensionality by config:
  k independent dots                       → action_dim = 2k
  k independent + n static dots            → action_dim = 2k
  k independent + n randomly moving dots   → action_dim = 2k
  p coupled pairs  (2p total dots)         → action_dim = 2p
  k independent + p coupled + n random     → action_dim = 2k + 2p

Default canvas: 64×64 pixels, dot_radius=2, max_displacement=16.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────
#  Palette
# ─────────────────────────────────────────────────────────────────

DEFAULT_PALETTE: List[Tuple[int, int, int]] = [
    (255,   0,   0),   # 0  red
    (  0, 220,   0),   # 1  green
    (  0, 100, 255),   # 2  blue
    (255, 220,   0),   # 3  yellow
    (255,   0, 255),   # 4  magenta
    (  0, 220, 220),   # 5  cyan
    (255, 128,   0),   # 6  orange
    (160,   0, 255),   # 7  purple
    (255, 255, 255),   # 8  white
    (128, 255,   0),   # 9  lime
    (255,  80, 160),   # 10 pink
    (  0, 200, 140),   # 11 teal
    (200, 200, 200),   # 12 light grey
    (255, 200, 120),   # 13 peach
    ( 80, 200, 255),   # 14 sky
    (200, 100,  50),   # 15 brown
]


# ─────────────────────────────────────────────────────────────────
#  Motion types
# ─────────────────────────────────────────────────────────────────

class MotionType(Enum):
    INDEPENDENT = auto()   # own (dx, dy) per dot; contributes 2 action dims per dot
    STATIC      = auto()   # never moves; contributes 0 action dims
    RANDOM      = auto()   # moves each step without action supervision; contributes 0
    COUPLED     = auto()   # pairs of dots share one (dx, dy); contributes 2 per pair


# ─────────────────────────────────────────────────────────────────
#  Dot group specification
# ─────────────────────────────────────────────────────────────────

@dataclass
class DotGroup:
    """One group of dots that share the same MotionType.

    Parameters
    ----------
    motion_type:
        How dots in this group move.
    num_dots:
        Number of dots.  For COUPLED, must be even (pairs = num_dots // 2).
    color_indices:
        Palette indices, one per dot.  Length must equal num_dots.
    max_displacement:
        Maximum |dx| and |dy| per step for INDEPENDENT / RANDOM / COUPLED dots.
        Has no effect for STATIC groups.
    """
    motion_type: MotionType
    num_dots: int
    color_indices: List[int]
    max_displacement: int = 16

    def __post_init__(self) -> None:
        if len(self.color_indices) != self.num_dots:
            raise ValueError(
                f"color_indices has {len(self.color_indices)} entries "
                f"but num_dots={self.num_dots}"
            )
        if self.motion_type is MotionType.COUPLED and self.num_dots % 2 != 0:
            raise ValueError(
                f"COUPLED group requires even num_dots, got {self.num_dots}. "
                f"Dots are paired as (0,1), (2,3), …"
            )

    @property
    def action_dim(self) -> int:
        """Number of scalar action values this group contributes."""
        if self.motion_type is MotionType.INDEPENDENT:
            return self.num_dots * 2
        if self.motion_type is MotionType.COUPLED:
            return (self.num_dots // 2) * 2   # one (dx, dy) per pair
        return 0                               # STATIC and RANDOM are uncontrolled

    @property
    def num_pairs(self) -> int:
        """Number of coupled pairs (only meaningful for COUPLED groups)."""
        return self.num_dots // 2


# ─────────────────────────────────────────────────────────────────
#  World configuration
# ─────────────────────────────────────────────────────────────────

@dataclass
class StructuredDotWorldConfig:
    """Full environment specification.

    Attributes
    ----------
    groups:
        Ordered list of DotGroup specs.  Dots are stacked in group order
        into the position array, so group k occupies rows [start_k, end_k).
    image_size:
        Canvas side length in pixels (square canvas).
    dot_radius:
        Radius in pixels for each dot.
    allow_overlap:
        If False (default), use rejection sampling to ensure no two dots
        share any pixels in either the start or end frame.
    palette:
        RGB triples used to colour dots.  colour_indices in each DotGroup
        index into this list.
    """
    groups: List[DotGroup]
    image_size: int = 64
    dot_radius: int = 2
    allow_overlap: bool = False
    palette: List[Tuple[int, int, int]] = field(
        default_factory=lambda: list(DEFAULT_PALETTE)
    )

    def __post_init__(self) -> None:
        if not self.groups:
            raise ValueError("At least one DotGroup is required.")

    # ---- derived properties ----------------------------------------

    @property
    def num_dots(self) -> int:
        return sum(g.num_dots for g in self.groups)

    @property
    def action_dim(self) -> int:
        return sum(g.action_dim for g in self.groups)

    @property
    def min_separation(self) -> int:
        """Minimum centre-to-centre pixel distance for non-overlapping dots."""
        return 2 * self.dot_radius + 1

    @property
    def max_displacement(self) -> int:
        """Largest max_displacement across all groups (used for margin calc)."""
        return max(g.max_displacement for g in self.groups)

    @property
    def margin(self) -> int:
        """Safe border so any displacement keeps every dot fully inside canvas."""
        return self.dot_radius + self.max_displacement

    # ---- group index ranges ----------------------------------------

    def group_ranges(self) -> List[Tuple[int, int]]:
        """List of (start, end) row indices into the flat position array."""
        ranges: List[Tuple[int, int]] = []
        offset = 0
        for g in self.groups:
            ranges.append((offset, offset + g.num_dots))
            offset += g.num_dots
        return ranges

    # ---- human-readable summary ------------------------------------

    def describe(self) -> str:
        lines = [
            f"StructuredDotWorldConfig  |  {self.image_size}×{self.image_size} px  "
            f"|  radius={self.dot_radius}  |  action_dim={self.action_dim}"
        ]
        for i, g in enumerate(self.groups):
            lines.append(
                f"  group {i}: {g.num_dots} dots  "
                f"motion={g.motion_type.name}  "
                f"action_dim={g.action_dim}  "
                f"max_disp={g.max_displacement}"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
#  Convenience constructors
# ─────────────────────────────────────────────────────────────────

def _auto_colors(num: int, offset: int = 0) -> List[int]:
    """Auto-assign palette indices starting at offset, cycling if needed."""
    n_palette = len(DEFAULT_PALETTE)
    return [(offset + i) % n_palette for i in range(num)]


def make_independent_static_config(
    num_independent: int = 2,
    num_static: int = 1,
    max_displacement: int = 16,
    **kwargs,
) -> StructuredDotWorldConfig:
    """k independently controlled dots + n motionless static dots.

    action_dim = 2 * num_independent
    Effective controllable dim = 2 * num_independent
    """
    groups: List[DotGroup] = []
    offset = 0
    if num_independent > 0:
        groups.append(DotGroup(
            motion_type=MotionType.INDEPENDENT,
            num_dots=num_independent,
            color_indices=_auto_colors(num_independent, offset),
            max_displacement=max_displacement,
        ))
        offset += num_independent
    if num_static > 0:
        groups.append(DotGroup(
            motion_type=MotionType.STATIC,
            num_dots=num_static,
            color_indices=_auto_colors(num_static, offset),
            max_displacement=1,   # irrelevant; set small to keep margin sane
        ))
    return StructuredDotWorldConfig(groups=groups, **kwargs)


def make_independent_random_config(
    num_independent: int = 2,
    num_random: int = 1,
    max_displacement: int = 16,
    **kwargs,
) -> StructuredDotWorldConfig:
    """k controlled dots + n randomly moving (uncontrolled) dots.

    action_dim = 2 * num_independent
    Effective controllable dim = 2 * num_independent
    """
    groups: List[DotGroup] = []
    offset = 0
    if num_independent > 0:
        groups.append(DotGroup(
            motion_type=MotionType.INDEPENDENT,
            num_dots=num_independent,
            color_indices=_auto_colors(num_independent, offset),
            max_displacement=max_displacement,
        ))
        offset += num_independent
    if num_random > 0:
        groups.append(DotGroup(
            motion_type=MotionType.RANDOM,
            num_dots=num_random,
            color_indices=_auto_colors(num_random, offset),
            max_displacement=max_displacement,
        ))
    return StructuredDotWorldConfig(groups=groups, **kwargs)


def make_coupled_config(
    num_pairs: int = 2,
    max_displacement: int = 16,
    **kwargs,
) -> StructuredDotWorldConfig:
    """p coupled dot pairs  →  2p total dots, action_dim = 2p.

    Each pair shares one (dx, dy) and moves rigidly together.
    Effective controllable dim = 2 * num_pairs (same as one independent dot per pair).
    """
    if num_pairs < 1:
        raise ValueError("num_pairs must be >= 1")
    n = num_pairs * 2
    groups = [DotGroup(
        motion_type=MotionType.COUPLED,
        num_dots=n,
        color_indices=_auto_colors(n),
        max_displacement=max_displacement,
    )]
    return StructuredDotWorldConfig(groups=groups, **kwargs)


def make_combined_config(
    num_independent: int = 1,
    num_coupled_pairs: int = 1,
    num_random: int = 1,
    max_displacement: int = 16,
    **kwargs,
) -> StructuredDotWorldConfig:
    """k independent + p coupled pairs + n random dots.

    action_dim = 2 * num_independent + 2 * num_coupled_pairs
    Effective controllable dim = 2k + 2p
    """
    groups: List[DotGroup] = []
    offset = 0
    if num_independent > 0:
        groups.append(DotGroup(
            motion_type=MotionType.INDEPENDENT,
            num_dots=num_independent,
            color_indices=_auto_colors(num_independent, offset),
            max_displacement=max_displacement,
        ))
        offset += num_independent
    if num_coupled_pairs > 0:
        n = num_coupled_pairs * 2
        groups.append(DotGroup(
            motion_type=MotionType.COUPLED,
            num_dots=n,
            color_indices=_auto_colors(n, offset),
            max_displacement=max_displacement,
        ))
        offset += n
    if num_random > 0:
        groups.append(DotGroup(
            motion_type=MotionType.RANDOM,
            num_dots=num_random,
            color_indices=_auto_colors(num_random, offset),
            max_displacement=max_displacement,
        ))
    if not groups:
        raise ValueError("At least one non-empty group is required.")
    return StructuredDotWorldConfig(groups=groups, **kwargs)


# ─────────────────────────────────────────────────────────────────
#  Rendering
# ─────────────────────────────────────────────────────────────────

def render_dots(
    positions: np.ndarray,          # (num_dots, 2) int  — (x, y) pixel coords
    color_indices: List[int],        # length == num_dots
    config: StructuredDotWorldConfig,
) -> np.ndarray:
    """Render all dots onto a white canvas.

    Returns
    -------
    (3, H, W) float32 in [0, 1], channels-first (PyTorch convention).
    """
    H = W = config.image_size
    canvas = np.ones((3, H, W), dtype=np.float32)
    yy, xx = np.mgrid[:H, :W]

    for dot_idx, (cx, cy) in enumerate(positions):
        rgb = np.array(
            config.palette[color_indices[dot_idx] % len(config.palette)],
            dtype=np.float32,
        ) / 255.0
        dist_sq = (xx - int(cx)) ** 2 + (yy - int(cy)) ** 2
        mask = dist_sq <= config.dot_radius ** 2
        for c in range(3):
            canvas[c] = np.where(mask, rgb[c], canvas[c])

    return canvas


def _flat_color_indices(config: StructuredDotWorldConfig) -> List[int]:
    """Collect all color_indices across groups in dot order."""
    out: List[int] = []
    for g in config.groups:
        out.extend(g.color_indices)
    return out


# ─────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────

class StructuredDotWorldDataset(Dataset):
    """PyTorch dataset generating (obs_t, action, obs_tp1, positions_t) transitions.

    Samples are generated on-the-fly from a seeded RNG; the dataset is
    deterministic and reproducible given (seed, index).

    Returns
    -------
    obs_t      : (3, H, W) float32 — observation at time t
    action     : (action_dim,) float32 — controllable action (INDEPENDENT + COUPLED)
    obs_tp1    : (3, H, W) float32 — observation at time t+1
    positions_t: (num_dots * 2,) float32 — flattened (x, y) positions at t
    """

    _MAX_REJECTION_ATTEMPTS: int = 1000

    def __init__(
        self,
        config: Optional[StructuredDotWorldConfig] = None,
        num_samples: int = 10_000,
        seed: int = 0,
    ):
        self.config = config or make_combined_config()
        self.num_samples = num_samples
        self.seed = seed
        # Pre-compute fixed derived quantities.
        self._group_ranges: List[Tuple[int, int]] = self.config.group_ranges()
        self._color_indices: List[int] = _flat_color_indices(self.config)

    # ── helpers ──────────────────────────────────────────────────

    def _has_overlap(self, positions: np.ndarray) -> bool:
        """True if any two dots are closer than min_separation."""
        min_sep = self.config.min_separation
        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(positions[i] - positions[j]) < min_sep:
                    return True
        return False

    def _sample_initial_positions(
        self,
        rng: np.random.Generator,
        lo: int,
        hi: int,
    ) -> np.ndarray:
        """Sample (num_dots, 2) integer positions in [lo, hi]."""
        cfg = self.config
        if cfg.allow_overlap or cfg.num_dots == 1:
            return rng.integers(lo, hi, size=(cfg.num_dots, 2), endpoint=True)
        for _ in range(self._MAX_REJECTION_ATTEMPTS):
            pos = rng.integers(lo, hi, size=(cfg.num_dots, 2), endpoint=True)
            if not self._has_overlap(pos):
                return pos
        raise RuntimeError(
            f"Could not place {cfg.num_dots} non-overlapping dots "
            f"(radius={cfg.dot_radius}) on a {cfg.image_size}×{cfg.image_size} "
            f"canvas after {self._MAX_REJECTION_ATTEMPTS} attempts.  "
            f"Try fewer dots, a smaller radius, or a larger canvas."
        )

    def _sample_transition(
        self,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample one (positions_t, action, positions_tp1) transition.

        The margin property of the config ensures that no bounds check is
        needed for new_positions: initial positions are placed at least
        max_displacement pixels from every edge, so every possible
        displacement stays inside the canvas.

        Returns
        -------
        positions_t  : (num_dots, 2) int
        action       : (action_dim,) float32
        positions_tp1: (num_dots, 2) int
        """
        cfg = self.config
        lo = cfg.margin
        hi = cfg.image_size - 1 - cfg.margin

        if lo >= hi:
            raise RuntimeError(
                f"Margin ({cfg.margin}) is too large for canvas size "
                f"({cfg.image_size}).  Reduce dot_radius or max_displacement."
            )

        positions_t = self._sample_initial_positions(rng, lo, hi)

        # Attempt to find displacements that avoid overlap at t+1.
        for _ in range(self._MAX_REJECTION_ATTEMPTS):
            displacements = np.zeros((cfg.num_dots, 2), dtype=np.int32)
            action_parts: List[np.ndarray] = []

            for group, (start, end) in zip(cfg.groups, self._group_ranges):
                md = group.max_displacement

                if group.motion_type is MotionType.INDEPENDENT:
                    disp = rng.integers(-md, md, size=(group.num_dots, 2), endpoint=True)
                    displacements[start:end] = disp
                    action_parts.append(disp.reshape(-1).astype(np.float32))

                elif group.motion_type is MotionType.STATIC:
                    pass  # zero displacement, no action entry

                elif group.motion_type is MotionType.RANDOM:
                    disp = rng.integers(-md, md, size=(group.num_dots, 2), endpoint=True)
                    displacements[start:end] = disp
                    # no action entry – uncontrolled

                elif group.motion_type is MotionType.COUPLED:
                    # Sample one (dx, dy) per pair; apply to both dots.
                    pair_disps = rng.integers(
                        -md, md, size=(group.num_pairs, 2), endpoint=True
                    )
                    for p_idx in range(group.num_pairs):
                        displacements[start + 2 * p_idx]     = pair_disps[p_idx]
                        displacements[start + 2 * p_idx + 1] = pair_disps[p_idx]
                    action_parts.append(pair_disps.reshape(-1).astype(np.float32))

            positions_tp1 = positions_t + displacements

            # Accept if overlap is not a concern or no overlap found.
            if cfg.allow_overlap or cfg.num_dots == 1 or not self._has_overlap(positions_tp1):
                break
        else:
            raise RuntimeError(
                f"Could not find non-overlapping displacements for "
                f"{cfg.num_dots} dots after {self._MAX_REJECTION_ATTEMPTS} "
                f"attempts.  Try fewer dots, smaller radius, or larger canvas."
            )

        if action_parts:
            action = np.concatenate(action_parts)
        else:
            # Fully uncontrolled world (all-static or all-random).
            action = np.zeros(0, dtype=np.float32)

        return positions_t, action, positions_tp1

    # ── Dataset interface ────────────────────────────────────────

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed + idx)
        positions_t, action, positions_tp1 = self._sample_transition(rng)

        obs_t   = render_dots(positions_t,   self._color_indices, self.config)
        obs_tp1 = render_dots(positions_tp1, self._color_indices, self.config)

        return (
            torch.from_numpy(obs_t),
            torch.from_numpy(action),
            torch.from_numpy(obs_tp1),
            torch.from_numpy(positions_t.reshape(-1).astype(np.float32)),
        )


# ─────────────────────────────────────────────────────────────────
#  Trajectory generation
# ─────────────────────────────────────────────────────────────────

def generate_trajectory(
    config: StructuredDotWorldConfig,
    num_steps: int = 10,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll out a multi-step trajectory under the given config.

    Returns
    -------
    observations : (num_steps + 1, 3, H, W) float32
    actions      : (num_steps, action_dim) float32
    positions    : (num_steps + 1, num_dots, 2) int32
    """
    ds = StructuredDotWorldDataset(config=config, num_samples=0, seed=0)
    rng = np.random.default_rng(seed)
    lo = config.margin
    hi = config.image_size - 1 - config.margin
    color_indices = _flat_color_indices(config)

    pos = ds._sample_initial_positions(rng, lo, hi)

    observations = [render_dots(pos, color_indices, config)]
    actions: List[np.ndarray] = []
    all_positions = [pos.copy()]

    for _ in range(num_steps):
        # Clamp the *displacement* (not the resulting position per dot) so that
        # coupled pairs keep their shared offset: a single clamped (dx, dy) is
        # written to both dots of a pair.  Clamping the position independently
        # would let one dot hit the wall while its partner keeps moving, tearing
        # the pair apart.  The recorded action is the realized (clamped)
        # displacement, so it always matches the transition.
        displacements = np.zeros((config.num_dots, 2), dtype=np.int32)
        action_parts: List[np.ndarray] = []
        for group, (start, end) in zip(config.groups, ds._group_ranges):
            md = group.max_displacement
            if group.motion_type is MotionType.INDEPENDENT:
                disp = rng.integers(-md, md, size=(group.num_dots, 2), endpoint=True)
                disp = np.clip(disp, lo - pos[start:end], hi - pos[start:end])
                displacements[start:end] = disp
                action_parts.append(disp.reshape(-1).astype(np.float32))
            elif group.motion_type is MotionType.STATIC:
                pass
            elif group.motion_type is MotionType.RANDOM:
                disp = rng.integers(-md, md, size=(group.num_dots, 2), endpoint=True)
                displacements[start:end] = np.clip(
                    disp, lo - pos[start:end], hi - pos[start:end]
                )
            elif group.motion_type is MotionType.COUPLED:
                pair_disps = rng.integers(
                    -md, md, size=(group.num_pairs, 2), endpoint=True
                )
                for p_idx in range(group.num_pairs):
                    a = start + 2 * p_idx
                    b = a + 1
                    # Clamp the shared displacement so BOTH dots stay in bounds.
                    low  = lo - np.minimum(pos[a], pos[b])
                    high = hi - np.maximum(pos[a], pos[b])
                    d = np.clip(pair_disps[p_idx], low, high)
                    displacements[a] = d
                    displacements[b] = d
                    action_parts.append(d.astype(np.float32))

        pos = pos + displacements
        action_vec = (
            np.concatenate(action_parts) if action_parts
            else np.zeros(0, dtype=np.float32)
        )
        actions.append(action_vec)
        observations.append(render_dots(pos, color_indices, config))
        all_positions.append(pos.copy())

    return (
        np.stack(observations),                         # (T+1, 3, H, W)
        np.stack(actions) if actions else np.empty((0, config.action_dim), dtype=np.float32),
        np.stack(all_positions).astype(np.int32),       # (T+1, num_dots, 2)
    )
