"""
sprite_world.py
===============
Single asymmetric sprite with pose state (x, y, θ), rendered rotated and
anti-aliased on a 64×64 white canvas.  θ is observable only because the
sprite has no rotational symmetry.

Each step a delta (dx, dy, dθ) is sampled for ALL THREE degrees of freedom
from a matched per-DOF distribution and applied; θ wraps at 2π.  A
``control_mask`` over (x, y, θ) selects which of the three deltas are exposed
in the action vector.  Unselected DOFs still move — they are *uncontrolled* —
so the effective controllable dimensionality equals the number of selected
DOFs.

This mirrors the structured-dot-world idea (controlled vs. uncontrolled
motion) for a single rigid sprite whose orientation θ is the extra DOF.

Sprite shapes (``Shape``), all with no rotational symmetry so θ is uniquely
recoverable over [0, 2π):

  ARROW     isoceles arrowhead — strongest, cleanest orientation cue (default)
  TEARDROP  rounded base + tapered point
  PACMAN    disc with a wedge mouth
  HEART     classic implicit heart (weak orientation cue; kept for reference)

Four nested control configurations (control_mask over [x, y, θ]):

  NONE  [F, F, F]   action_dim = 0   fully uncontrolled
  X     [T, F, F]   action_dim = 1   only x exposed
  XY    [T, T, F]   action_dim = 2   x and y exposed
  XYT   [T, T, T]   action_dim = 3   all DOFs exposed

In every config all three DOFs move identically; only the action interface
shrinks.  An encoder minimizing inverse loss should therefore learn to
represent only the exposed DOFs.

Matched distribution: each DOF delta is drawn from the *same* Uniform(-1, 1)
and scaled by its own physical range (``delta_scale``), so the three DOFs are
statistically interchangeable up to scale — the only thing distinguishing a
controlled DOF from an uncontrolled one is the action mask, not its motion
statistics.

Returned sample (matches train.py's batch contract
``obs_t, action, obs_tp1, state_t``):

  obs_t    : (3, H, W) float32      — observation at t
  action   : (action_dim,) float32  — deltas of the controlled DOFs only
  obs_tp1  : (3, H, W) float32      — observation at t+1
  state_t  : (3,) float32           — pose (x, y, θ) at t

Note on normalization: x, y are pixels and θ is radians, so the action vector
mixes units.  train.py normalizes by a single ``max_displacement`` scalar; for
this world prefer per-DOF normalization via ``config.action_scale`` when
training is wired up (out of scope here).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .structured_dot_world import DEFAULT_PALETTE


# ─────────────────────────────────────────────────────────────────
#  Control configurations
# ─────────────────────────────────────────────────────────────────

DOF_LABELS: Tuple[str, str, str] = ("x", "y", "θ")


class ControlConfig(Enum):
    """Nested control masks over the three DOFs (x, y, θ).

    Each member's value is the boolean mask selecting which deltas enter the
    action vector.  The masks are nested: NONE ⊂ X ⊂ XY ⊂ XYT.
    """
    NONE = (False, False, False)   # action_dim = 0
    X    = (True,  False, False)   # action_dim = 1
    XY   = (True,  True,  False)   # action_dim = 2
    XYT  = (True,  True,  True)    # action_dim = 3

    @property
    def mask(self) -> np.ndarray:
        return np.array(self.value, dtype=bool)

    @property
    def action_dim(self) -> int:
        return int(np.count_nonzero(self.value))

    @property
    def controlled_labels(self) -> List[str]:
        """DOF names exposed in the action vector, in order."""
        return [DOF_LABELS[i] for i, on in enumerate(self.value) if on]


# ─────────────────────────────────────────────────────────────────
#  Sprite shapes
# ─────────────────────────────────────────────────────────────────
#
# Each shape is an inside-test on normalized local coords (u, v): the unit
# sprite in its canonical (un-rotated) frame, with +u the heading at θ=0 and
# v pointing 'up' in sprite space.  None has rotational symmetry.

def _convex_inside(u: np.ndarray, v: np.ndarray, verts) -> np.ndarray:
    """Vectorized inside-test for a CCW convex polygon."""
    inside = np.ones(u.shape, dtype=bool)
    n = len(verts)
    for i in range(n):
        x0, y0 = verts[i]
        x1, y1 = verts[(i + 1) % n]
        cross = (x1 - x0) * (v - y0) - (y1 - y0) * (u - x0)   # left of edge
        inside &= cross >= 0.0
    return inside


def _arrow(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Elongated isoceles arrowhead, apex pointing +u."""
    return _convex_inside(u, v, [(1.6, 0.0), (-0.9, 1.0), (-0.9, -1.0)])


def _teardrop(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rounded base (−u) tapering to a point at +u."""
    circ = (u + 0.3) ** 2 + v ** 2 <= 0.85 ** 2
    tri = _convex_inside(u, v, [(1.5, 0.0), (-0.3, 0.85), (-0.3, -0.85)])
    return circ | tri


def _pacman(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Disc of radius 1.2 with a ±0.6 rad wedge mouth opening toward +u."""
    r = np.hypot(u, v)
    ang = np.abs(np.arctan2(v, u))
    return (r <= 1.2) & (ang >= 0.6)


def _heart(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Classic implicit heart ``(u²+v²−1)³ − u²v³ ≤ 0`` (cusp toward −u)."""
    return (u ** 2 + v ** 2 - 1.0) ** 3 - u ** 2 * v ** 3 <= 0.0


class Shape(Enum):
    ARROW = "arrow"
    TEARDROP = "teardrop"
    PACMAN = "pacman"
    HEART = "heart"


_SHAPE_FNS: Dict[Shape, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    Shape.ARROW: _arrow,
    Shape.TEARDROP: _teardrop,
    Shape.PACMAN: _pacman,
    Shape.HEART: _heart,
}


def _shape_extent(fn, n: int = 400) -> float:
    """Max radius (normalized units) of a unit shape from its origin."""
    g = np.linspace(-1.8, 1.8, n)
    uu, vv = np.meshgrid(g, g)
    inside = fn(uu, vv)
    return float(np.hypot(uu[inside], vv[inside]).max())


# Bounding radius of each unit shape; scales linearly with sprite_scale.
SHAPE_EXTENT: Dict[Shape, float] = {s: _shape_extent(fn)
                                    for s, fn in _SHAPE_FNS.items()}


# ─────────────────────────────────────────────────────────────────
#  World configuration
# ─────────────────────────────────────────────────────────────────

@dataclass
class SpriteWorldConfig:
    """Full environment specification for the single-sprite world.

    Attributes
    ----------
    control:
        Which DOFs are exposed in the action vector (see ControlConfig).
    shape:
        Sprite silhouette (see Shape).
    image_size:
        Canvas side length in pixels (square canvas).
    sprite_scale:
        Pixels per normalized sprite unit.  Bounding radius is
        ``SHAPE_EXTENT[shape] * sprite_scale`` pixels.
    supersample:
        Anti-aliasing factor; the mask is sampled at ``supersample``× linear
        resolution and box-averaged down.  1 disables anti-aliasing.
    color_index:
        Palette index used to fill the sprite.
    max_delta_xy:
        Max |dx| and |dy| per step in pixels.
    max_delta_theta:
        Max |dθ| per step in radians.  If None (default), set to the
        *matched rim speed* ``max_delta_xy / sprite_scale`` so a max rotation
        moves the sprite's rim by about the same pixel distance as a max
        translation.
    palette:
        RGB triples; color_index indexes into this list.
    """
    control: ControlConfig = ControlConfig.XYT
    shape: Shape = Shape.ARROW
    image_size: int = 64
    sprite_scale: float = 7.5
    supersample: int = 4
    color_index: int = 0
    max_delta_xy: float = 8.0
    max_delta_theta: Optional[float] = None
    palette: List[Tuple[int, int, int]] = field(
        default_factory=lambda: list(DEFAULT_PALETTE)
    )

    # ---- derived properties ----------------------------------------

    @property
    def control_mask(self) -> np.ndarray:
        return self.control.mask

    @property
    def action_dim(self) -> int:
        return self.control.action_dim

    @property
    def max_delta_theta_eff(self) -> float:
        """Effective max |dθ|; matched rim speed when not set explicitly."""
        if self.max_delta_theta is not None:
            return float(self.max_delta_theta)
        return self.max_delta_xy / self.sprite_scale

    @property
    def delta_scale(self) -> np.ndarray:
        """Per-DOF physical range used to scale the matched Uniform(-1, 1)."""
        return np.array(
            [self.max_delta_xy, self.max_delta_xy, self.max_delta_theta_eff],
            dtype=np.float64,
        )

    @property
    def action_scale(self) -> np.ndarray:
        """Per-component normalizer for the action vector (controlled DOFs)."""
        return self.delta_scale[self.control_mask]

    @property
    def bounding_radius(self) -> int:
        """Sprite bounding radius in pixels (rotation-invariant)."""
        return int(np.ceil(SHAPE_EXTENT[self.shape] * self.sprite_scale))

    @property
    def margin(self) -> int:
        """Safe border so any translation keeps the sprite fully in canvas.

        The trailing +1 guarantees ≥1px clearance so the anti-aliased edge
        never bleeds into the border even in the worst-case displacement.
        """
        return self.bounding_radius + int(np.ceil(self.max_delta_xy)) + 1

    @property
    def max_displacement(self) -> float:
        """Interface mirror of StructuredDotWorldConfig.max_displacement.

        Returns the pixel range only; θ is on a different scale — prefer
        ``action_scale`` for per-DOF normalization.
        """
        return self.max_delta_xy

    # ---- human-readable summary ------------------------------------

    def describe(self) -> str:
        labels = self.control.controlled_labels or ["—"]
        return (
            f"SpriteWorldConfig  |  {self.image_size}×{self.image_size} px  "
            f"|  shape={self.shape.name}  sprite_scale={self.sprite_scale}  "
            f"|  control={self.control.name} ({','.join(labels)})  "
            f"|  action_dim={self.action_dim}\n"
            f"  max_delta: xy={self.max_delta_xy} px  "
            f"θ={self.max_delta_theta_eff:.3f} rad  "
            f"|  bounding_radius={self.bounding_radius} px  margin={self.margin} px"
        )


# ─────────────────────────────────────────────────────────────────
#  Convenience constructor
# ─────────────────────────────────────────────────────────────────

def make_sprite_config(
    control: ControlConfig = ControlConfig.XYT,
    shape: Shape = Shape.ARROW,
    **kwargs,
) -> SpriteWorldConfig:
    """Build a SpriteWorldConfig for one of the nested control configs."""
    return SpriteWorldConfig(control=control, shape=shape, **kwargs)


# ─────────────────────────────────────────────────────────────────
#  Rendering
# ─────────────────────────────────────────────────────────────────

def render_sprite(state: np.ndarray, config: SpriteWorldConfig) -> np.ndarray:
    """Render the sprite at pose ``state = (x, y, θ)`` onto a white canvas.

    θ rotates the sprite: the shape is evaluated in the sprite's canonical
    frame by rotating each (super-sampled) pixel offset by −θ about (x, y),
    then box-averaged down for anti-aliasing.

    Returns
    -------
    (3, H, W) float32 in [0, 1], channels-first (PyTorch convention).
    """
    H = W = config.image_size
    ss = max(1, int(config.supersample))
    x, y, theta = float(state[0]), float(state[1]), float(state[2])

    # Super-sampled pixel-center coordinates (sub-pixel grid centered on px).
    lin = (np.arange(H * ss) + 0.5) / ss - 0.5
    xx, yy = np.meshgrid(lin, lin)
    dx = xx - x
    dy = yy - y
    c, s = np.cos(theta), np.sin(theta)
    # Rotate pixel offset by −θ into the sprite frame, then flip vertical so
    # 'up' in sprite space is up on screen (image y grows downward).
    u = (c * dx + s * dy) / config.sprite_scale
    v = -(-s * dx + c * dy) / config.sprite_scale

    inside = _SHAPE_FNS[config.shape](u, v)
    alpha = inside.reshape(H, ss, W, ss).mean(axis=(1, 3))   # (H, W) in [0, 1]

    rgb = np.array(
        config.palette[config.color_index % len(config.palette)],
        dtype=np.float32,
    ) / 255.0
    canvas = np.empty((3, H, W), dtype=np.float32)
    for ch in range(3):
        canvas[ch] = 1.0 - alpha * (1.0 - rgb[ch])           # blend over white
    return canvas


# ─────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────

class SpriteWorldDataset(Dataset):
    """PyTorch dataset generating (obs_t, action, obs_tp1, state_t) transitions.

    Samples are generated on-the-fly from a seeded RNG; the dataset is
    deterministic and reproducible given (seed, index).

    Returns
    -------
    obs_t   : (3, H, W) float32
    action  : (action_dim,) float32 — deltas of the controlled DOFs only
    obs_tp1 : (3, H, W) float32
    state_t : (3,) float32          — pose (x, y, θ) at t
    """

    def __init__(
        self,
        config: Optional[SpriteWorldConfig] = None,
        num_samples: int = 10_000,
        seed: int = 0,
    ):
        self.config = config or SpriteWorldConfig()
        self.num_samples = num_samples
        self.seed = seed

    # ── helpers ──────────────────────────────────────────────────

    def _sample_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        """Sample pose (x, y, θ); x, y leave a full margin from every edge."""
        cfg = self.config
        lo, hi = cfg.margin, cfg.image_size - 1 - cfg.margin
        if lo >= hi:
            raise RuntimeError(
                f"Margin ({cfg.margin}) is too large for canvas size "
                f"({cfg.image_size}).  Reduce sprite_scale or max_delta_xy."
            )
        x = rng.uniform(lo, hi)
        y = rng.uniform(lo, hi)
        theta = rng.uniform(0.0, 2.0 * np.pi)
        return np.array([x, y, theta], dtype=np.float64)

    def _sample_delta(self, rng: np.random.Generator) -> np.ndarray:
        """Sample (dx, dy, dθ) for all three DOFs from the matched distribution.

        Every DOF moves regardless of the control mask; the mask only selects
        which deltas are later exposed in the action vector.
        """
        unit = rng.uniform(-1.0, 1.0, size=3)        # matched across DOFs
        return unit * self.config.delta_scale         # → per-DOF physical units

    def _sample_transition(
        self, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config
        state_t = self._sample_initial_state(rng)
        delta = self._sample_delta(rng)
        state_tp1 = state_t + delta
        state_tp1[2] = state_tp1[2] % (2.0 * np.pi)   # wrap θ; x, y stay in margin
        action = delta[cfg.control_mask].astype(np.float32)
        return state_t, action, state_tp1

    # ── Dataset interface ────────────────────────────────────────

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed + idx)
        state_t, action, state_tp1 = self._sample_transition(rng)

        obs_t = render_sprite(state_t, self.config)
        obs_tp1 = render_sprite(state_tp1, self.config)

        return (
            torch.from_numpy(obs_t),
            torch.from_numpy(action),
            torch.from_numpy(obs_tp1),
            torch.from_numpy(state_t.astype(np.float32)),
        )


# ─────────────────────────────────────────────────────────────────
#  Trajectory generation
# ─────────────────────────────────────────────────────────────────

def generate_trajectory(
    config: SpriteWorldConfig,
    num_steps: int = 10,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll out a continuous multi-step trajectory under the given config.

    Returns
    -------
    observations : (T+1, 3, H, W) float32
    actions      : (T, action_dim) float32 — controlled DOFs per step
    states       : (T+1, 3) float32        — pose (x, y, θ) per step
    """
    ds = SpriteWorldDataset(config=config, num_samples=0, seed=0)
    rng = np.random.default_rng(seed)
    lo, hi = config.margin, config.image_size - 1 - config.margin

    state = np.array(
        [rng.uniform(lo, hi), rng.uniform(lo, hi), rng.uniform(0.0, 2.0 * np.pi)],
        dtype=np.float64,
    )

    observations = [render_sprite(state, config)]
    actions: List[np.ndarray] = []
    states = [state.astype(np.float32).copy()]

    for _ in range(num_steps):
        delta = ds._sample_delta(rng)
        state = state + delta
        state[0] = np.clip(state[0], lo, hi)          # clamp drift to canvas
        state[1] = np.clip(state[1], lo, hi)
        state[2] = state[2] % (2.0 * np.pi)           # wrap θ
        actions.append(delta[config.control_mask].astype(np.float32))
        states.append(state.astype(np.float32).copy())
        observations.append(render_sprite(state, config))

    return (
        np.stack(observations),
        np.stack(actions) if actions
        else np.empty((0, config.action_dim), dtype=np.float32),
        np.stack(states),
    )
