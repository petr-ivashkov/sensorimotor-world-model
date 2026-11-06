"""A single dynamic U-shaped block using the existing Push-T interface.

The new body has a 120 x 120 outer box at the default scale, 20-unit walls,
an 80-unit opening, and a 100-unit-deep cavity. Its area is 6,400 square
units (the default T has 6,300), total mass is one, and all polygons are
centered jointly at their area-weighted center of mass. The default circular
pusher has radius 15, so it can enter the opening.

This module does not replace or modify the installed Push-T environment.
"""
from __future__ import annotations

import math

from stable_worldmodel.envs.pusht.env import PushT


SHAPE_ENV_ID = "swm/PushT-U-v0"
SHAPE_ENV_ENTRY_POINT = "planning.experiments.online_adaptation.shape_env:PushU"


def u_polygons(scale: float = 40.0) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Return three nonoverlapping CCW rectangles, centered at the body's COM.

    ``scale=40`` matches the inherited block-scale default. The polygons share
    boundary edges but have disjoint interiors, avoiding overlapping material.
    The centered outer bounds are x=[-60,60], y=[-47.5,72.5] at that scale.
    """
    factor = float(scale) / 40.0
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("U block scale must be positive and finite")
    # Coordinates before centering: base, left arm, right arm. Areas are 2,400,
    # 2,000, and 2,000. Their combined centroid is (0,-12.5).
    rectangles = (
        (-60.0, 60.0, -60.0, -40.0),
        (-60.0, -40.0, -40.0, 60.0),
        (40.0, 60.0, -40.0, 60.0),
    )
    total_area = sum((x1-x0) * (y1-y0) for x0, x1, y0, y1 in rectangles)
    center_x = sum((x1-x0) * (y1-y0) * (x0+x1)/2 for x0, x1, y0, y1 in rectangles) / total_area
    center_y = sum((x1-x0) * (y1-y0) * (y0+y1)/2 for x0, x1, y0, y1 in rectangles) / total_area
    return tuple(
        tuple(((x-center_x) * factor, (y-center_y) * factor)
              for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
        for x0, x1, y0, y1 in rectangles
    )


def _polygon_area(vertices) -> float:
    return abs(sum(x0*y1 - x1*y0 for (x0, y0), (x1, y1)
                   in zip(vertices, vertices[1:] + vertices[:1]))) / 2


def make_u_body(space, position, angle: float = 0.0, scale: float = 40.0,
                color="LightSlateGray", mask=None):
    """Create a dynamic mass-one U with uniform density and compound inertia.

    Shape friction/elasticity retain the upstream constructors' defaults. Their
    legacy ``body.friction`` attribute is preserved, too; material behavior is
    not changed as part of the geometry intervention.
    """
    import pygame
    import pymunk

    polygons = u_polygons(scale)
    areas = [_polygon_area(vertices) for vertices in polygons]
    total_area = sum(areas)
    # Each moment is about the common COM because all vertices were centered
    # jointly. This includes each rectangle's parallel-axis contribution.
    moment = sum(pymunk.moment_for_poly(area / total_area, vertices)
                 for area, vertices in zip(areas, polygons))
    body = pymunk.Body(1.0, moment, body_type=pymunk.Body.DYNAMIC)
    body.center_of_gravity = (0.0, 0.0)
    shapes = [pymunk.Poly(body, vertices) for vertices in polygons]
    for shape in shapes:
        shape.color = pygame.Color(color)
        shape.filter = pymunk.ShapeFilter(mask=(pymunk.ShapeFilter.ALL_MASKS()
                                              if mask is None else mask))
    body.position = tuple(float(coordinate) for coordinate in position)
    body.angle = float(angle)
    body.friction = 1
    space.add(body, *shapes)
    return body


class PushU(PushT):
    """Push-T state/control contract with U geometry at the default shape index.

    The distinct environment ID and ``env_name`` identify geometry provenance.
    Index 2 stays within the inherited variation schema; in this environment it
    denotes U rather than T. Existing data/goal rendering follows actual body
    polygons, so it also renders U-shaped target poses.
    """

    geometry_name = "U"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from stable_worldmodel.spaces import Discrete

        self.shapes[2] = "U"
        # Sampling block.shape in this registered environment must stay U.
        # Otherwise random variation could silently turn a U run back into a
        # native shape despite its fixed environment provenance.
        self.variation_space["block"].spaces["shape"] = Discrete(
            1, start=2, init_value=2,
        )
        self.env_name = "PushT-U"

    def add_shape(self, shape, *args, **kwargs):
        if shape == "U":
            return make_u_body(self.space, *args, **kwargs)
        return super().add_shape(shape, *args, **kwargs)


def register_shape_environments() -> str:
    """Register the custom ID once; refuse to overwrite another implementation."""
    import gymnasium as gym

    existing = gym.envs.registry.get(SHAPE_ENV_ID)
    if existing is None:
        gym.register(id=SHAPE_ENV_ID, entry_point=SHAPE_ENV_ENTRY_POINT)
    elif existing.entry_point != SHAPE_ENV_ENTRY_POINT:
        raise ValueError(f"Environment ID {SHAPE_ENV_ID!r} is already registered to another implementation")
    return SHAPE_ENV_ID
