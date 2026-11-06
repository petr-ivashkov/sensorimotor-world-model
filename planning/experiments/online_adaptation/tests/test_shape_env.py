"""Geometry/physics checks without resetting the incompatible local simulator."""
import math
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pymunk

from planning.experiments.online_adaptation.shape_env import (
    PushU, SHAPE_ENV_ENTRY_POINT, SHAPE_ENV_ID,
    make_u_body, register_shape_environments, u_polygons,
)


def area_and_centroid(vertices):
    vertices = np.asarray(vertices)
    following = np.roll(vertices, -1, axis=0)
    cross = vertices[:, 0] * following[:, 1] - following[:, 0] * vertices[:, 1]
    area = cross.sum() / 2
    centroid = ((vertices + following) * cross[:, None]).sum(axis=0) / (6 * area)
    return area, centroid


class ShapeEnvironmentTests(unittest.TestCase):
    def test_area_bounds_com_and_disjoint_material(self):
        polygons = u_polygons()
        self.assertEqual(len(polygons), 3)
        all_vertices = np.concatenate(polygons)
        np.testing.assert_array_equal(all_vertices.min(0), [-60, -47.5])
        np.testing.assert_array_equal(all_vertices.max(0), [60, 72.5])
        components = [area_and_centroid(vertices) for vertices in polygons]
        self.assertEqual([area for area, _ in components], [2400, 2000, 2000])
        np.testing.assert_allclose(sum(area * center for area, center in components) / 6400,
                                   [0, 0], atol=1e-12)
        # Rectangles can share edges, but their interior intersection has no area.
        for index, vertices in enumerate(polygons):
            lower, upper = np.min(vertices, axis=0), np.max(vertices, axis=0)
            for other in polygons[index + 1:]:
                overlap = np.minimum(upper, np.max(other, axis=0)) - np.maximum(lower, np.min(other, axis=0))
                self.assertLessEqual(float(np.prod(np.maximum(overlap, 0))), 0)

    def test_scale_changes_dimensions_area_and_inertia_consistently(self):
        full = np.asarray(u_polygons())
        half = np.asarray(u_polygons(20))
        np.testing.assert_array_equal(half, full / 2)
        body = make_u_body(pymunk.Space(), (0, 0), scale=40)
        smaller = make_u_body(pymunk.Space(), (0, 0), scale=20)
        self.assertAlmostEqual(smaller.moment, body.moment / 4)
        for invalid in (0, -1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                u_polygons(invalid)

    def test_factory_creates_dynamic_mass_one_with_compound_inertia(self):
        space = pymunk.Space()
        body = make_u_body(space, (200, 200))
        self.assertEqual(body.body_type, pymunk.Body.DYNAMIC)
        self.assertEqual(body.mass, 1)
        self.assertEqual(len(body.shapes), 3)
        np.testing.assert_allclose(body.center_of_gravity, [0, 0], atol=1e-12)
        # Uniform-density rectangle moments plus parallel-axis shifts.
        expected = (0.375 * ((120**2 + 20**2)/12 + 37.5**2)
                    + 2 * 0.3125 * ((20**2 + 100**2)/12 + 50**2 + 22.5**2))
        self.assertAlmostEqual(body.moment, expected)
        self.assertAlmostEqual(body.moment, 3410.4166666666665)
        for shape in body.shapes:
            self.assertEqual(shape.friction, 0)
            self.assertEqual(shape.elasticity, 0)

    def test_cavity_is_empty_and_pusher_fits_through_opening(self):
        space = pymunk.Space()
        body = make_u_body(space, (0, 0))
        for point in ((0, 0), (0, 30), (0, 60)):
            self.assertIsNone(space.point_query_nearest(point, 0, pymunk.ShapeFilter()))
        for point in ((-50, 20), (50, 20), (0, -37.5)):
            self.assertIsNotNone(space.point_query_nearest(point, 0, pymunk.ShapeFilter()))
        # A radius-15 pusher can travel down the cavity centerline, including
        # through the open top, without intersecting any wall.
        for y in (-10, 0, 30, 60, 90):
            clearance = min(shape.point_query((0, y)).distance for shape in body.shapes)
            self.assertGreaterEqual(clearance, 15)
        polygons = u_polygons()
        self.assertEqual(polygons[2][0][0] - polygons[1][1][0], 80)
        self.assertEqual(polygons[1][2][1] - polygons[1][0][1], 100)

    def test_pusher_enters_cavity_then_moves_dynamic_block_by_inner_base(self):
        space = pymunk.Space()
        space.gravity = (0, 0)
        block = make_u_body(space, (0, 0))
        pusher = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        pusher.position = (0, 100)
        pusher.velocity = (0, -40)
        circle = pymunk.Circle(pusher, 15)
        space.add(pusher, circle)

        # Enter from above and travel between the arms to y=10. There is still
        # 22.5 units between the circle's bottom and the inner base, so neither
        # arm nor base should move the U prematurely.
        for _ in range(450):
            space.step(0.005)
        self.assertAlmostEqual(pusher.position.y, 10, places=8)
        np.testing.assert_allclose(block.position, [0, 0], atol=1e-9)
        self.assertAlmostEqual(block.angle, 0, places=9)

        # Continue down into the base. A kinematic pusher transfers normal
        # contact forces to this dynamic compound body, even at zero friction.
        for _ in range(400):
            space.step(0.005)
        self.assertLess(block.position.y, -10)
        self.assertLess(block.velocity.y, 0)
        self.assertAlmostEqual(block.position.x, 0, places=6)
        self.assertTrue(np.isfinite([*block.position, *block.velocity, block.angle,
                                     block.angular_velocity]).all())

    def test_orientation_is_full_turn_and_factory_sets_pose(self):
        angle, position = 0.73, np.array([210, 190])
        body = make_u_body(pymunk.Space(), position, angle=angle)
        self.assertAlmostEqual(body.angle, angle)
        np.testing.assert_allclose(body.position, position)
        point = (0, -37.5)
        rotation = np.array([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]])
        np.testing.assert_allclose(body.local_to_world(point), position + rotation @ point)
        # The bottom-bar center becomes empty cavity under pi and pi/2 turns;
        # a full turn gives the same occupied point. Existing 2pi angle metric
        # therefore has no hidden 90/180-degree symmetry for this U.
        space = pymunk.Space()
        original = make_u_body(space, (0, 0))
        self.assertIsNotNone(space.point_query_nearest(point, 0, pymunk.ShapeFilter()))
        for rotated in ((0, 37.5), (37.5, 0)):
            self.assertIsNone(space.point_query_nearest(rotated, 0, pymunk.ShapeFilter()))
        np.testing.assert_allclose(original.local_to_world(point), point)

    def test_custom_constructor_preserves_state_action_and_default_variations(self):
        env = PushU()
        self.assertEqual(env.shapes[2], "U")
        self.assertEqual(env.env_name, "PushT-U")
        self.assertEqual(env.action_space.shape, (2,))
        self.assertEqual(env.observation_space["state"].shape, (7,))
        self.assertEqual(env.observation_space["proprio"].shape, (4,))
        self.assertEqual(env.variation_space["block"]["shape"].value, 2)
        self.assertEqual(env.variation_space["block"]["shape"].n, 1)
        self.assertEqual(env.variation_space["block"]["shape"].start, 2)
        self.assertEqual(env.variation_space["block"]["scale"].value, 40)
        self.assertIsNone(env.space)  # No reset or simulator step was performed.
        env.space = pymunk.Space()
        body = env.add_shape("U", position=(0, 0), angle=0.2, scale=40)
        self.assertEqual(len(body.shapes), 3)
        self.assertEqual(body.body_type, pymunk.Body.DYNAMIC)
        # Native dispatch remains available for non-U shapes.
        square = env.add_shape("square", position=(200, 200), angle=0.4, scale=20)
        self.assertEqual(len(square.shapes), 1)
        self.assertEqual(square.body_type, pymunk.Body.DYNAMIC)

    def test_shape_resampling_cannot_change_registered_u_geometry(self):
        from stable_worldmodel.spaces import reset_variation_space

        env = PushU()
        for seed in range(8):
            reset_variation_space(env.variation_space, seed,
                                  {"variation": ["block.shape"]})
            self.assertEqual(env.variation_space["block"]["shape"].value, 2)
            self.assertEqual(env.shapes[2], "U")
        with self.assertRaises((ValueError, AssertionError)):
            reset_variation_space(env.variation_space, 0,
                                  {"variation": ["block.shape"],
                                   "variation_values": {"block.shape": 1}})
        self.assertIsNone(env.space)

    def test_registration_is_idempotent_and_never_overwrites_another_env(self):
        with patch.dict(gym.envs.registry):
            gym.envs.registry.pop(SHAPE_ENV_ID, None)
            original = gym.spec("swm/PushT-v1")
            self.assertEqual(register_shape_environments(), SHAPE_ENV_ID)
            registered = gym.spec(SHAPE_ENV_ID)
            self.assertEqual(registered.entry_point, SHAPE_ENV_ENTRY_POINT)
            self.assertEqual(register_shape_environments(), SHAPE_ENV_ID)
            self.assertIs(gym.spec(SHAPE_ENV_ID), registered)
            self.assertIs(gym.spec("swm/PushT-v1"), original)
            gym.envs.registry.pop(SHAPE_ENV_ID)
            gym.register(id=SHAPE_ENV_ID, entry_point="other_module:OtherEnv")
            foreign = gym.spec(SHAPE_ENV_ID)
            with self.assertRaisesRegex(ValueError, "another implementation"):
                register_shape_environments()
            self.assertIs(gym.spec(SHAPE_ENV_ID), foreign)


if __name__ == "__main__":
    unittest.main()
