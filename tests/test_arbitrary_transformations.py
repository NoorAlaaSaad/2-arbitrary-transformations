"""Check the geometry and visibility rules with small, known examples."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from arbitrary_transformations import (
    camera_intrinsics,
    load_lidar_points,
    project_points,
    render_projection,
    rigid_transform,
    rotation_matrix,
    transform_points,
)


class TransformationTests(unittest.TestCase):
    def test_lidar_basis_maps_to_camera_axes(self):
        expected_rotation = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        rotation = rotation_matrix(90, -90, 0)
        assert_allclose(rotation, expected_rotation, atol=1e-14)
        assert_allclose(
            transform_points(np.eye(3), rigid_transform(90, -90, 0)),
            [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
            atol=1e-14,
        )

    def test_rotation_preserves_distances_and_orientation(self):
        rotation = rotation_matrix(23, -17, 49)
        assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0)
        points = np.array([[1, 2, 3], [-2, 1, 5]], dtype=float)
        transformed = transform_points(
            points, rigid_transform(23, -17, 49, translation=(4, -3, 2))
        )
        self.assertAlmostEqual(
            np.linalg.norm(transformed[0] - transformed[1]),
            np.linalg.norm(points[0] - points[1]),
        )

    def test_composition_applies_rightmost_transform_first(self):
        first = rigid_transform(yaw_degrees=90, translation=(1, 2, 3))
        second = rigid_transform(pitch_degrees=90, translation=(-2, 0, 1))
        points = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        assert_allclose(
            transform_points(points, second @ first),
            [[1, 3, 0], [1, 2, 1]],
            atol=1e-14,
        )
        assert_array_equal(first[3], [0, 0, 0, 1])

    def test_viewer_looks_along_requested_lidar_azimuth(self):
        lidar_to_camera = rigid_transform(90, -90, 0)
        for azimuth in (0, 45, 90):
            with self.subTest(azimuth=azimuth):
                angle = np.deg2rad(azimuth)
                look_direction = np.array([[np.cos(angle), np.sin(angle), 0]])
                viewer = rigid_transform(pitch_degrees=azimuth)
                assert_allclose(
                    transform_points(look_direction, viewer @ lidar_to_camera),
                    [[0, 0, 1]],
                    atol=1e-14,
                )


class ProjectionTests(unittest.TestCase):
    def test_intrinsics_support_centered_and_literal_principal_points(self):
        assert_array_equal(
            camera_intrinsics(), [[800, 0, 960], [0, 800, 360], [0, 0, 1]]
        )
        assert_array_equal(
            camera_intrinsics(principal_x=360, principal_y=960),
            [[800, 0, 360], [0, 800, 960], [0, 0, 1]],
        )

    def test_projection_uses_columns_then_rows_and_filters_invalid_points(self):
        intrinsics = camera_intrinsics(2, 4, 3, 2)
        points = np.array(
            [
                [2, 1, 2],       # (u, v) = (5, 4)
                [-1, -0.5, 2],   # (u, v) = (2, 1)
                [0, 0, -1],      # Behind the camera.
                [0, 0, 0],
                [0, 0, 0.1],     # On the excluded near plane.
                [np.nan, 0, 1],
                [0, np.inf, 1],
                [0, 0, np.inf],
                [-3.1, 0, 2],    # Slightly outside the left image boundary.
                [5, 0, 2],       # On the excluded right boundary.
                [0, -2, 2],      # Above the image.
                [0, 2, 2],       # On the excluded bottom boundary.
            ]
        )
        pixels, depths = project_points(points, intrinsics, width=8, height=6)
        assert_array_equal(pixels, [[5, 4], [2, 1]])
        assert_array_equal(depths, [2, 2])
        self.assertTrue(np.issubdtype(pixels.dtype, np.integer))

    def test_empty_and_fully_hidden_clouds(self):
        intrinsics = camera_intrinsics(2, 2, 2, 2)
        clouds = [np.empty((0, 3)), np.array([[0, 0, -1], [np.nan, 0, 1]])]
        for points in clouds:
            with self.subTest(point_count=len(points)):
                pixels, depths = project_points(points, intrinsics, 5, 5)
                self.assertEqual(pixels.shape, (0, 2))
                self.assertEqual(depths.shape, (0,))
                image = render_projection(points, intrinsics, width=5, height=5)
                self.assertEqual(image.shape, (5, 5, 3))
                self.assertEqual(image.dtype, np.uint8)
                self.assertFalse(image.any())

    def test_nearest_depth_wins_regardless_of_input_order(self):
        intrinsics = camera_intrinsics(2, 2, 2, 2)
        near = np.array([[0, 0, 2]], dtype=float)
        far = np.array([[0, 0, 8]], dtype=float)
        expected = render_projection(near, intrinsics, width=5, height=5)
        far_image = render_projection(far, intrinsics, width=5, height=5)
        self.assertTrue(expected[2, 2].any())
        self.assertFalse(np.array_equal(expected[2, 2], far_image[2, 2]))
        self.assertEqual(np.count_nonzero(np.any(expected != 0, axis=2)), 1)
        for points in (np.vstack([near, far]), np.vstack([far, near])):
            assert_array_equal(
                render_projection(points, intrinsics, width=5, height=5), expected
            )


class LidarLoadingTests(unittest.TestCase):
    def test_loads_xyz_without_reflectance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.bin"
            np.array([[1, -2, 3, 0.25], [4, 5, 6, 0.75]], dtype="<f4").tofile(path)
            points = load_lidar_points(path)
        assert_array_equal(points, [[1, -2, 3], [4, 5, 6]])
        self.assertEqual(points.dtype, np.float64)

    def test_rejects_empty_truncated_and_nonfinite_data(self):
        invalid_files = {
            "empty": b"",
            "incomplete_float": b"\x00",
            "incomplete_point": np.array([1, 2, 3], dtype="<f4").tobytes(),
            "nonfinite_xyz": np.array([1, np.nan, 3, 0.5], dtype="<f4").tobytes(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.bin"
            for name, contents in invalid_files.items():
                with self.subTest(case=name):
                    path.write_bytes(contents)
                    with self.assertRaises(ValueError):
                        load_lidar_points(path)


if __name__ == "__main__":
    unittest.main()
