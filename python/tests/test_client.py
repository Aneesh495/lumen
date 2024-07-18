"""
Unit and integration test suite for Lumen Python SDK.
"""

import os
import unittest
import tempfile
import math

from lumen import (
    Camera,
    CameraTrajectory,
    GaussianCloud,
    Gaussian3D,
    LumenEngine,
    RasterizerConfig,
    export_ppm,
    export_depth_visualization,
    ascii_render,
)


class TestLumenCamera(unittest.TestCase):
    def test_camera_forward(self):
        cam = Camera(position=(0.0, 0.0, 5.0), target=(0.0, 0.0, 0.0))
        f = cam.forward_vector()
        self.assertAlmostEqual(f[0], 0.0, places=5)
        self.assertAlmostEqual(f[1], 0.0, places=5)
        self.assertAlmostEqual(f[2], -1.0, places=5)

    def test_view_matrix_identity(self):
        cam = Camera(position=(0.0, 0.0, 0.0), target=(0.0, 0.0, -1.0))
        mat = cam.view_matrix_4x4()
        self.assertEqual(len(mat), 4)
        self.assertEqual(len(mat[0]), 4)
        self.assertAlmostEqual(mat[3][3], 1.0, places=5)

    def test_orbit_trajectory(self):
        cams = CameraTrajectory.orbit(target=(0.0, 0.0, 0.0), radius=4.0, num_frames=12)
        self.assertEqual(len(cams), 12)
        for c in cams:
            dx = c.position[0]
            dz = c.position[2]
            dist_xz = math.sqrt(dx * dx + dz * dz)
            self.assertGreater(dist_xz, 0.5)

    def test_camera_interpolation(self):
        c1 = Camera(position=(0.0, 0.0, 2.0))
        c2 = Camera(position=(0.0, 0.0, 4.0))
        mid = CameraTrajectory.interpolate(c1, c2, 0.5)
        self.assertAlmostEqual(mid.position[2], 3.0, places=5)


class TestLumenCloud(unittest.TestCase):
    def test_cube_generation(self):
        cube = GaussianCloud.synthetic_cube(points_per_axis=3, cube_size=2.0)
        self.assertEqual(len(cube), 27)
        bbox = cube.bounding_box()
        self.assertAlmostEqual(bbox[0][0], -1.0, places=5)
        self.assertAlmostEqual(bbox[1][0], 1.0, places=5)
        cen = cube.centroid()
        self.assertAlmostEqual(cen[0], 0.0, places=5)
        self.assertAlmostEqual(cen[1], 0.0, places=5)
        self.assertAlmostEqual(cen[2], 0.0, places=5)

    def test_sphere_generation(self):
        sphere = GaussianCloud.synthetic_sphere(point_count=50, radius=2.0)
        self.assertEqual(len(sphere), 50)

    def test_ply_roundtrip(self):
        cloud = GaussianCloud.synthetic_cube(points_per_axis=2, cube_size=1.0)
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cloud.save_ply(tmp_path)
            loaded = GaussianCloud.load_ply(tmp_path)
            self.assertEqual(len(loaded), len(cloud))
            self.assertAlmostEqual(loaded.splats[0].position[0], cloud.splats[0].position[0], places=4)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestLumenEngine(unittest.TestCase):
    def test_rasterize_scene(self):
        cfg = RasterizerConfig(
            image_width=80,
            image_height=60,
            tile_size=16,
            near_plane=0.1,
            far_plane=50.0,
        )
        engine = LumenEngine(cfg)
        cube = GaussianCloud.synthetic_cube(points_per_axis=3, cube_size=1.0)
        cam = Camera(position=(0.0, 0.0, 2.5), target=(0.0, 0.0, 0.0))

        res = engine.render(cube, cam)
        self.assertEqual(res.width, 80)
        self.assertEqual(res.height, 60)
        self.assertEqual(len(res.rgb_bytes), 80 * 60 * 3)
        self.assertEqual(len(res.depth_values), 80 * 60)
        self.assertEqual(len(res.alpha_values), 80 * 60)

        # Check stats
        self.assertEqual(res.stats["total_splats"], 27)
        self.assertGreater(res.stats["visible_splats"], 0)
        self.assertGreater(res.stats["touched_tiles"], 0)

        # Center pixel should have received alpha
        mid_idx = (60 // 2) * 80 + (80 // 2)
        self.assertGreater(res.alpha_values[mid_idx], 0.0)

        # Test ascii render
        ascii_art = ascii_render(res.width, res.height, res.alpha_values, 30, 15)
        self.assertIsInstance(ascii_art, str)
        self.assertGreater(len(ascii_art), 0)

    def test_image_export(self):
        cfg = RasterizerConfig(image_width=40, image_height=30)
        engine = LumenEngine(cfg)
        cube = GaussianCloud.synthetic_cube(points_per_axis=2, cube_size=0.8)
        cam = Camera(position=(0.0, 0.0, 2.0))
        res = engine.render(cube, cam)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ppm_path = os.path.join(tmp_dir, "render.ppm")
            depth_path = os.path.join(tmp_dir, "depth.ppm")

            export_ppm(ppm_path, res.width, res.height, res.rgb_bytes)
            self.assertTrue(os.path.exists(ppm_path))
            self.assertGreater(os.path.getsize(ppm_path), 0)

            export_depth_visualization(depth_path, res.width, res.height, res.depth_values)
            self.assertTrue(os.path.exists(depth_path))
            self.assertGreater(os.path.getsize(depth_path), 0)


if __name__ == "__main__":
    unittest.main()
