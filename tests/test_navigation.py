"""Run with python -m unittest discover -s tests -v."""
import ast
import math
from pathlib import Path
import unittest

import cv2
import mujoco
import numpy as np

from camera_rig import CameraFrame, CameraRig, WheelOdometry, load_config
from vision_navigation import FloorVision, ObservedMap, VisionNavigator, ground_points


class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_encoder_odometry_and_reset(self):
        odom = WheelOdometry(0.1, 0.4)
        odom.update([20, 20])  # baseline is not a displacement
        np.testing.assert_allclose(odom.update([30, 30]), [1, 0, 0])
        np.testing.assert_allclose(odom.update([28, 32]), [1, 0, 1])
        odom.reset()
        np.testing.assert_allclose(odom.update([28, 32]), [0, 0, 0])

    def test_projection_uses_calibration(self):
        # Downward-looking camera, image centre hits floor below the camera.
        f = CameraFrame("head_cam", np.zeros((10, 10, 3), np.uint8),
                        np.array([[10, 0, 5], [0, 10, 5], [0, 0, 1]]),
                        np.array([1, 2, 1]), np.eye(3), 0)
        points, valid = ground_points(f, [[5, 5], [10, 5]])
        self.assertTrue(valid.all())
        np.testing.assert_allclose(points, [[1, 2], [1.5, 2]])
        f.rotation = np.diag([1, -1, -1])
        self.assertFalse(ground_points(f, [[5, 5]])[1][0])

    def test_planner_detours_and_does_not_cross_unknown(self):
        grid = ObservedMap(self.config)
        free = np.array([(x, y) for x in np.arange(-1, 3, .05)
                         for y in np.arange(-1.5, 1.5, .05)])
        wall = np.array([(1, y) for y in np.arange(-.5, .55, .05)])
        grid.integrate(free, wall, {}, np.zeros(3))
        path = grid.route([0, 0], [2, 0])
        self.assertTrue(path)
        np.testing.assert_allclose(path[-1], [2.05, .05])
        self.assertGreater(max(abs(p[1]) for p in path), .7)
        self.assertTrue(all(grid.safe_segment(a, b) for a, b in zip(path, path[1:])))
        # A destination is not permission to traverse unseen cells.
        path = ObservedMap(self.config).route([0, 0], [8, 8])
        self.assertEqual(path, [])

    def test_new_obstacle_blocks_previously_clear_path(self):
        grid = ObservedMap(self.config)
        free = np.array([(x, y) for x in np.arange(0, 2, .05) for y in np.arange(-1, 1, .05)])
        for _ in range(8):
            grid.integrate(free, np.empty((0, 2)), {}, np.zeros(3))
        self.assertTrue(grid.safe_segment([0, 0], [1, 0]))
        grid.integrate(np.empty((0, 2)), np.array([[.6, 0]]), {}, np.zeros(3))
        self.assertFalse(grid.safe_segment([0, 0], [1, 0]))

    def test_missing_stale_and_black_images_stop(self):
        nav = VisionNavigator(self.config, goal=[1, 0])
        self.assertEqual(nav.command(np.zeros(3), 0), (0, 0))
        nav.last_frame = 0
        self.assertEqual(nav.command(np.zeros(3), 1), (0, 0))
        frames = [CameraFrame(c["name"], np.zeros((240, 320, 3), np.uint8),
                              np.eye(3), np.array([0, 0, 1]), np.eye(3), 2)
                  for c in self.config["cameras"]]
        nav.observe(frames, np.zeros(3))
        self.assertEqual(nav.command(np.zeros(3), 2), (0, 0))
        self.assertEqual(nav.state, "CAMERA_LOST")

    def test_unknown_clearance_includes_robot_width(self):
        grid = ObservedMap(self.config)
        # A visible centre line alone is not enough clearance for the base.
        line = np.array([(x, 0) for x in np.arange(.4, 2, .05)])
        grid.integrate(line, np.empty((0, 2)), {}, np.zeros(3))
        self.assertFalse(grid.safe_segment([0, 0], [1, 0]))

    def test_stalled_navigation_reports_blocked(self):
        nav = VisionNavigator(self.config, goal=[1, 0])
        nav.last_frame = 0
        nav.command(np.zeros(3), 0)
        nav.last_frame = 46
        self.assertEqual(nav.command(np.zeros(3), 46), (0, 0))
        self.assertEqual(nav.state, "BLOCKED")

    def test_room_is_unknown_until_seen_in_rgb(self):
        nav = VisionNavigator(self.config, room="kitchen")
        self.assertIsNone(nav.goal)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        rgb = np.full((240, 320, 3), 190, np.uint8)
        rgb[72:168, 112:208] = cv2.aruco.generateImageMarker(dictionary, 1, 96)[..., None]
        f = CameraFrame("head_cam", rgb, np.array([[200, 0, 160], [0, 200, 120], [0, 0, 1]]),
                        np.array([1, 0, 1]), np.eye(3), 0)
        _, _, labels = FloorVision(self.config).observe([f])
        self.assertIn("kitchen", labels)
        np.testing.assert_allclose(labels["kitchen"], [1, 0], atol=.01)

    def test_navigator_has_no_simulator_or_layout_dependency(self):
        tree = ast.parse(Path("vision_navigation.py").read_text())
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        self.assertNotIn("layout", imports)
        self.assertNotIn("mujoco", imports)
        self.assertNotIn("layout", Path("run_world.py").read_text())


class CameraIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from build_world import build
        cls.model = mujoco.MjModel.from_xml_path(build())

    def setUp(self):
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.rig = CameraRig(self.model, load_config())
        self.addCleanup(self.rig.close)

    def test_three_rgb_cameras_and_no_rangefinders(self):
        self.assertFalse(np.any(self.model.sensor_type == mujoco.mjtSensor.mjSENS_RANGEFINDER))
        _, frames = self.rig.sample(self.data)
        self.assertEqual(len(frames), 3)
        self.assertTrue(all(f.rgb.shape == (240, 320, 3) for f in frames))
        self.assertNotIn("overview", [f.name for f in frames])

    def test_mounts_follow_hand_joints_and_ignore_world_pose(self):
        pose, before = self.rig.sample(self.data)
        self.data.qpos[:3] += [2, -1, 0]
        angle = math.pi / 4
        self.data.qpos[3:7] = [math.cos(angle/2), 0, 0, math.sin(angle/2)]
        mujoco.mj_forward(self.model, self.data)
        self.rig.reset()
        pose, moved = self.rig.sample(self.data)
        np.testing.assert_allclose(pose, [0, 0, 0])
        for a, b in zip(before, moved):
            np.testing.assert_allclose(a.origin, b.origin, atol=.005)
            np.testing.assert_allclose(a.rotation, b.rotation, atol=.005)
        self.data.qpos[self.model.jnt_qposadr[self.model.joint("lj6").id]] = .5
        mujoco.mj_forward(self.model, self.data)
        self.rig.reset()
        _, bent = self.rig.sample(self.data)
        np.testing.assert_allclose(moved[0].rotation, bent[0].rotation)
        np.testing.assert_allclose(moved[2].rotation, bent[2].rotation)
        self.assertGreater(np.linalg.norm(moved[1].rotation-bent[1].rotation), .2)

    def test_rendered_room_marker_is_detected(self):
        # Room labels now flank the doorway rather than sitting mid-room, so
        # face the door (the robot's spawn heading) to see one.
        self.data.qpos[3:7] = [math.sqrt(.5), 0, 0, -math.sqrt(.5)]
        mujoco.mj_forward(self.model, self.data)
        _, frames = self.rig.sample(self.data)
        _, _, labels = FloorVision(load_config()).observe(frames)
        self.assertIn("bedroom", labels)

    def test_reaches_local_goal_from_different_world_poses(self):
        from run_world import Arms, BalanceBase
        for position, yaw in (([-3.5, 2.6], -math.pi/2), ([3.5, 2.6], 0.0)):
            with self.subTest(position=position, yaw=yaw):
                mujoco.mj_resetData(self.model, self.data)
                self.data.qpos[:2] = position
                self.data.qpos[3:7] = [math.cos(yaw/2), 0, 0, math.sin(yaw/2)]
                mujoco.mj_forward(self.model, self.data)
                self.rig.reset()
                base, arms = BalanceBase(self.model, self.data), Arms(self.model, self.data)
                nav = VisionNavigator(load_config(), goal=[1, 0])
                for _ in range(5000):
                    pose, frames = self.rig.sample(self.data)
                    if frames:
                        nav.observe(frames, pose)
                        base.command(*nav.command(pose, self.data.time))
                    arms.step()
                    base.step()
                    if base.fallen or nav.done:
                        break
                self.assertFalse(base.fallen)
                self.assertTrue(nav.done, (nav.state, self.rig.odom.pose))
                self.assertGreater(math.dist(position, base.pose()[:2]), .6)

    def test_rendered_crate_blocks_the_direct_path(self):
        from build_world import build
        try:
            model = mujoco.MjModel.from_xml_path(build(obstacle=True))
            data = mujoco.MjData(model)
            # Move the obstacle in the evaluator only; perception sees RGB.
            model.geom_pos[model.geom("obstacle_crate").id] = [-3.5, 1.8, .3]
            mujoco.mj_forward(model, data)
            rig = CameraRig(model, load_config())
            try:
                pose, frames = rig.sample(data)
                nav = VisionNavigator(load_config(), goal=[1.5, 0])
                nav.observe(frames, pose)
                self.assertFalse(nav.map.safe_segment([0, 0], [1.5, 0]))
                self.assertGreater(np.count_nonzero(nav.map.evidence > 0), 0)
            finally:
                rig.close()
        finally:
            build()  # leave the normal generated world in place


if __name__ == "__main__":
    unittest.main()
