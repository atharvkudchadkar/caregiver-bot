"""Run with python -m unittest discover -s tests -v."""
import ast
import math
import os
import tempfile
from pathlib import Path
import unittest

import cv2
import mujoco
import numpy as np

from camera_rig import CameraFrame, CameraRig, WheelOdometry, load_config
from vision_navigation import (FloorVision, Frame2D, MapMemory, ObservedMap, VisionNavigator,
                               ground_points)

# Camera looking horizontally along +x of the base: OpenGL cam axes -> base axes.
FORWARD = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)


def sign_frame(config, marker_id, name="head_cam", pixels=96, origin=(1, 0, 1.5), t=0.0):
    """Synthetic frame: a marker of `pixels` px centred in a grey image."""
    h, w = config["height"], config["width"]
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    rgb = np.full((h, w, 3), 190, np.uint8)
    y0, x0 = (h - pixels) // 2, (w - pixels) // 2
    rgb[y0:y0 + pixels, x0:x0 + pixels] = cv2.aruco.generateImageMarker(dictionary, marker_id, pixels)[..., None]
    K = np.array([[200, 0, (w - 1) / 2], [0, 200, (h - 1) / 2], [0, 0, 1]], float)
    return CameraFrame(name, rgb, K, np.array(origin, float), FORWARD, t)


class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_encoder_odometry_and_reset(self):
        odom = WheelOdometry(0.1, 0.4)
        odom.update([20, 20])   # baseline is not a displacement
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
        grid.integrate(free, wall, np.zeros(3))
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
            grid.integrate(free, np.empty((0, 2)), np.zeros(3))
        self.assertTrue(grid.safe_segment([0, 0], [1, 0]))
        grid.integrate(np.empty((0, 2)), np.array([[.6, 0]]), np.zeros(3))
        self.assertFalse(grid.safe_segment([0, 0], [1, 0]))

    def test_missing_stale_and_black_images_stop(self):
        nav = VisionNavigator(self.config, goal=[1, 0])
        self.assertEqual(nav.command(np.zeros(3), 0), (0, 0))
        nav.last_frame = 0
        self.assertEqual(nav.command(np.zeros(3), 1), (0, 0))
        h, w = self.config["height"], self.config["width"]
        frames = [CameraFrame(c["name"], np.zeros((h, w, 3), np.uint8),
                              np.eye(3), np.array([0, 0, 1]), np.eye(3), 2)
                  for c in self.config["cameras"]]
        nav.observe(frames, np.zeros(3))
        self.assertEqual(nav.command(np.zeros(3), 2), (0, 0))
        self.assertEqual(nav.state, "CAMERA_LOST")

    def test_unknown_clearance_includes_robot_width(self):
        grid = ObservedMap(self.config)
        # A visible centre line alone is not enough clearance for the base.
        line = np.array([(x, 0) for x in np.arange(.4, 2, .05)])
        grid.integrate(line, np.empty((0, 2)), np.zeros(3))
        self.assertFalse(grid.safe_segment([0, 0], [1, 0]))

    def test_stalled_navigation_reports_blocked(self):
        nav = VisionNavigator(self.config, goal=[1, 0])
        nav.last_frame = 0
        nav.command(np.zeros(3), 0)
        nav.last_frame = 46
        self.assertEqual(nav.command(np.zeros(3), 46), (0, 0))
        self.assertEqual(nav.state, "BLOCKED")

    def test_wall_sign_gives_room_side_position_and_facing(self):
        # Marker 1 = kitchen/outside, 96 px wide at f=200 -> 0.34*200/96 = 0.708 m ahead
        # of the camera. The code sits marker_offset right of the sign centre and the sign
        # hangs sign_door_offset right of the door, so the door is `shift` to our left (+y).
        shift = self.config["marker_offset"] + self.config["sign_door_offset"]
        _, _, signs = FloorVision(self.config).observe([sign_frame(self.config, 1)])
        self.assertEqual(len(signs), 1)
        sign = signs[0]
        self.assertEqual((sign["room"], sign["side"]), ("kitchen", "outside"))
        np.testing.assert_allclose(sign["position"], [1.708, shift], atol=.03)
        np.testing.assert_allclose(sign["normal"], [-1, 0], atol=.05)   # faces the camera
        self.assertAlmostEqual(sign["height"], 1.5, delta=.05)

    def test_room_is_unknown_until_its_sign_is_seen(self):
        shift = self.config["marker_offset"] + self.config["sign_door_offset"]
        nav = VisionNavigator(self.config, room="kitchen")
        self.assertIsNone(nav.goal)
        frames = [sign_frame(self.config, 1)] + [
            CameraFrame(c["name"], np.zeros((self.config["height"], self.config["width"], 3), np.uint8),
                        np.eye(3), np.array([0, 0, 1]), np.eye(3), 0)
            for c in self.config["cameras"] if c["name"] != "head_cam"]
        nav.observe(frames, np.array([0.5, 0, 0]))
        # Door at x=2.208 facing -x; the entry point lies entry_depth beyond it, inside the room.
        door = [2.208, shift]
        np.testing.assert_allclose(nav.memory.rooms["kitchen"]["door"], door, atol=.03)
        np.testing.assert_allclose(nav.goal, [door[0] + self.config["entry_depth"], door[1]], atol=.05)

    def test_landmark_fixes_odometry_frame(self):
        # Sign seen 1 m ahead facing us; memory says it faces +y: we are looking along -y.
        T = Frame2D.from_landmark([1, 0], [-1, 0], [4, 2], [0, 1])
        np.testing.assert_allclose(T.apply([[1, 0]])[0], [4, 2], atol=1e-9)
        np.testing.assert_allclose(T.rotate([-1, 0]), [0, 1], atol=1e-9)
        np.testing.assert_allclose(T.apply_pose([0, 0, 0]), [4, 3, -math.pi / 2], atol=1e-9)

    def test_memory_survives_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "memory.npz")
            memory = MapMemory(self.config, path)
            free = np.array([(x, y) for x in np.arange(0, 2, .05) for y in np.arange(-1, 1, .05)])
            memory.grid.integrate(free, np.array([[1.5, 0]]), np.zeros(3))
            memory.update_landmark(1, "kitchen", "outside", [2, 0], [-1, 0])
            memory.update_room("kitchen")
            self.assertTrue(memory.save())
            again = MapMemory(self.config, path)
            self.assertTrue(again.loaded)
            np.testing.assert_array_equal(again.grid.evidence, memory.grid.evidence)
            np.testing.assert_array_equal(again.grid.seen, memory.grid.seen)
            np.testing.assert_allclose(again.landmarks[1]["pos"], [2, 0])
            # Outside sign facing -x: "into the room" is +x, so the entry lies beyond the door.
            np.testing.assert_allclose(again.rooms["kitchen"]["entry"], [2 + self.config["entry_depth"], 0])
            self.assertTrue(again.grid.safe_segment([0, 0], [.8, 0]))

    def test_remembered_sign_relocalizes_a_new_run(self):
        memory = MapMemory(self.config)
        memory.update_landmark(1, "kitchen", "outside", [4, 2], [0, 1])
        memory.update_room("kitchen")
        memory.loaded = True
        nav = VisionNavigator(self.config, room="kitchen", memory=memory)
        self.assertFalse(nav.localized)
        self.assertIsNone(nav.goal)   # remembered rooms are unusable until localized
        frames = [sign_frame(self.config, 1)] + [
            CameraFrame(c["name"], np.zeros((self.config["height"], self.config["width"], 3), np.uint8),
                        np.eye(3), np.array([0, 0, 1]), np.eye(3), 0)
            for c in self.config["cameras"] if c["name"] != "head_cam"]
        nav.observe(frames, np.zeros(3))   # door seen 1.708 m ahead, `shift` to our left, facing -x
        self.assertTrue(nav.localized)
        # The door faces +y in the map, so we stand 1.708 m up +y from it, looking along -y,
        # and the door is on our left (+x): we are shift to its -x side.
        shift = self.config["marker_offset"] + self.config["sign_door_offset"]
        np.testing.assert_allclose(nav.to_map(np.zeros(3))[:2], [4 - shift, 2 + 1.708], atol=.03)
        self.assertAlmostEqual(nav.to_map(np.zeros(3))[2], -math.pi / 2, delta=.05)
        np.testing.assert_allclose(nav.goal, [4, 2 - self.config["entry_depth"]], atol=.05)

    def test_navigator_has_no_simulator_or_world_dependency(self):
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
        config = load_config()
        self.assertTrue(all(f.rgb.shape == (config["height"], config["width"], 3) for f in frames))
        self.assertNotIn("overview", [f.name for f in frames])

    def test_mounts_follow_hand_joints_and_ignore_world_pose(self):
        pose, before = self.rig.sample(self.data)
        self.data.qpos[:3] += [2, -1, 0]
        angle = math.pi / 4
        self.data.qpos[3:7] = [math.cos(angle / 2), 0, 0, math.sin(angle / 2)]
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
        self.assertGreater(np.linalg.norm(moved[1].rotation - bent[1].rotation), .2)

    def test_rendered_door_sign_is_detected_from_spawn(self):
        # Spawn: bedroom at y=2.6 facing the door wall (y=0.75). The inside bedroom sign
        # hangs on that wall to the right of the door; the recovered door centre is dead ahead.
        _, frames = self.rig.sample(self.data)
        _, _, signs = FloorVision(load_config()).observe(frames)
        found = {(s["room"], s["side"]): s for s in signs}
        self.assertIn(("bedroom", "inside"), found, [(s["room"], s["side"]) for s in signs])
        sign = found[("bedroom", "inside")]
        np.testing.assert_allclose(sign["position"], [2.6 - 0.827, 0], atol=.25)   # robot frame
        np.testing.assert_allclose(sign["normal"], [-1, 0], atol=.2)              # faces the robot
        self.assertAlmostEqual(sign["height"], 0.95, delta=.15)

    def test_reaches_local_goal_from_different_world_poses(self):
        from run_world import Arms, BalanceBase
        for position, yaw in (([-3.5, 2.6], -math.pi / 2), ([3.5, 2.6], 0.0)):
            with self.subTest(position=position, yaw=yaw):
                mujoco.mj_resetData(self.model, self.data)
                self.data.qpos[:2] = position
                self.data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
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
            build()   # leave the normal generated world in place


if __name__ == "__main__":
    unittest.main()
