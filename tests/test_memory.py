import json
import math
from pathlib import Path
import uuid
import unittest

import cv2
import mujoco
import numpy as np

from camera_rig import CameraFrame, CameraRig, load_config
from navigation_memory import NavigationMemory, compose, inverse
from room_signs import LandmarkObservation, SignReader, sign_image
from vision_navigation import ObservedMap, VisionNavigator


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        Path("verification/test_temp").mkdir(parents=True, exist_ok=True)
        self.path = Path("verification/test_temp")/f"memory_{uuid.uuid4().hex}.npz"
        self.addCleanup(lambda: self.path.unlink(missing_ok=True))

    def test_roundtrip_is_learned_and_relocalizes_after_pose_change(self):
        grid = ObservedMap(self.config)
        memory = NavigationMemory(grid, self.path)
        self.assertFalse(memory.landmarks)
        self.assertFalse(memory.rooms)
        marker = LandmarkObservation(17, np.array([2, 1, .95]), math.pi, "bedroom", "inside", 0)
        for _ in range(3):
            memory.update([marker], np.zeros(3))
        free = np.array([[.4, 0], [.5, .1], [.7, .2]])
        grid.integrate(free, np.array([[1, -.8]]), {}, np.zeros(3))
        self.assertTrue(memory.save())
        loaded_grid = ObservedMap(self.config)
        loaded = NavigationMemory(loaded_grid, self.path)
        np.testing.assert_array_equal(grid.evidence, loaded_grid.evidence)
        np.testing.assert_array_equal(grid.traveled, loaded_grid.traveled)
        self.assertIn("bedroom", loaded.rooms)
        self.assertFalse(loaded.localized)
        self.assertFalse(loaded.save())
        actual_start = np.array([.5, -.3, .4])  # evaluator-only arbitrary new starting pose
        relative = compose(inverse(actual_start), [2, 1, math.pi])
        observed = LandmarkObservation(17, np.r_[relative[:2], .95], relative[2], None, None, 0)
        for _ in range(3):
            loaded.update([observed], np.zeros(3))
        self.assertTrue(loaded.localized)
        np.testing.assert_allclose(loaded.pose(np.zeros(3)), actual_start, atol=1e-6)
        # Translation subsequently works by odometry, with no code visible.
        loaded.update([], [.5, 0, 0])
        np.testing.assert_allclose(loaded.pose([.5, 0, 0]), compose(actual_start, [.5, 0, 0]))

    def test_codes_do_not_supply_room_semantics(self):
        memory = NavigationMemory(ObservedMap(self.config), self.path)
        memory.update([LandmarkObservation(3, np.array([2, 1, .95]), math.pi, None, None, 0)], np.zeros(3))
        self.assertIn("3", memory.landmarks)
        self.assertFalse(memory.rooms)
        self.assertFalse(memory.grid.targets)

    def test_unknown_code_cannot_unlock_saved_map(self):
        memory = NavigationMemory(ObservedMap(self.config), self.path)
        memory.update([LandmarkObservation(1, np.array([2, 1, .95]), math.pi, None, None, 0)], np.zeros(3))
        memory.save()
        nav = VisionNavigator(self.config, goal=[1, 0], memory_path=self.path)
        nav.memory.update([LandmarkObservation(20, np.array([2, 1, .95]), math.pi, None, None, 0)], np.zeros(3))
        nav.last_frame = 0
        self.assertEqual(nav.command(np.zeros(3), 0)[0], 0)
        self.assertFalse(nav.memory.localized)

    def test_corrupt_memory_does_not_get_silently_overwritten(self):
        self.path.write_bytes(b"not a valid archive")
        with self.assertRaisesRegex(ValueError, "Cannot load memory"):
            NavigationMemory(ObservedMap(self.config), self.path)
        self.assertEqual(self.path.read_bytes(), b"not a valid archive")

    def test_english_recognition_is_independent_of_marker_id(self):
        reader = SignReader(self.config)
        # Fronto-parallel vertical board in a calibrated synthetic RGB camera.
        for marker_id, name in ((7, "kitchen"), (31, "bathroom")):
            rgb = cv2.cvtColor(sign_image(marker_id, name, "entrance"), cv2.COLOR_GRAY2RGB)
            frame = CameraFrame("head_cam", rgb, np.array([[1000., 0, 400], [0, 1000., 300], [0, 0, 1]]),
                                np.array([0, 0, .84]), np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]), 0)
            labels = reader.read(frame)
            self.assertEqual(len(labels), 1)
            self.assertEqual((labels[0].marker_id, labels[0].room, labels[0].face), (marker_id, name, "entrance"))

    def test_occluded_side_line_cannot_be_guessed(self):
        image = sign_image(12, "bathroom", "entrance")
        image[475:580] = 255
        frame = CameraFrame("head_cam", cv2.cvtColor(image, cv2.COLOR_GRAY2RGB),
                            np.array([[1000., 0, 400], [0, 1000., 300], [0, 0, 1]]),
                            np.array([0, 0, .84]), np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]), 0)
        observations = SignReader(self.config).read(frame)
        self.assertEqual(len(observations), 1)
        self.assertIsNone(observations[0].room)
        self.assertIsNone(observations[0].face)

    def test_one_bad_reading_cannot_change_an_entrance_into_inside(self):
        memory = NavigationMemory(ObservedMap(self.config))
        observation = LandmarkObservation(19, np.array([2, 0, .95]), math.pi, "bathroom", "entrance", 0)
        for _ in range(3):
            memory.update([observation], np.zeros(3))
        self.assertFalse(memory.rooms["bathroom"]["visited"])
        observation.face = "inside"
        memory.update([observation], np.zeros(3))
        self.assertEqual(memory.landmarks["19"]["face"], "entrance")
        self.assertFalse(memory.rooms["bathroom"]["visited"])

    def test_rendered_relocalization_at_a_new_start(self):
        from build_world import build
        model = mujoco.MjModel.from_xml_path(build())
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        rig = CameraRig(model, self.config)
        self.addCleanup(rig.close)
        nav = VisionNavigator(self.config, memory_path=self.path)
        odom, frames = rig.sample(data)
        nav.observe(frames, odom)
        nav.memory.save()
        # Same room, physically displaced startup; no world pose goes to nav.
        data.qpos[1] -= .4
        mujoco.mj_forward(model, data)
        restored = VisionNavigator(self.config, memory_path=self.path)
        for i in range(3):
            rig.reset()
            odom, frames = rig.sample(data)
            restored.observe(frames, odom)
        self.assertTrue(restored.memory.localized)
        np.testing.assert_allclose(restored.current_pose[:2], [.4, 0], atol=.10)

    def test_navigation_continues_when_codes_leave_view(self):
        from build_world import build
        from run_world import Arms, BalanceBase
        model = mujoco.MjModel.from_xml_path(build())
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        rig = CameraRig(model, self.config)
        self.addCleanup(rig.close)
        first = VisionNavigator(self.config, memory_path=self.path)
        odom, frames = rig.sample(data)
        first.observe(frames, odom)
        first.memory.save()
        restored = VisionNavigator(self.config, goal=[1, 0], memory_path=self.path)
        for _ in range(3):
            restored.observe(frames, odom)
        self.assertTrue(restored.memory.localized)
        # The code detector now supplies no landmarks. RGB floor perception and
        # the saved occupancy map remain available to the controller.
        restored.vision.sign_reader.read = lambda frame: []
        base, arms = BalanceBase(model, data), Arms(model, data)
        for _ in range(5000):
            odom, frames = rig.sample(data)
            if frames:
                restored.observe(frames, odom)
                base.command(*restored.command(odom, data.time))
            arms.step()
            base.step()
            if base.fallen or restored.done:
                break
        self.assertFalse(base.fallen)
        self.assertTrue(restored.done)


if __name__ == "__main__":
    unittest.main()
