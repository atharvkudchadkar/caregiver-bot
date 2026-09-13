"""Standalone VR transport, controller mapping and MuJoCo regressions."""
import copy
import hashlib
import json
import math
import socket
from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from quest_teleop import (CameraFrames, QuestInput, QuestTeleop,
                          XR_TO_ROBOT, make_handler, TIMEOUT_S)
from robot_base import wrap

ROOT = Path(__file__).resolve().parents[1]


def packet():
    return dict(active=True, recenter=False,
                head=dict(position=[0, 1.6, 0], orientation=[0, 0, 0, 1]),
                hands={s: dict(position=[x, 1.2, -.25], orientation=[0, 0, 0, 1],
                               stick=[0, 0], squeeze=0) for s, x in [('left', -.2), ('right', .2)]})


class QuestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT/'world.xml').exists():
            from build_world import build
            build()

    def simulation(self):
        m = mujoco.MjModel.from_xml_path(str(ROOT/'world.xml'))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        receiver = QuestInput()
        teleop = QuestTeleop(m, d, receiver)
        return m, d, receiver, teleop

    def home_packet(self, teleop):
        p = packet()
        d, follower = teleop.data, teleop.follower
        rot = d.xmat[follower.base_id].reshape(3, 3)
        for side in p['hands']:
            relative = rot.T @ (d.site_xpos[follower.sites[side]] - d.cam_xpos[follower.head_camera_id])
            p['hands'][side]['position'] = (np.array(p['head']['position']) + XR_TO_ROBOT.T @ relative/.7).tolist()
        return p

    def test_validation_and_recenter_edge(self):
        receiver = QuestInput()
        p = packet()
        for bad in [None, [], {'hands': []}, {**p, 'hands': {'left': 2}}]:
            with self.assertRaises((ValueError, TypeError)):
                receiver.update(bad)
        for key, value in [('position', [float('nan'), 0, 0]), ('orientation', [0, 0, 0, 0]),
                           ('stick', [0]), ('squeeze', float('inf'))]:
            bad = copy.deepcopy(p)
            bad['hands']['left'][key] = value
            with self.assertRaises(ValueError):
                receiver.update(bad)
        receiver.update({**p, 'recenter': True})
        receiver.update(p)
        self.assertTrue(receiver.latest()[0]['recenter'])
        self.assertFalse(receiver.latest()[0]['recenter'])
        receiver.update({'active': False, 'hands': {}})
        self.assertFalse(receiver.latest()[0]['active'])

    def test_physics_step_mapping_and_timeout(self):
        m, d, receiver, teleop = self.simulation()
        p = self.home_packet(teleop)
        p['hands']['left']['stick'] = [0, -1]
        p['hands']['left']['squeeze'] = 1
        receiver.update(p)
        teleop.tick()
        self.assertAlmostEqual(d.time, m.opt.timestep)
        self.assertAlmostEqual(teleop.base.v_cmd, .25)
        self.assertAlmostEqual(teleop.base.w_cmd, 0)
        self.assertTrue(teleop.push.linked)
        self.assertEqual(teleop.arms.targets['left_left_gripper'], 0)
        self.assertEqual(teleop.arms.targets['right_left_gripper'], 0)
        receiver.received -= TIMEOUT_S+1
        for _ in range(12):
            teleop.tick()
        self.assertEqual(teleop.base.v_cmd, 0)
        self.assertEqual(teleop.base.w_cmd, 0)
        self.assertFalse(teleop.was_active)
        self.assertFalse(teleop.base.fallen)

    def test_missing_controller_holds_its_arm(self):
        m, d, receiver, teleop = self.simulation()
        p = self.home_packet(teleop)
        receiver.update(p)
        teleop.tick()
        teleop.arms.set('lj2', .5)
        del p['hands']['left']
        receiver.update(p)
        for _ in range(12):
            teleop.tick()
        self.assertEqual(teleop.tracked_hands, {'right'})
        self.assertLess(abs(teleop.arms.targets['lj2']), .05)
        self.assertEqual(teleop.base.v_cmd, 0)

    def test_head_yaw_physically_drives_equivalent_turn_without_falling(self):
        m, d, receiver, teleop = self.simulation()
        p = self.home_packet(teleop)
        start = np.array(teleop.base.pose())
        p['hands']['left']['stick'] = [0, -.6]
        receiver.update(p)
        teleop.tick()  # establish neutral head and base headings
        turn = .5
        p['head']['orientation'] = Rotation.from_euler('y', turn).as_quat().tolist()
        for step in range(int(3/m.opt.timestep)):
            if step % 10 == 0:
                receiver.update(p)
            teleop.tick()
            self.assertFalse(teleop.base.fallen)
        end = np.array(teleop.base.pose())
        self.assertGreater(np.linalg.norm(end[:2]-start[:2]), .2)
        self.assertAlmostEqual(wrap(end[2]-start[2]), turn, delta=.08)
        self.assertTrue(np.isfinite(d.qpos).all())

    def test_vr_arms_collide_with_every_chair_part(self):
        m, _, _, teleop = self.simulation()
        arm_geoms = [gid for gid in range(m.ngeom)
                     if int(m.geom_bodyid[gid]) in teleop.arm_bodies]
        chair_geoms = [gid for gid in range(m.ngeom) if m.geom(gid).name.startswith('wc_')]
        self.assertTrue(arm_geoms)
        self.assertTrue(chair_geoms)
        for arm in arm_geoms:
            for chair in chair_geoms:
                self.assertTrue((m.geom_contype[arm] & m.geom_conaffinity[chair]) or
                                (m.geom_contype[chair] & m.geom_conaffinity[arm]),
                                (m.geom(arm).name, m.geom(chair).name))

    def test_grip_press_auto_attaches_at_chair_orientation_and_toggles_release(self):
        m, d, receiver, teleop = self.simulation()
        chair_yaw = .65
        a = teleop.push.chair_qadr
        d.qpos[a:a+2] = [-4.15, 2.45]
        d.qpos[a+3:a+7] = [math.cos(chair_yaw/2), 0, 0, math.sin(chair_yaw/2)]
        mujoco.mj_forward(m, d)
        p = self.home_packet(teleop)
        p['hands']['right']['squeeze'] = 1
        receiver.update(p)
        teleop.tick()
        self.assertTrue(teleop.push.linked)
        self.assertAlmostEqual(wrap(teleop.base.pose()[2]-chair_yaw), 0, delta=.01)
        for side in ('left', 'right'):
            self.assertLess(np.linalg.norm(d.site_xpos[teleop.follower.sites[side]] -
                                           d.site_xpos[teleop.push.handle_site[side]]), .03)
        hitch = np.array(teleop.push.hitch[:3])
        p['hands']['right']['squeeze'] = 0
        p['hands']['left']['stick'] = [0, -.5]
        p['head']['orientation'] = Rotation.from_euler('y', .2).as_quat().tolist()
        for step in range(750):
            if step % 10 == 0:
                receiver.update(p)
            teleop.tick()
            self.assertFalse(teleop.base.fallen)
        np.testing.assert_allclose(teleop.push.chair_local_pose(), hitch, atol=1e-6)
        self.assertAlmostEqual(wrap(teleop.base.pose()[2]-chair_yaw), .2, delta=.08)
        p['hands']['right']['squeeze'] = 1
        receiver.update(p)
        for _ in range(12): teleop.tick()
        self.assertFalse(teleop.push.linked)

    def test_grip_press_outside_two_metres_does_not_attach(self):
        m, d, receiver, teleop = self.simulation()
        a = teleop.push.chair_qadr
        d.qpos[a:a+2] = [0, 0]
        mujoco.mj_forward(m, d)
        p = self.home_packet(teleop)
        p['hands']['left']['squeeze'] = 1
        receiver.update(p); teleop.tick()
        self.assertFalse(teleop.push.linked)

    def test_both_arm_targets_follow_upward_motion_and_joint_limits(self):
        m, d, receiver, teleop = self.simulation()
        p = self.home_packet(teleop)
        receiver.update(p)
        normalized, _ = receiver.latest()
        for side in ('left', 'right'):
            initial = d.site_xpos[teleop.follower.sites[side]].copy()
            hand = normalized['hands'][side]
            hand['position'][1] += .1/.7
            teleop.follower.follow(side, hand, normalized['head'])
            desired = teleop.follower.target_position[side]
            np.testing.assert_allclose(desired-initial, [0, 0, .1], atol=1e-6)
            trial = mujoco.MjData(m)
            trial.qpos[:] = d.qpos
            qadr, _, limits, names = teleop.follower.joints[side]
            targets = [teleop.arms.targets[name] for name in names]
            self.assertTrue(np.all(targets >= limits[:, 0]))
            self.assertTrue(np.all(targets <= limits[:, 1]))
            trial.qpos[qadr] = targets
            mujoco.mj_forward(m, trial)
            self.assertLess(np.linalg.norm(trial.site_xpos[teleop.follower.sites[side]]-desired), .03)

    def test_http_transport_validation_and_stale_camera(self):
        receiver, frames = QuestInput(), CameraFrames()
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(receiver, frames))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(url) as response:
                self.assertIn(b'Enter VR and connect', response.read())
            ok, jpg = cv2.imencode('.jpg', np.zeros((36, 64, 3), np.uint8))
            frames.publish(jpg.tobytes())
            with urlopen(url+'/frame.jpg') as response:
                self.assertEqual(response.headers['Content-Type'], 'image/jpeg')
                self.assertEqual(response.read(), jpg.tobytes())
            frames.published -= 1
            with self.assertRaises(HTTPError) as error:
                urlopen(url+'/frame.jpg')
            self.assertEqual(error.exception.code, 503)
            req = Request(url+'/input', data=json.dumps(packet()).encode(),
                          headers={'Content-Type': 'application/json'})
            with urlopen(req) as response:
                self.assertEqual(response.status, 204)
            event = Request(url+'/event', data=json.dumps({'message': 'session ready'}).encode(),
                            headers={'Content-Type': 'application/json'})
            with urlopen(event) as response:
                self.assertEqual(response.status, 204)
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(url+'/input', data=b'{"hands":{"left":2},"active":false}'))
            self.assertEqual(error.exception.code, 400)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)

    def test_runner_streams_without_modifying_world_or_main_app(self):
        paths = [ROOT/'run_world.py', ROOT/'world.xml']
        before = [hashlib.sha256(p.read_bytes()).digest() for p in paths]
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        process = subprocess.Popen([sys.executable, str(ROOT/'quest_teleop.py'), '--headless',
                                    '--seconds', '3', '--port', str(port)], cwd=ROOT,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            camera = None
            deadline = time.monotonic()+10
            while time.monotonic() < deadline and process.poll() is None:
                try:
                    with urlopen(f'http://127.0.0.1:{port}/frame.jpg', timeout=1) as response:
                        camera = cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)
                        break
                except (URLError, TimeoutError):
                    time.sleep(.05)
            self.assertIsNotNone(camera, 'Standalone runner must serve its actual rendered head camera')
            self.assertEqual(camera.shape, (360, 640, 3))
            self.assertGreater(camera.std(), 10, 'Camera must show scene content, not a blank image')
            stdout, stderr = process.communicate(timeout=15)
            self.assertEqual(process.returncode, 0, stdout+stderr)
            self.assertIn('Independent VR simulation', stdout)
        finally:
            if process.poll() is None:
                process.terminate()
            process.communicate(timeout=5)
        self.assertEqual(before, [hashlib.sha256(p.read_bytes()).digest() for p in paths])
        result = subprocess.run([sys.executable, '-c',
                                 "import quest_teleop, sys; assert 'run_world' not in sys.modules"],
                                cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
