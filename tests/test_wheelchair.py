"""Physical regression checks: proportional robot geometry and actual towing."""
import math
import unittest

import mujoco
import numpy as np

import layout
from build_world import build, ROBOT_XML
from camera_rig import load_config
from push_wheelchair import PushController, set_demo_pose
from robot_base import wrap


class WheelchairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = build()

    def test_complete_robot_scale_and_calibration(self):
        source = mujoco.MjModel.from_xml_path(ROBOT_XML)
        model = mujoco.MjModel.from_xml_path(self.world)
        s = layout.ROBOT_SCALE
        # Compiled link positions and mesh dimensions must both shrink;
        # shrinking just collision proxies or link positions is insufficient.
        for bid in range(1, source.nbody):
            name = source.body(bid).name
            if name == 'mobile_base':
                continue
            np.testing.assert_allclose(model.body(name).pos, source.body(bid).pos*s,
                                       atol=1e-5, err_msg=name)
        for mid in range(source.nmesh):
            name = source.mesh(mid).name
            dest = model.mesh(name).id
            a, n = source.mesh_vertadr[mid], source.mesh_vertnum[mid]
            b = model.mesh_vertadr[dest]
            # MuJoCo recentres meshes in principal-inertia axes; nearly
            # symmetric wheels can choose different axes after scaling.
            # Distances between matching vertices are rotation invariant.
            indices = np.linspace(0, n-1, min(n, 100), dtype=int)
            original = source.mesh_vert[a+indices]
            scaled = model.mesh_vert[b+indices]
            np.testing.assert_allclose(
                np.linalg.norm(scaled-scaled[0], axis=1),
                np.linalg.norm(original-original[0], axis=1)*s,
                atol=2e-6, err_msg=name)
        for name in ('rj0', 'lj0'):
            np.testing.assert_allclose(model.joint(name).range, source.joint(name).range*s)
            np.testing.assert_allclose(model.actuator(name+'_target').ctrlrange,
                                       model.joint(name).range)
        config = load_config()
        self.assertAlmostEqual(config['wheel_radius'], .0846*s)
        self.assertAlmostEqual(config['wheel_track'], .32216*s)
        np.testing.assert_allclose(model.camera('head_cam').pos, [.16*s, 0, 1.5*s])
        self.assertAlmostEqual(model.qpos0[2], config['base_height'])

    def attach(self, demo=False):
        model = mujoco.MjModel.from_xml_path(self.world)
        data = mujoco.MjData(model)
        if demo:
            set_demo_pose(model, data)
        mujoco.mj_forward(model, data)
        push = PushController(model, data, start_state='drive')
        push.attach_chair()
        for _ in range(int(60/model.opt.timestep)):
            push.tick()
            self.assertFalse(push.base.fallen, 'Robot fell during attachment')
            if push.state == 'drive':
                break
        self.assertTrue(push.linked, 'Both handles must actually clamp before towing')
        self.assertTrue(push.grasp_locked)
        push.manual = True
        return push

    def test_attach_from_normal_spawn(self):
        self.attach()

    def test_clamp_drive_turn_reverse_and_release(self):
        push = self.attach(demo=True)
        d, m = push.data, push.model
        max_error = 0.
        def drive(v, w, seconds):
            nonlocal max_error
            push.base.command(v, w)
            for _ in range(int(seconds/m.opt.timestep)):
                push.step()
                self.assertFalse(push.base.fallen)
                self.assertTrue(np.isfinite(d.qpos).all())
                if push.linked:
                    rx, ry, yaw = push.base.pose()
                    cx, cy, cyaw = push.chair_pose()
                    c, s = math.cos(yaw), math.sin(yaw)
                    relative = [c*(cx-rx)+s*(cy-ry), -s*(cx-rx)+c*(cy-ry)]
                    error = np.linalg.norm(np.asarray(relative)-push.hitch[:2])
                    max_error = max(max_error, error)
                    self.assertLess(error, 1e-6)
                    self.assertLess(abs(wrap(cyaw-yaw-push.hitch[2])), 1e-6)
                    np.testing.assert_allclose(d.xpos[push.chair_body],
                                               d.qpos[push.chair_qadr:push.chair_qadr+3], atol=1e-9)
        start = np.array(push.chair_pose())
        drive(.08, 0, 3)
        self.assertGreater(np.linalg.norm(np.array(push.chair_pose())[:2]-start[:2]), .15)
        drive(.04, .3, 2)
        self.assertGreater(abs(wrap(push.chair_pose()[2]-start[2])), .3)
        drive(0, 0, 2)
        reverse_start = np.array(push.chair_pose())
        drive(-.08, 0, 3)
        heading = np.array([math.cos(reverse_start[2]), math.sin(reverse_start[2])])
        self.assertLess(np.dot(np.array(push.chair_pose())[:2]-reverse_start[:2], heading), -.1)
        drive(0, 0, 3)
        push.detach_chair()
        self.assertFalse(push.linked)
        self.assertIsNone(push.hitch)
        released = np.array(push.chair_pose())[:2]
        robot = np.array(push.base.pose())[:2]
        drive(-.08, 0, 3)
        self.assertGreater(np.linalg.norm(np.array(push.base.pose())[:2]-robot), .12)
        self.assertLess(np.linalg.norm(np.array(push.chair_pose())[:2]-released), .06)
        print(f'Wheelchair follow error: {max_error:.2e} m; release verified.')


if __name__ == '__main__':
    unittest.main()
