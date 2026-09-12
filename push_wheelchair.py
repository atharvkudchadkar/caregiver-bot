"""Camera-guided, empty-wheelchair grasp and short push in MuJoCo.

Run: python3 push_wheelchair.py [--headless] [--seconds 30]
The rear ArUco marker supplies the chair pose; the known offsets from that
marker supply handle targets. This is a simulation demo, not a hardware driver.
"""
import argparse
import math
import os
import shutil
import sys
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np
from scipy.optimize import least_squares

from build_world import build
from run_world import Arms, BalanceBase, wrap


HEIGHT, WIDTH = 360, 480
MARKER_TO_HANDLE_X = -0.119  # handle grip is 119 mm behind the marker face
PUSH_SPEED = 0.08
PUSH_SECONDS = 4.0
PREGRASP_BACKOFF = 0.14
OPEN_GRIP = 1.0
CLOSED_GRIP = {"right": 0.14, "left": 0.0}
PAD_CENTER_OFFSET = 0.024  # 21 mm rubber radius + 6 mm pad half-thickness - 3 mm compression


def obj_id(model, kind, name):
    result = mujoco.mj_name2id(model, kind, name)
    if result < 0:
        raise ValueError(f"missing {name}; rebuild world.xml with python3 build_world.py")
    return result


def set_demo_pose(model, data):
    """Leave a clear, reachable rear approach in the bedroom."""
    for joint, xyz in (("base_free", (-4.8, 2.2, 0.005)),
                       ("wc_base_free", (-3.7, 2.2, -0.055))):
        jid = obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        adr = model.jnt_qposadr[jid]
        data.qpos[adr:adr + 3] = xyz
        data.qpos[adr + 3:adr + 7] = (1, 0, 0, 0)
    mujoco.mj_forward(model, data)


class MarkerDetector:
    """Decode ArUco ID 0 in head RGB and recover its plane from camera depth."""

    def __init__(self, model):
        self.model = model
        self.camera_id = obj_id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
        self.renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco = cv2.aruco.ArucoDetector(dictionary)
        fovy = math.radians(model.cam_fovy[self.camera_id])
        self.focal = HEIGHT / (2 * math.tan(fovy / 2))

    def detect(self, data):
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(data, camera="head_camera")
        rgb = self.renderer.render().copy()
        corners, ids, _ = self.aruco.detectMarkers(rgb)
        if ids is None or 0 not in ids:
            return None
        quad = corners[int(np.flatnonzero(ids.flatten() == 0)[0])].reshape(4, 2)
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(data, camera="head_camera")
        depth = self.renderer.render()
        camera_pos = data.cam_xpos[self.camera_id]
        camera_rotation = data.cam_xmat[self.camera_id].reshape(3, 3)

        def backproject(pixel):
            u, v = pixel
            ui, vi = int(round(u)), int(round(v))
            if not (0 <= ui < WIDTH and 0 <= vi < HEIGHT):
                return None
            z = float(depth[vi, ui])
            if not math.isfinite(z) or z <= 0 or z > 3:
                return None
            ray = np.array([(u + 0.5 - WIDTH / 2) * z / self.focal,
                            -(v + 0.5 - HEIGHT / 2) * z / self.focal, -z])
            return camera_pos + camera_rotation @ ray

        center_pixel = quad.mean(axis=0)
        center = backproject(center_pixel)
        inner = [backproject(center_pixel + 0.75 * (corner - center_pixel)) for corner in quad]
        if center is None or any(point is None for point in inner):
            return None
        tl, tr, br, bl = inner
        across = (tr - tl + br - bl) / 2
        down = (bl - tl + br - tr) / 2
        x_axis = np.cross(across, down)
        x_axis[2] = 0  # the demo chair remains upright on the floor
        span = np.linalg.norm(x_axis)
        if span < 0.005:
            return None
        x_axis /= span
        if np.dot(x_axis, center - camera_pos) < 0:
            x_axis = -x_axis
        y_axis = np.cross((0, 0, 1), x_axis)
        return center, x_axis, y_axis

    def close(self):
        self.renderer.close()


def handle_targets(marker_pose):
    center, x_axis, y_axis = marker_pose
    # The rendered marker face backprojects about 8 mm above its model site at
    # this oblique head-camera angle; compensate before aligning the jaw pads.
    common = center + MARKER_TO_HANDLE_X * x_axis + np.array([0, 0, 0.116])
    return {"right": common - 0.18 * y_axis, "left": common + 0.18 * y_axis}


def solve_grasp_ik(model, data, targets, handle_axis):
    """Align both finger pads around each horizontal tube, using all 7 arm joints."""
    trial = mujoco.MjData(model)
    trial.qpos[:] = data.qpos
    result = {}
    for side, prefix in (("right", "rj"), ("left", "lj")):
        joints = [obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{i}") for i in range(7)]
        qadr = model.jnt_qposadr[joints]
        lo, hi = model.jnt_range[joints].T
        lo, hi = lo + 1e-5, hi - 1e-5
        lower = obj_id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_lower_pad_site")
        upper = obj_id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_upper_pad_site")
        for finger in ("left", "right"):
            name = f"{side}_{finger}_gripper"
            joint = obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            trial.qpos[model.jnt_qposadr[joint]] = CLOSED_GRIP[side]
        start = np.clip(data.qpos[qadr], lo, hi)
        up = np.array([0.0, 0.0, 1.0])

        def residual(q):
            trial.qpos[qadr] = q
            mujoco.mj_forward(model, trial)
            low_pos, high_pos = trial.site_xpos[lower], trial.site_xpos[upper]
            low_R = trial.site_xmat[lower].reshape(3, 3)
            high_R = trial.site_xmat[upper].reshape(3, 3)
            return np.r_[
                12 * (low_pos - (targets[side] - PAD_CENTER_OFFSET * up)),
                12 * (high_pos - (targets[side] + PAD_CENTER_OFFSET * up)),
                1.5 * (low_R[:, 2] - up),
                1.5 * (high_R[:, 2] + up),
                0.5 * np.cross(low_R[:, 0], handle_axis),
                0.5 * np.cross(high_R[:, 0], handle_axis),
                0.01 * (q - start),
            ]

        best = None
        seeds = (start, np.clip(np.array([-.44, 1.5, 0, 0.25, 0, 0, 0]), lo, hi),
                 np.clip(np.array([-.44, 1.5, 0, 0.25, 1.5, 0, 1.5]), lo, hi))
        for seed in seeds:
            candidate = least_squares(residual, seed, bounds=(lo, hi), max_nfev=200)
            error = np.linalg.norm(residual(candidate.x)[:6]) / 12
            if best is None or error < best[0]:
                best = (error, candidate.x)
        if best[0] > 0.02:
            raise RuntimeError(f"{side} pads cannot surround handle ({best[0]:.3f} m IK error)")
        result.update({f"{prefix}{i}": float(q) for i, q in enumerate(best[1])})
    return result


class PushController:
    def __init__(self, model, data):
        self.model, self.data = model, data
        self.base = BalanceBase(model, data)
        self.arms = Arms(model, data)
        self.detector = MarkerDetector(model)
        self.state = "find"
        self.last_seen = -1.0
        self.marker_pose = None
        self.state_start = 0.0
        self.last_image = -1.0
        self.grasp_site = {side: obj_id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_grasp")
                           for side in ("right", "left")}
        self.pad_site = {(side, jaw): obj_id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_{jaw}_pad_site")
                         for side in ("right", "left") for jaw in ("lower", "upper")}
        self.pad_geom = {(side, jaw): obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_{jaw}_pad")
                         for side in ("right", "left") for jaw in ("lower", "upper")}
        self.handle_geoms = {side: {obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"wc_{side}_handle_grip"),
                                    obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"wc_{side}_push_handle")}
                             for side in ("right", "left")}
        self.all_handles = set().union(*self.handle_geoms.values())
        self.all_pads = set(self.pad_geom.values())
        self.blocking_geoms = {g for g in range(model.ngeom)
                               if model.geom_contype[g] in (1, 4)
                               and not (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                                           model.geom_bodyid[g]) or "").startswith("wc_")}
        self.handle_site = {side: obj_id(model, mujoco.mjtObj.mjOBJ_SITE, f"wc_{side}_handle_grasp")
                            for side in ("right", "left")}
        self.constraints = {side: obj_id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"{side}_wheelchair_grasp")
                            for side in ("right", "left")}
        self.chair_body = obj_id(model, mujoco.mjtObj.mjOBJ_BODY, "wc_wheelchair")
        self.arm_pose = {}
        self.arm_start = {}
        self.finger_dofs = {side: [(model.jnt_qposadr[j], model.jnt_dofadr[j])
                                   for finger in ("left", "right")
                                   for j in [obj_id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                                    f"{side}_{finger}_gripper")]]
                            for side in ("right", "left")}
        self.clamp_since = None

    def change(self, state):
        self.state, self.state_start = state, self.data.time
        if state == "push":
            self.push_base_start = np.array(self.base.pose()[:2])
            self.push_chair_start = self.data.xpos[self.chair_body, :2].copy()
        if state == "close":
            for geom in self.pad_geom.values():
                self.model.geom_contype[geom] = 16
                self.model.geom_conaffinity[geom] = 8
        if state == "verify":
            self.clamp_since = None
        print(f"{self.data.time:.1f}s: {state}", flush=True)

    def apply_prescribed_pose(self):
        if not self.arm_pose or self.state not in ("reach", "insert", "close", "verify", "push", "done"):
            return
        d = self.data
        qpos_before = d.qpos.copy()
        qvel_before = d.qvel.copy()
        duration = 4 if self.state == "reach" else 2
        arm_fraction = np.clip((d.time - self.state_start) / duration, 0, 1) if self.state in ("reach", "insert") else 1
        for name, (qadr, dadr, target) in self.arm_pose.items():
            d.qpos[qadr] = self.arm_start[name] + arm_fraction * (target - self.arm_start[name])
            d.qvel[dadr] = 0
        for side in ("right", "left"):
            close_fraction = np.clip((d.time - self.state_start) / 2, 0, 1) if self.state == "close" else 0
            value = OPEN_GRIP - close_fraction * (OPEN_GRIP - CLOSED_GRIP[side])
            if self.state in ("verify", "push", "done"):
                value = CLOSED_GRIP[side]
            for qadr, dadr in self.finger_dofs[side]:
                d.qpos[qadr], d.qvel[dadr] = value, 0
        mujoco.mj_forward(self.model, d)
        violation = self.handle_penetration()
        if violation:
            d.qpos[:] = qpos_before
            d.qvel[:] = qvel_before
            mujoco.mj_forward(self.model, d)
            print(f"blocked arm motion through handle: {violation}", flush=True)
            self.change("failed")

    def handle_penetration(self):
        """Reject deep non-grasp collisions before committing prescribed motion."""
        if self.state in ("reach", "insert"):
            # Pads are contact-disabled while opening/approaching, but still
            # checked geometrically so kinematic interpolation cannot tunnel.
            for side in ("right", "left"):
                for jaw in ("lower", "upper"):
                    pad = self.pad_geom[(side, jaw)]
                    for handle in self.handle_geoms[side]:
                        distance = mujoco.mj_geomDistance(self.model, self.data, pad, handle, 0.02, None)
                        if distance < -0.001:
                            return (f"{side}_{jaw}_pad", float(distance))
        for contact in self.data.contact:
            pair = {contact.geom1, contact.geom2}
            if not pair.intersection(self.all_handles):
                continue
            other = next(iter(pair - self.all_handles), None)
            if other in self.blocking_geoms and contact.dist < -0.003:
                return (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other),
                        float(contact.dist))
            if other in self.all_pads and contact.dist < -0.012:
                return (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other),
                        float(contact.dist))
        return None

    def clamp_status(self, min_contacts=2, xy_tolerance=0.015, z_tolerance=0.012):
        """Require opposing pad-to-handle contacts and near-horizontal faces."""
        d = self.data
        contacts = {(side, jaw): 0 for side in ("right", "left") for jaw in ("lower", "upper")}
        for contact in d.contact:
            pair = {contact.geom1, contact.geom2}
            if contact.dist > 0:
                continue
            for key, pad in self.pad_geom.items():
                side, _ = key
                if pad in pair and pair.intersection(self.handle_geoms[side]):
                    contacts[key] += 1
        for side in ("right", "left"):
            handle = d.site_xpos[self.handle_site[side]]
            for jaw, sign in (("lower", -1), ("upper", 1)):
                key = (side, jaw)
                pos = d.site_xpos[self.pad_site[key]]
                normal = d.site_xmat[self.pad_site[key]].reshape(3, 3)[:, 2]
                if (contacts[key] < min_contacts or np.linalg.norm(pos[:2] - handle[:2]) > xy_tolerance
                        or abs((pos[2] - handle[2]) - sign * PAD_CENTER_OFFSET) > z_tolerance
                        or sign * normal[2] > -0.9):
                    return False, contacts
        return True, contacts

    def correct_grasp(self):
        found = self.detector.detect(self.data)
        if found is None:
            print("rear marker lost during grasp", flush=True)
            self.change("failed")
            return False
        self.marker_pose = found
        try:
            corrected = solve_grasp_ik(self.model, self.data, handle_targets(found), found[1])
        except RuntimeError as error:
            print(error, flush=True)
            self.change("failed")
            return False
        qpos_before = self.data.qpos.copy()
        qvel_before = self.data.qvel.copy()
        for name, value in corrected.items():
            qadr, dadr, _ = self.arm_pose[name]
            self.arm_pose[name] = (qadr, dadr, value)
            self.arms.set(name, value)
            self.data.qpos[qadr], self.data.qvel[dadr] = value, 0
        mujoco.mj_forward(self.model, self.data)
        violation = self.handle_penetration()
        if violation:
            self.data.qpos[:] = qpos_before
            self.data.qvel[:] = qvel_before
            mujoco.mj_forward(self.model, self.data)
            print(f"grasp correction would penetrate handle: {violation}", flush=True)
            self.change("failed")
            return False
        return True

    def plan_arm_pose(self, marker_pose, backoff=0.0):
        targets = handle_targets(marker_pose)
        if backoff:
            targets = {side: point - backoff * marker_pose[1] for side, point in targets.items()}
        try:
            joints = solve_grasp_ik(self.model, self.data, targets, marker_pose[1])
        except RuntimeError as error:
            print(error, flush=True)
            self.change("failed")
            return False
        for name, value in joints.items():
            joint = obj_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qadr = self.model.jnt_qposadr[joint]
            dadr = self.model.jnt_dofadr[joint]
            self.arm_start[name] = float(self.data.qpos[qadr])
            self.arm_pose[name] = (qadr, dadr, value)
            self.arms.set(name, value)
        return True

    def tick(self):
        d = self.data
        self.apply_prescribed_pose()
        if d.time - self.last_image >= 0.1 and self.state in ("find", "approach", "align"):
            self.last_image = d.time
            found = self.detector.detect(d)
            if found is not None:
                self.marker_pose, self.last_seen = found, d.time
                if self.state == "find":
                    self.change("approach")
        if self.state in ("find", "approach", "align") and d.time - self.last_seen > 0.6:
            self.base.stop()
            if self.state != "find":
                self.change("lost marker")
        elif self.state == "approach":
            center, x_axis, _ = self.marker_pose
            bx, by, yaw = self.base.pose()
            desired = center[:2] - 0.40 * x_axis[:2]
            delta = desired - np.array([bx, by])
            heading = math.atan2(x_axis[1], x_axis[0])
            yaw_error = wrap(heading - yaw)
            distance = float(np.dot(delta, x_axis[:2]))
            lateral = float(x_axis[0] * delta[1] - x_axis[1] * delta[0])
            if abs(yaw_error) < 0.07 and abs(distance) < 0.025 and abs(lateral) < 0.03:
                self.base.stop()
                self.change("align")
            else:
                self.base.command(np.clip(0.45 * distance, -0.12, 0.12) if abs(yaw_error) < 0.2 else 0,
                                  np.clip(1.5 * yaw_error + 0.7 * lateral, -0.3, 0.3))
        elif self.state == "align":
            self.base.stop()
            if d.time - self.state_start > 0.6:
                if self.plan_arm_pose(self.marker_pose, PREGRASP_BACKOFF):
                    self.arms.gripper("right", OPEN_GRIP)
                    self.arms.gripper("left", OPEN_GRIP)
                    self.change("reach")
        elif self.state == "reach":
            if d.time - self.state_start > 4:
                found = self.detector.detect(d)
                if found is None:
                    self.change("failed")
                elif self.plan_arm_pose(found):
                    self.marker_pose = found
                    self.change("insert")
        elif self.state == "insert":
            if d.time - self.state_start > 2:
                if self.correct_grasp():
                    for side in ("right", "left"):
                        self.arms.gripper(side, CLOSED_GRIP[side])
                    self.change("close")
        elif self.state == "close" and d.time - self.state_start > 2:
            if self.correct_grasp():
                self.change("verify")
        elif self.state == "verify":
            clamped, contacts = self.clamp_status()
            if clamped:
                if self.clamp_since is None:
                    self.clamp_since = d.time
                elif d.time - self.clamp_since >= 0.5:
                    # The connect constraints supplement the verified pads;
                    # they do not stand in for the contact check.
                    for eq in self.constraints.values():
                        d.eq_active[eq] = 1
                    print(f"both handles clamped: {contacts}", flush=True)
                    self.change("push")
            else:
                self.clamp_since = None
                if d.time - self.state_start > 2:
                    print(f"no stable two-sided clamp: {contacts}", flush=True)
                    self.change("failed")
        elif self.state == "push":
            # The placeholder gripper/contact model does not transmit push
            # force reliably. A bounded virtual coupling transfers the base's
            # displacement to the chair while its wheels remain dynamic.
            base_xy = np.array(self.base.pose()[:2])
            desired = self.push_chair_start + base_xy - self.push_base_start
            error_xy = desired - d.xpos[self.chair_body, :2]
            d.xfrc_applied[self.chair_body, :2] = np.clip(400 * error_xy, -35, 35)
            clamped, contacts = self.clamp_status(min_contacts=1, xy_tolerance=0.05,
                                                  z_tolerance=0.03)
            if not clamped:
                print(f"clamp lost during push: {contacts}", flush=True)
                self.base.stop()
                self.change("failed")
            elif d.time - self.state_start >= PUSH_SECONDS:
                self.base.stop()
                self.change("done")
            else:
                self.base.command(PUSH_SPEED, 0)
        if self.state in ("lost marker", "failed", "done"):
            self.base.stop()
            d.xfrc_applied[self.chair_body, :3] = 0
        self.arms.step()
        self.base.step()
        if self.base.fallen and self.state not in ("fallen", "done"):
            self.change("fallen")

    def close(self):
        self.detector.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=30)
    args = parser.parse_args()
    if sys.platform == "darwin" and not args.headless and os.environ.get("_CAREGIVER_MJPYTHON") != "1":
        launcher = shutil.which("mjpython")
        if launcher is None:
            parser.error("the macOS MuJoCo viewer requires mjpython")
        env = os.environ.copy()
        env["_CAREGIVER_MJPYTHON"] = "1"
        os.execvpe(launcher, [launcher, os.path.abspath(__file__), *sys.argv[1:]], env)

    world = build()
    model = mujoco.MjModel.from_xml_path(world)
    # The explicit pad boxes are the gripper collision surfaces. The imported
    # visual hand/finger meshes have broad convex hulls that overlap the handle
    # even at a valid clamp, so use the pads as their contact proxies.
    for geom in range(model.ngeom):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom]) or ""
        if ("finger" in body_name or body_name in ("hand__hand", "l_hand__hand")) and model.geom_contype[geom] == 4:
            model.geom_contype[geom] = 0
    # Activate only the small jaw pads when the fingers start closing.
    model.geom_contype[model.geom_contype == 16] = 0
    model.geom_conaffinity[model.geom_conaffinity == 8] = 0
    data = mujoco.MjData(model)
    set_demo_pose(model, data)
    controller = PushController(model, data)
    try:
        if args.headless:
            for _ in range(int(args.seconds / model.opt.timestep)):
                controller.tick()
                if controller.state in ("done", "failed", "fallen", "lost marker"):
                    break
        else:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                viewer.cam.lookat[:] = (-3.8, 2.2, 0.8)
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.0, 145, -25
                while viewer.is_running() and data.time < args.seconds:
                    start = time.monotonic()
                    for _ in range(10):
                        controller.tick()
                    viewer.sync()
                    delay = 10 * model.opt.timestep - (time.monotonic() - start)
                    if delay > 0:
                        time.sleep(delay)
        chair_id = obj_id(model, mujoco.mjtObj.mjOBJ_BODY, "wc_wheelchair")
        chair_x = float(data.xpos[chair_id, 0])
        print(f"final: {controller.state}, t={data.time:.1f}s, "
              f"robot={controller.base.pose()[:2]}, chair moved {chair_x + 3.7:.3f} m")
        if args.headless and controller.state != "done":
            raise SystemExit(1)
    finally:
        controller.close()


if __name__ == "__main__":
    main()
