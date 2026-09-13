"""Known-pose wheelchair grasp, then drive robot and chair as one body.

Run: python3 push_wheelchair.py [--headless] [--seconds 55]
G lines up from the wheelchair's actual pose in the sim (always known),
then the arms pinch the handle tips. After the clamp, the chair is locked
to the robot as one vehicle. X detaches. Arrow keys drive; Space stops or
accepts a lineup. Nothing pauses or ends the run.
"""
import argparse
import math
import os
import shutil
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
from scipy.optimize import least_squares

from build_world import build
from robot_base import Arms, BalanceBase, wrap


MARKER_TO_TIP = -0.181  # handle tip is 181 mm behind the marker face
DRIVE_SPEED = 0.08
TURN_SPEED = 0.35
# Forward, left arc, forward again — shows the linked pair translating and turning.
DRIVE_PLAN = (
    (DRIVE_SPEED, 0.0, 2.5),
    (0.05, TURN_SPEED, 2.2),
    (DRIVE_SPEED, 0.0, 2.5),
)
GLFW_KEYS = {"up": 265, "down": 264, "left": 263, "right": 262, "space": 32,
             "g": 71, "x": 88}
PARK_ARM = {f"{p}{i}": 0.0 for p in ("rj", "lj") for i in range(7)}
ATTACH_STATES = ("find", "approach", "align", "arm_pregrasp", "move_in", "close", "verify")
ARM_SETUP_BACKOFF = 0.07   # open jaws hover this far behind the tip
BASE_SLIDE_DISTANCE = 0.07  # must match backoff so the tip actually enters the jaws
APPROACH_STANDOFF = 0.55    # park this far behind the rear marker before pinching
# Inflated chair footprint in its own frame: wheels/frame plus robot clearance.
CHAIR_REAR_X, CHAIR_FRONT_X, CHAIR_SIDE_Y = -0.70, 0.85, 0.64
DETOUR_REAR_X, DETOUR_FRONT_X, DETOUR_SIDE_Y = -0.94, 1.02, 0.84
ARM_SETTLE_S = 6.0
LINEUP_STATES = ("find", "approach")
CLOSE_SETTLE_S = 6.0
VERIFY_SETTLE_S = 5.0
OPEN_GRIP = 1.0
CLOSED_GRIP = {"right": 0.0, "left": 0.0}
PAD_CENTER_OFFSET = 0.021  # 13 mm tip half-thickness + 6 mm pad + 2 mm clearance
SAFE_POSE_AGE_S = 0.2  # keep rollback clear of the contact boundary


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


def handle_targets(marker_pose):
    """World points of the two handle tips (the free ends that face the robot)."""
    center, x_axis, y_axis = marker_pose
    common = center + MARKER_TO_TIP * x_axis + np.array([0, 0, 0.116])
    return {"right": common - 0.18 * y_axis, "left": common + 0.18 * y_axis}


def chair_avoidance_waypoints(start, marker_pose):
    """Route around the inflated wheelchair footprint to its rear centerline."""
    center, x_axis, y_axis = marker_pose
    chair_origin = center[:2] + 0.231 * x_axis[:2]
    def local(point):
        delta = np.asarray(point)[:2] - chair_origin
        return np.array([np.dot(delta, x_axis[:2]), np.dot(delta, y_axis[:2])])
    def world(point):
        return chair_origin + point[0] * x_axis[:2] + point[1] * y_axis[:2]
    goal = local(center[:2] - APPROACH_STANDOFF * x_axis[:2])
    source = local(start)
    # A segment enters the chair envelope if any interval overlaps in x and y.
    low, high = 0.0, 1.0
    for index, bounds in enumerate(((CHAIR_REAR_X, CHAIR_FRONT_X),
                                    (-CHAIR_SIDE_Y, CHAIR_SIDE_Y))):
        delta = goal[index] - source[index]
        if abs(delta) < 1e-9:
            if not bounds[0] <= source[index] <= bounds[1]:
                return []
            continue
        a, b = sorted(((bounds[0] - source[index]) / delta,
                       (bounds[1] - source[index]) / delta))
        low, high = max(low, a), min(high, b)
    if low > high:
        return []
    side = 1 if source[1] >= 0 else -1
    points = []
    if source[0] > CHAIR_FRONT_X:
        points.append((DETOUR_FRONT_X, side * DETOUR_SIDE_Y))
    points.append((DETOUR_REAR_X, side * DETOUR_SIDE_Y))
    return [world(point) for point in points]


def solve_grasp_ik(model, data, targets, handle_axis):
    """Put the jaw pads above and below each handle tip (the grasp that worked).

    Closing the gripper joint then pinches the stub.
    """
    trial = mujoco.MjData(model)
    trial.qpos[:] = data.qpos
    result = {}
    up = np.array([0.0, 0.0, 1.0])
    axis = handle_axis / max(np.linalg.norm(handle_axis), 1e-9)
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
        tip = targets[side]

        def residual(q):
            trial.qpos[qadr] = q
            mujoco.mj_forward(model, trial)
            low_pos, high_pos = trial.site_xpos[lower], trial.site_xpos[upper]
            low_R = trial.site_xmat[lower].reshape(3, 3)
            high_R = trial.site_xmat[upper].reshape(3, 3)
            return np.r_[
                12 * (low_pos - (tip - PAD_CENTER_OFFSET * up)),
                12 * (high_pos - (tip + PAD_CENTER_OFFSET * up)),
                1.5 * (low_R[:, 2] - up),
                1.5 * (high_R[:, 2] + up),
                0.5 * np.cross(low_R[:, 0], axis),
                0.5 * np.cross(high_R[:, 0], axis),
                0.01 * (q - start),
            ]

        best = None
        seeds = (start, np.clip(np.array([-.44, 1.5, 0, 0.25, 0, 0, 0]), lo, hi),
                 np.clip(np.array([-.44, 1.5, 0, 0.25, 1.5, 0, 1.5]), lo, hi))
        if side == "left":
            seeds = seeds + (np.clip(np.array([-.44, -1.5, 0, 0.25, 0, 0, 0]), lo, hi),)
        for seed in seeds:
            candidate = least_squares(residual, seed, bounds=(lo, hi), max_nfev=400)
            error = np.linalg.norm(residual(candidate.x)[:6]) / 12
            if best is None or error < best[0]:
                best = (error, candidate.x)
        if best[0] > 0.02:
            raise RuntimeError(f"{side} pads cannot surround handle tip ({best[0]:.3f} m IK error)")
        result.update({f"{prefix}{i}": float(q) for i, q in enumerate(best[1])})
    return result


def stiffen_grasp_servos(model, arms):
    """Give the placeholder arm inertias enough armature that a position servo
    can close a finger without sending QACC to infinity.

    The stock XML (kp=0.03, mass ~5e-5 kg) cannot squeeze a handle. Navigation
    still uses the XML values; this only retunes the in-memory model.
    """
    for name, aid in arms.act.items():
        jid = int(model.actuator_trnid[aid, 0])
        dadr = model.jnt_dofadr[jid]
        if "gripper" in name:
            kp, kv, force, rate = 0.8, 0.08, 0.5, 0.8
            armature, damping = 0.008, 0.15
        elif name.endswith("0"):
            kp, kv, force, rate = 2.0, 0.3, 1.0, 0.2
            armature, damping = 0.02, 0.4
        else:
            kp, kv, force, rate = 0.4, 0.06, 0.4, 0.5
            armature, damping = 0.006, 0.08
        model.dof_armature[dadr] = max(float(model.dof_armature[dadr]), armature)
        model.dof_damping[dadr] = max(float(model.dof_damping[dadr]), damping)
        model.actuator_gainprm[aid, 0] = kp
        model.actuator_biasprm[aid, 1] = -kp
        model.actuator_biasprm[aid, 2] = -kv
        model.actuator_forcerange[aid] = (-force, force)
        model.actuator_forcelimited[aid] = 1
        arms.ramp[name] = rate


class PushController:
    def __init__(self, model, data, base=None, arms=None, start_state="approach"):
        """`base`/`arms`, if given, are shared with an existing driver (e.g. the
        navigator's own instances in run_world.py) instead of new ones - so
        this controller and navigation drive the same actuators rather than
        fighting each other. `start_state` defaults to "approach" for the
        standalone demo below (auto-attach on launch); run_world.py passes
        "drive" so the merged app starts idle, unattached, until G is pressed
        or a voice command asks for it.
        """
        self.model, self.data = model, data
        self.base = base if base is not None else BalanceBase(model, data)
        self.arms = arms if arms is not None else Arms(model, data)
        stiffen_grasp_servos(model, self.arms)
        self.arms.hold_current()
        self.state = start_state
        self.state_start = 0.0
        self.grasp_site = {side: obj_id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_grasp")
                           for side in ("right", "left")}
        self.pad_site = {(side, jaw): obj_id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_{jaw}_pad_site")
                         for side in ("right", "left") for jaw in ("lower", "upper")}
        self.finger_geoms = {(side, jaw): {obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_{jaw}_pad")}
                             for side in ("right", "left") for jaw in ("lower", "upper")}
        self.handle_geoms = {side: {obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"wc_{side}_{part}")
                                    for part in ("handle_tip", "handle_grip", "push_handle")}
                             for side in ("right", "left")}
        self.handle_site = {side: obj_id(model, mujoco.mjtObj.mjOBJ_SITE, f"wc_{side}_handle_grasp")
                            for side in ("right", "left")}
        self.chair_body = obj_id(model, mujoco.mjtObj.mjOBJ_BODY, "wc_wheelchair")
        chair_joint = obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "wc_base_free")
        self.chair_qadr = model.jnt_qposadr[chair_joint]
        self.chair_dadr = model.jnt_dofadr[chair_joint]
        self.linked = False
        self.grasp_locked = False
        self.manual = False
        self.hitch = None
        self.drive_phase = -1
        self.arm_pose = {}
        self.arm_names = [f"{p}{i}" for p in ("rj", "lj") for i in range(7)]
        self.grip_joint = {
            ("right", "lower"): "right_left_gripper",
            ("right", "upper"): "right_right_gripper",
            ("left", "lower"): "left_left_gripper",
            ("left", "upper"): "left_right_gripper",
        }
        self.grip_latched = set()
        self.clamp_since = None
        self.barrier_geoms = {
            gid for gid in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").startswith(("wall_", "obstacle_"))
        }
        self.robot_col = {obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                          for name in ("col_base", "col_mast")}
        self.hull_id = obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, "wc_hull")
        # Every chair part, including wheels, rails, handles and footrests,
        # remains collidable while hitched.
        self.chair_mesh_geoms = []
        for gid in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if name.startswith("wc_") and name != "wc_hull":
                self.chair_mesh_geoms.append(
                    (gid, int(model.geom_contype[gid]), int(model.geom_conaffinity[gid])))
        self.block_geoms = {gid for gid, _, _ in self.chair_mesh_geoms} | {self.hull_id} | self.robot_col
        self._safe_pose = None
        self._safe_pose_time = None
        self._barrier_blocked = False
        self._barrier_origin = None
        self._planned_arms = False
        self.lineup_manual = False
        self._approach_settle = None
        self._align_settle = None
        self._approach_report = -1.0
        self.marker_site = obj_id(model, mujoco.mjtObj.mjOBJ_SITE, "wc_rear_marker_center")
        self.weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "chair_hitch")
        if self.weld_id >= 0:
            data.eq_active[self.weld_id] = 0
        self.marker_pose = self.marker_from_chair()
        self._hold_xy = None
        self.approach_waypoints = chair_avoidance_waypoints(self.base.pose()[:2], self.marker_pose)

    def change(self, state):
        self.state, self.state_start = state, self.data.time
        if state == "drive":
            self.drive_phase = -1
        if state == "approach":
            self._approach_settle = None
        if state == "align":
            self._align_settle = None
        if state in ("arm_pregrasp", "close", "verify"):
            self._hold_xy = self.data.qpos[self.base.qadr:self.base.qadr + 2].copy()
        else:
            self._hold_xy = None
        if state == "close":
            self.grip_latched.clear()
        if state == "verify":
            self.clamp_since = None
        print(f"{self.data.time:.1f}s: {state}", flush=True)

    def chair_pose(self):
        q = self.data.qpos[self.chair_qadr:self.chair_qadr + 7]
        w, x, y, z = q[3:7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return float(q[0]), float(q[1]), yaw

    def set_chair_mesh_collisions(self, enabled):
        for gid, contype, conaffinity in self.chair_mesh_geoms:
            self.model.geom_contype[gid] = contype
            self.model.geom_conaffinity[gid] = conaffinity

    def link_chair(self):
        """Lock the chair to the robot; they drive as one vehicle."""
        rx, ry, ryaw = self.base.pose()
        cx, cy, cyaw = self.chair_pose()
        dx, dy = cx - rx, cy - ry
        c, s = math.cos(ryaw), math.sin(ryaw)
        self.hitch = (c * dx + s * dy, -s * dx + c * dy, wrap(cyaw - ryaw),
                      float(self.data.qpos[self.chair_qadr + 2]))
        self.linked = True
        self.set_chair_mesh_collisions(True)
        self.apply_hitch()
        self._safe_pose = self.data.qpos.copy()
        self._safe_pose_time = self.data.time
        self._barrier_blocked = False
        self._barrier_origin = None
        print(f"  one body at ({self.hitch[0]:+.2f}, {self.hitch[1]:+.2f}) m, "
              f"{math.degrees(self.hitch[2]):+.1f} deg in the robot frame", flush=True)

    def lock_grasp(self):
        """Freeze the verified arm and finger servo targets at their contact pose."""
        clamped, contacts = self.clamp_status()
        if not clamped:
            raise RuntimeError(f"cannot lock an incomplete grasp: {contacts}")
        self.arms.hold_current()
        self.grasp_locked = True
        print("  grippers locked at verified handle contacts; X releases them", flush=True)

    def release_grasp(self):
        self.grasp_locked = False
        self.linked = False
        self.hitch = None
        self._barrier_blocked = False
        self._barrier_origin = None
        self.set_chair_mesh_collisions(True)

    def attach_chair(self):
        """Line up from the known chair pose, then reach and pinch."""
        if self.linked:
            print("already attached; press X to detach", flush=True)
            return
        if self.state in ATTACH_STATES:
            print("already lining up on the wheelchair", flush=True)
            return
        self.release_grasp()
        self.manual = False
        self.marker_pose = self.marker_from_chair()
        self.approach_waypoints = chair_avoidance_waypoints(self.base.pose()[:2], self.marker_pose)
        self.grip_latched.clear()
        self.clamp_since = None
        self._planned_arms = False
        self.lineup_manual = False
        self._approach_settle = None
        self.arms.gripper("right", OPEN_GRIP)
        self.arms.gripper("left", OPEN_GRIP)
        print("lining up on the wheelchair; arrows nudge, Space resumes auto alignment", flush=True)
        self.change("approach")

    def detach_chair(self):
        if not self.linked and self.state == "drive":
            print("not attached; press G to line up on the wheelchair", flush=True)
            return
        if self.state in ATTACH_STATES:
            self.base.stop()
            self.manual = True
            self.change("drive")
        self.release_grasp()
        self.arms.gripper("right", OPEN_GRIP)
        self.arms.gripper("left", OPEN_GRIP)
        for name, value in PARK_ARM.items():
            self.arms.set(name, value)
        self.manual = True
        if self.state != "drive":
            self.change("drive")
        print("detached; arrows drive the robot, G lines up on the wheelchair", flush=True)

    def apply_hitch(self):
        """Write the chair freejoint so it stays welded to the robot in SE(2)."""
        if not self.linked or self.hitch is None:
            return
        rx, ry, ryaw = self.base.pose()
        lx, ly, lyaw, z = self.hitch
        c, s = math.cos(ryaw), math.sin(ryaw)
        cx = rx + c * lx - s * ly
        cy = ry + s * lx + c * ly
        cyaw = ryaw + lyaw
        adr, dadr = self.chair_qadr, self.chair_dadr
        self.data.qpos[adr:adr + 3] = (cx, cy, z)
        half = 0.5 * cyaw
        self.data.qpos[adr + 3:adr + 7] = (math.cos(half), 0.0, 0.0, math.sin(half))
        r_dadr = self.base.dadr
        v_robot = self.data.qvel[r_dadr:r_dadr + 3]
        w_body = self.data.qvel[r_dadr + 3:r_dadr + 6]
        w_world = self.data.xmat[self.base.base_id].reshape(3, 3) @ w_body
        offset = np.array([cx - rx, cy - ry, 0.0])
        self.data.qvel[dadr:dadr + 3] = v_robot + np.cross(w_world, offset)
        self.data.qvel[dadr + 3:dadr + 6] = (0.0, 0.0, w_world[2])

    def wall_hits(self):
        """Outward floor-plane vectors for every chair part vs barriers."""
        hits = []
        for contact in self.data.contact:
            if contact.dist >= 0:
                continue
            pair = {contact.geom1, contact.geom2}
            if not (pair & self.barrier_geoms and pair & self.block_geoms):
                continue
            normal = np.array(contact.frame[:3], dtype=float)
            out = -normal if contact.geom1 in self.barrier_geoms else normal
            hits.append((out[:2], -float(contact.dist)))
        return hits

    def block_wall_clip(self):
        """Roll the linked pair back before any chair component clips a barrier."""
        if not self.linked:
            return
        mujoco.mj_forward(self.model, self.data)
        hits = self.wall_hits()
        if not hits:
            if self._safe_pose_time is None or self.data.time - self._safe_pose_time >= SAFE_POSE_AGE_S:
                self._safe_pose = self.data.qpos.copy()
                self._safe_pose_time = self.data.time
            if (self._barrier_origin is not None and
                    np.linalg.norm(self.data.qpos[self.base.qadr:self.base.qadr + 2]
                                   - self._barrier_origin) > 0.10):
                self._barrier_blocked = False
                self._barrier_origin = None
            return
        if self._safe_pose is not None:
            # Preserve pitch, wheel and arm dynamics so reverse/turn commands
            # can escape; only roll back the pair's blocked planar travel.
            self.data.qpos[self.base.qadr:self.base.qadr + 2] = self._safe_pose[self.base.qadr:self.base.qadr + 2]
            self.data.qvel[self.base.dadr:self.base.dadr + 2] = 0
            if self.base.v_cmd > 0:
                self.base.command(0.0, self.base.w_cmd)
            self.base.reset_targets()
            self.apply_hitch()
            mujoco.mj_forward(self.model, self.data)
            if self.wall_hits():
                self.data.qpos[self.base.qadr + 3:self.base.qadr + 7] = self._safe_pose[self.base.qadr + 3:self.base.qadr + 7]
                self.data.qvel[self.base.dadr + 3:self.base.dadr + 6] = 0
                self.apply_hitch()
                mujoco.mj_forward(self.model, self.data)
            if not self._barrier_blocked:
                self._barrier_blocked = True
                self._barrier_origin = self.data.qpos[self.base.qadr:self.base.qadr + 2].copy()
                self.manual = True
                print("  wheelchair clearance reached a barrier; steer away with arrows", flush=True)
            self.base.fallen = False
            return
        push = np.zeros(2)
        for out, depth in hits:
            span = float(np.linalg.norm(out))
            if span < 1e-8:
                continue
            push += (out / span) * (depth + 0.004)
        push = np.clip(push, -0.04, 0.04)
        self.data.qpos[self.base.qadr:self.base.qadr + 2] += push
        self.apply_hitch()
        self.base.fallen = False

    def hold_grippers_on_contact(self):
        """Keep closing until both opposing pads of each hand touch."""
        _, contacts = self.clamp_status()
        for side in ("right", "left"):
            if any(k[0] == side for k in self.grip_latched):
                continue
            if contacts[(side, "lower")] < 1 or contacts[(side, "upper")] < 1:
                continue
            name = self.grip_joint[(side, "upper")]
            aid = self.arms.act[name]
            q = float(self.data.qpos[self.model.jnt_qposadr[int(self.model.actuator_trnid[aid, 0])]])
            self.grip_latched.add((side, "lower"))
            self.grip_latched.add((side, "upper"))
            print(f"  {side} hand has opposing pad contacts at gripper {q:.2f}", flush=True)

    def clamp_status(self, min_contacts=1, xy_tolerance=0.045, z_tolerance=0.015):
        """Require opposing pad-to-handle contacts and near-horizontal faces."""
        d = self.data
        contacts = {(side, jaw): 0 for side in ("right", "left") for jaw in ("lower", "upper")}
        for contact in d.contact:
            pair = {contact.geom1, contact.geom2}
            if contact.dist > 0 or abs(contact.frame[2]) < 0.8:
                continue
            for key, geoms in self.finger_geoms.items():
                side, _ = key
                if pair.intersection(geoms) and pair.intersection(self.handle_geoms[side]):
                    contacts[key] += 1
        for side in ("right", "left"):
            handle = d.site_xpos[self.handle_site[side]]
            for jaw, sign in (("lower", -1), ("upper", 1)):
                key = (side, jaw)
                pos = d.site_xpos[self.pad_site[key]]
                normal = d.site_xmat[self.pad_site[key]].reshape(3, 3)[:, 2]
                if (contacts[key] < min_contacts or np.linalg.norm(pos[:2] - handle[:2]) > xy_tolerance
                        or abs((pos[2] - handle[2]) - sign * PAD_CENTER_OFFSET) > z_tolerance
                        or sign * normal[2] > -0.8):
                    return False, contacts
        return True, contacts

    def plan_arm_pose(self, marker_pose, lateral_backoff=0.0, abort=True):
        targets = handle_targets(marker_pose)
        if lateral_backoff:
            targets = {side: point - lateral_backoff * marker_pose[1]
                       for side, point in targets.items()}
        last_error = None
        joints = None
        for extra in (0.0, 0.03, -0.02, 0.06, 0.10):
            try:
                adjusted = {side: point - extra * marker_pose[1] for side, point in targets.items()}
                joints = solve_grasp_ik(self.model, self.data, adjusted, marker_pose[1])
                break
            except RuntimeError as error:
                last_error = error
        if joints is None:
            print(f"{last_error}; press G to try again" if abort else last_error, flush=True)
            if abort:
                self.manual = True
                self.change("drive")
            return False
        for name, value in joints.items():
            self.arm_pose[name] = value
            self.arms.set(name, value)
        return True

    def marker_from_chair(self):
        """Always-known rear-marker pose from the wheelchair freejoint."""
        center = self.data.site_xpos[self.marker_site].copy()
        _, _, cyaw = self.chair_pose()
        x_axis = np.array([math.cos(cyaw), math.sin(cyaw), 0.0])
        y_axis = np.cross((0.0, 0.0, 1.0), x_axis)
        return center, x_axis, y_axis

    def lineup_errors(self, standoff=APPROACH_STANDOFF, waypoint=None):
        """Errors to the known standoff pose behind the wheelchair."""
        center, x_axis, _ = self.marker_pose
        bx, by, yaw = self.base.pose()
        desired = (center[:2] - standoff * x_axis[:2]
                   if waypoint is None else np.asarray(waypoint))
        delta = desired - np.array([bx, by])
        dist_goal = float(np.linalg.norm(delta))
        chair_yaw = math.atan2(x_axis[1], x_axis[0])
        if dist_goal > (0.08 if waypoint is not None else 0.06):
            bearing = math.atan2(delta[1], delta[0])
            yaw_error = wrap(bearing - yaw)
        else:
            yaw_error = wrap(chair_yaw - yaw)
        distance = float(np.dot(delta, x_axis[:2]))
        lateral = float(x_axis[0] * delta[1] - x_axis[1] * delta[0])
        return yaw_error, distance, lateral, dist_goal

    def aligned_at_rear(self):
        """Check the base against the chair frame, not the bearing to a waypoint."""
        center, x_axis, y_axis = self.marker_from_chair()
        bx, by, yaw = self.base.pose()
        offset = np.array([bx, by]) - center[:2]
        behind_error = float(np.dot(offset, x_axis[:2]) + APPROACH_STANDOFF)
        center_error = float(np.dot(offset, y_axis[:2]))
        heading_error = wrap(yaw - math.atan2(x_axis[1], x_axis[0]))
        return (abs(behind_error) < 0.05 and abs(center_error) < 0.06
                and abs(heading_error) < 0.08)

    def accept_lineup(self):
        """Resume automatic lineup; Space never bypasses rear alignment."""
        self.marker_pose = self.marker_from_chair()
        self.approach_waypoints = chair_avoidance_waypoints(self.base.pose()[:2], self.marker_pose)
        self.lineup_manual = False
        self._approach_settle = None
        print("lining up at the centered rear grasp pose", flush=True)

    def drive_lineup(self, keycode):
        if not self.lineup_manual:
            self.lineup_manual = True
            self.base.stop()
            print("manual lineup: arrows move, Space resumes auto alignment, X cancels", flush=True)
        if keycode == GLFW_KEYS["space"]:
            return
        if keycode == GLFW_KEYS["up"]:
            self.base.command(self.base.v_cmd + 0.08, self.base.w_cmd)
        elif keycode == GLFW_KEYS["down"]:
            self.base.command(self.base.v_cmd - 0.08, self.base.w_cmd)
        elif keycode == GLFW_KEYS["left"]:
            self.base.command(self.base.v_cmd, self.base.w_cmd + 0.25)
        elif keycode == GLFW_KEYS["right"]:
            self.base.command(self.base.v_cmd, self.base.w_cmd - 0.25)
        print(f"cmd v={self.base.v_cmd:+.2f} m/s  w={self.base.w_cmd:+.2f} rad/s", flush=True)

    def decide(self):
        """Run one state-machine step: read the chair, set base/arm targets.

        Does not step physics or apply the hitch - callers that share `base`/
        `arms` with another driver (run_world.py) step them once themselves
        after deciding who - navigation, teleop, or this - owns the tick.
        `tick()` below wraps this for the standalone demo.
        """
        d = self.data
        # Always refresh the chair pose while lining up / reaching.
        if self.state in ("find", "approach", "align", "arm_pregrasp", "move_in"):
            self.marker_pose = self.marker_from_chair()
        if self.state == "find":
            self.change("approach")
        elif self.state == "approach":
            waypoint = self.approach_waypoints[0] if self.approach_waypoints else None
            yaw_error, distance, lateral, dist_goal = self.lineup_errors(waypoint=waypoint)
            if waypoint is not None and dist_goal < 0.10:
                self.approach_waypoints.pop(0)
                self.base.stop()
                self.base.reset_targets()
                self._approach_settle = None
                print(f"  cleared wheelchair waypoint; {len(self.approach_waypoints)} remain", flush=True)
                waypoint = self.approach_waypoints[0] if self.approach_waypoints else None
                yaw_error, distance, lateral, dist_goal = self.lineup_errors(waypoint=waypoint)
            if d.time - self._approach_report >= 2.0:
                self._approach_report = d.time
                print(f"  lineup yaw={math.degrees(yaw_error):+.1f} deg  "
                      f"goal={dist_goal:.2f} m  offset={distance:+.2f} m  side={lateral:+.2f} m"
                      f"{'  (manual)' if self.lineup_manual else ''}  "
                      f"- Space resumes auto alignment", flush=True)
            if waypoint is None and self.aligned_at_rear():
                if self._approach_settle is None:
                    self._approach_settle = d.time
                elif d.time - self._approach_settle >= 0.4:
                    self.base.stop()
                    self.change("align")
            elif self.lineup_manual:
                self._approach_settle = None
            else:
                self._approach_settle = None
                if abs(yaw_error) > 0.35:
                    self.base.command(0.0, np.clip(2.4 * yaw_error, -0.8, 0.8))
                elif dist_goal > 0.12:
                    speed = np.clip(0.55 * dist_goal, 0.04, 0.16)
                    if abs(yaw_error) > 0.2:
                        speed *= 0.35
                    self.base.command(speed, np.clip(2.0 * yaw_error, -0.7, 0.7))
                else:
                    self.base.command(np.clip(0.40 * distance, -0.08, 0.08),
                                      np.clip(2.0 * yaw_error + 0.8 * lateral, -0.55, 0.55))
        elif self.state == "align":
            self.base.stop()
            if not self.aligned_at_rear():
                self._planned_arms = False
                self.change("approach")
            elif self._align_settle is None:
                self._align_settle = d.time
            elif d.time - self._align_settle >= 0.3 and not self._planned_arms:
                self._planned_arms = True
                self.marker_pose = self.marker_from_chair()
                print("planning the pinch from the wheelchair pose", flush=True)
                if self.plan_arm_pose(self.marker_pose, lateral_backoff=ARM_SETUP_BACKOFF):
                    self.arms.gripper("right", OPEN_GRIP)
                    self.arms.gripper("left", OPEN_GRIP)
                    self.change("arm_pregrasp")
        elif self.state == "arm_pregrasp":
            self.base.stop()
            self.base.reset_targets()
            if self._hold_xy is not None:
                self.data.qpos[self.base.qadr:self.base.qadr + 2] = self._hold_xy
                self.data.qvel[self.base.dadr:self.base.dadr + 2] = 0
            if (self.arms.near(self.arm_names, tol=0.08)
                    or d.time - self.state_start > ARM_SETTLE_S):
                x_axis = self.marker_pose[1]
                self.slide_target = np.array(self.base.pose()[:2]) + BASE_SLIDE_DISTANCE * x_axis[:2]
                self.change("move_in")
        elif self.state == "move_in":
            _, x_axis, _ = self.marker_pose
            _, _, yaw = self.base.pose()
            bx, by, _ = self.base.pose()
            delta = self.slide_target - np.array([bx, by])
            distance = float(np.dot(delta, x_axis[:2]))
            lateral = float(x_axis[0] * delta[1] - x_axis[1] * delta[0])
            yaw_error = wrap(math.atan2(x_axis[1], x_axis[0]) - yaw)
            if abs(yaw_error) < 0.08 and abs(distance) < 0.012 and abs(lateral) < 0.03:
                self.base.stop()
                self.base.reset_targets()
                for side in ("right", "left"):
                    self.arms.gripper(side, CLOSED_GRIP[side])
                self.change("close")
            else:
                self.base.command(np.clip(0.35 * distance, -0.08, 0.08) if abs(yaw_error) < 0.2 else 0,
                                  np.clip(1.5 * yaw_error + 0.7 * lateral, -0.25, 0.25))
        elif self.state == "close":
            self.base.stop()
            self.base.reset_targets()
            if self._hold_xy is not None:
                self.data.qpos[self.base.qadr:self.base.qadr + 2] = self._hold_xy
                self.data.qvel[self.base.dadr:self.base.dadr + 2] = 0
            self.hold_grippers_on_contact()
            if (len(self.grip_latched) >= 4
                    or d.time - self.state_start > CLOSE_SETTLE_S):
                self.change("verify")
        elif self.state == "verify":
            self.base.stop()
            self.base.reset_targets()
            if self._hold_xy is not None:
                self.data.qpos[self.base.qadr:self.base.qadr + 2] = self._hold_xy
                self.data.qvel[self.base.dadr:self.base.dadr + 2] = 0
            clamped, contacts = self.clamp_status()
            if clamped:
                if self.clamp_since is None:
                    self.clamp_since = d.time
                elif d.time - self.clamp_since >= 0.5:
                    print(f"both handles clamped: {contacts}", flush=True)
                    self.lock_grasp()
                    self.link_chair()
                    self.change("drive")
            else:
                self.clamp_since = None
                if d.time - self.state_start > VERIFY_SETTLE_S:
                    print(f"no stable two-sided clamp: {contacts}; press G to try again",
                          flush=True)
                    self.manual = True
                    self.change("drive")
        elif self.state == "drive":
            if not self.manual:
                elapsed = d.time - self.state_start
                t = 0.0
                phase = None
                for i, (v, w, dur) in enumerate(DRIVE_PLAN):
                    if elapsed < t + dur:
                        phase = i, v, w
                        break
                    t += dur
                if phase is None:
                    self.base.stop()
                    self.manual = True
                    print("auto path finished; arrows still drive the pair", flush=True)
                else:
                    i, v, w = phase
                    if i != self.drive_phase:
                        self.drive_phase = i
                        print(f"  drive phase {i + 1}/{len(DRIVE_PLAN)}: "
                              f"v={v:.2f} m/s  w={w:+.2f} rad/s", flush=True)
                    self.base.command(v, w)

    def step(self):
        """Physics + hitch/clip maintenance for one tick, after decide()."""
        self.arms.step()
        self.base.step()
        self.apply_hitch()
        self.block_wall_clip()
        if self.base.fallen:
            self.base.fallen = False
            self.base.reset_targets()

    def tick(self):
        self.decide()
        self.step()

    def skip_phase(self):
        """Leave the current auto drive segment; last segment hands over to arrows."""
        if self.state != "drive" or self.manual:
            self.base.stop()
            return
        elapsed = self.data.time - self.state_start
        t = 0.0
        for i, (_, _, dur) in enumerate(DRIVE_PLAN):
            if elapsed < t + dur:
                self.state_start = self.data.time - (t + dur)
                print(f"  left phase {i + 1}/{len(DRIVE_PLAN)}", flush=True)
                return
            t += dur
        self.manual = True
        self.base.stop()

    def on_key(self, keycode):
        """Drive anytime; G attaches, X detaches, Space skips/stops/accepts lineup."""
        if keycode not in GLFW_KEYS.values():
            return
        if keycode == GLFW_KEYS["g"]:
            self.attach_chair()
            return
        if keycode == GLFW_KEYS["x"]:
            self.detach_chair()
            return
        if self.state in LINEUP_STATES:
            if keycode == GLFW_KEYS["space"]:
                self.accept_lineup()
                return
            if keycode in (GLFW_KEYS["up"], GLFW_KEYS["down"], GLFW_KEYS["left"], GLFW_KEYS["right"]):
                self.drive_lineup(keycode)
            return
        if self.state in ATTACH_STATES:
            return
        if keycode == GLFW_KEYS["space"] and self.state == "drive" and not self.manual:
            self.skip_phase()
            print(f"cmd v={self.base.v_cmd:+.2f} m/s  w={self.base.w_cmd:+.2f} rad/s", flush=True)
            return
        if not self.manual:
            self.manual = True
            if self.state != "drive":
                self.change("drive")
            print("manual drive: arrows move, G attach, X detach, Space stop", flush=True)
        if keycode == GLFW_KEYS["space"]:
            self.base.stop()
        elif keycode == GLFW_KEYS["up"]:
            self.base.command(self.base.v_cmd + 0.08, self.base.w_cmd)
        elif keycode == GLFW_KEYS["down"]:
            self.base.command(self.base.v_cmd - 0.08, self.base.w_cmd)
        elif keycode == GLFW_KEYS["left"]:
            self.base.command(self.base.v_cmd, self.base.w_cmd + 0.25)
        elif keycode == GLFW_KEYS["right"]:
            self.base.command(self.base.v_cmd, self.base.w_cmd - 0.25)
        print(f"cmd v={self.base.v_cmd:+.2f} m/s  w={self.base.w_cmd:+.2f} rad/s", flush=True)

    def close(self):
        return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=90)
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
    data = mujoco.MjData(model)
    set_demo_pose(model, data)
    controller = PushController(model, data)
    try:
        if args.headless:
            for _ in range(int(args.seconds / model.opt.timestep)):
                controller.tick()
                if controller.state in ("failed", "fallen", "lost marker"):
                    break
                if controller.linked and controller.manual and controller.drive_phase >= 0:
                    break
        else:
            with mujoco.viewer.launch_passive(model, data, key_callback=controller.on_key) as viewer:
                viewer.cam.lookat[:] = (-3.8, 2.2, 0.8)
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.0, 145, -25
                steps = 20
                while viewer.is_running():
                    start = time.monotonic()
                    for _ in range(steps):
                        controller.tick()
                    viewer.cam.lookat[:2] = controller.base.pose()[:2]
                    viewer.sync()
                    delay = steps * model.opt.timestep - (time.monotonic() - start)
                    if delay > 0:
                        time.sleep(delay)
        chair_id = obj_id(model, mujoco.mjtObj.mjOBJ_BODY, "wc_wheelchair")
        chair_x = float(data.xpos[chair_id, 0])
        _, _, chair_yaw = controller.chair_pose()
        print(f"final: {controller.state}, t={data.time:.1f}s, "
              f"robot={controller.base.pose()[:2]}, "
              f"chair moved {chair_x + 3.7:.3f} m, yaw {math.degrees(chair_yaw):.0f} deg")
        if args.headless and controller.state in ("failed", "fallen", "lost marker"):
            raise SystemExit(1)
    finally:
        controller.close()


if __name__ == "__main__":
    main()
