"""Shared low-level robot drivers: the balance controller and arm servos.

Split out of run_world.py so push_wheelchair.py (the wheelchair grasp/hitch
controller) and run_world.py (navigation, voice, teleop) can both depend on
these without importing each other - run_world.py optionally hands its own
BalanceBase/Arms instances to a PushController so both drive the same
actuators instead of fighting over them.
"""
import math

import mujoco
import numpy as np


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class BalanceBase:
    """Two-wheel balancer with a (v, w) command interface."""

    V_MAX, W_MAX = 0.4, 0.8
    TORQUE_MAX = 20.0
    FALL_DEG = 20.0

    def __init__(self, model, data):
        self.m, self.d = model, data
        self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mobile_base")
        self.motors = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                       for n in ("left_motor", "right_motor")]
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_free")
        self.qadr = model.jnt_qposadr[jid]
        self.dadr = model.jnt_dofadr[jid]
        self._vel = np.zeros(6)
        self.v_cmd = self.w_cmd = 0.0
        self.fallen = False
        self.reset_targets()

    # -- commands -----------------------------------------------------------
    def command(self, v, w):
        self.v_cmd = float(np.clip(v, -self.V_MAX, self.V_MAX))
        self.w_cmd = float(np.clip(w, -self.W_MAX, self.W_MAX))

    def stop(self):
        self.command(0.0, 0.0)

    def reset_targets(self):
        self.progress_target = None   # world-frame point the base should be under
        self.yaw_target = None

    # -- state --------------------------------------------------------------
    def pose(self):
        q = self.d.qpos[self.qadr:self.qadr + 7]
        w, x, y, z = q[3:7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return float(q[0]), float(q[1]), yaw

    def pitch(self):
        R = self.d.xmat[self.base_id].reshape(3, 3)
        return math.atan2(-R[2, 0], R[2, 2])   # + = leaning forward (over +x_body)

    # -- control ------------------------------------------------------------
    def step(self):
        """One physics step. Call at model.opt.timestep."""
        m, d = self.m, self.d
        mujoco.mj_step1(m, d)             # kinematics/sensors for the current state
        pitch = self.pitch()
        if abs(pitch) > math.radians(self.FALL_DEG) or not np.isfinite(d.qpos).all():
            self.fallen = True
        if self.fallen:
            d.ctrl[self.motors] = 0.0
            mujoco.mj_step2(m, d)
            return

        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.base_id, self._vel, 1)
        pitch_rate, yaw_rate, v_fwd = self._vel[1], self._vel[2], self._vel[3]
        x, y, yaw = self.pose()
        heading = np.array([math.cos(yaw), math.sin(yaw)])
        pos = np.array([x, y])
        dt = m.opt.timestep

        # Forward-progress target: integrates v_cmd along the heading, lateral drift discarded.
        if self.progress_target is None:
            self.progress_target = pos.copy()
        self.progress_target += self.v_cmd * heading * dt
        e_fwd = float(np.clip(np.dot(self.progress_target - pos, heading), -0.3, 0.3))
        self.progress_target = pos + heading * e_fwd

        # Teammate's gains, body-frame version. Positive torque drives beneath a forward lean.
        tau_bal = (100.0 * pitch + 20.0 * pitch_rate
                   + 20.0 * (-e_fwd)
                   + 15.0 * (v_fwd - self.v_cmd))

        if self.yaw_target is None:
            self.yaw_target = yaw
        self.yaw_target = wrap(self.yaw_target + self.w_cmd * dt)
        e_yaw = float(np.clip(wrap(self.yaw_target - yaw), -0.5, 0.5))
        tau_turn = 4.0 * e_yaw + 1.0 * (self.w_cmd - yaw_rate)   # TODO tune on the real thing

        d.ctrl[self.motors[0]] = np.clip(tau_bal - tau_turn, -self.TORQUE_MAX, self.TORQUE_MAX)
        d.ctrl[self.motors[1]] = np.clip(tau_bal + tau_turn, -self.TORQUE_MAX, self.TORQUE_MAX)
        mujoco.mj_step2(m, d)


class Arms:
    """Position targets for the arm, lift and gripper servos.

    Joint names are the URDF's: rj0/lj0 (lift, metres), rj1..rj6 / lj1..lj6
    (radians), plus 'gripper_right' / 'gripper_left' (0 open .. 1 closed),
    which drive both finger servos together.
    """

    def __init__(self, model, data):
        self.m, self.d = model, data
        self.act = {}
        for aid in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
            if name.endswith("_target"):
                self.act[name[:-len("_target")]] = aid
        self.targets = {name: 0.0 for name in self.act}
        self.ramp = {name: (0.15 if name.endswith("0") else 0.6) for name in self.act}  # units/s

    def names(self):
        return list(self.act) + ["gripper_right", "gripper_left"]

    def set(self, name, value):
        if name == "gripper_right":
            self.set("right_left_gripper", value)
            self.set("right_right_gripper", value)
            return
        if name == "gripper_left":
            self.set("left_left_gripper", value)
            self.set("left_right_gripper", value)
            return
        if name not in self.act:
            raise KeyError(f"unknown joint {name!r}; one of {self.names()}")
        lo, hi = self.m.actuator_ctrlrange[self.act[name]]
        self.targets[name] = float(np.clip(value, lo, hi))

    def gripper(self, side, value):
        self.set(f"gripper_{side}", value)

    def hold_current(self):
        """Match servo setpoints to the current joint angles so nothing snaps."""
        for name, aid in self.act.items():
            q = float(self.d.qpos[self.m.jnt_qposadr[int(self.m.actuator_trnid[aid, 0])]])
            self.targets[name] = q
            self.d.ctrl[aid] = q

    def near(self, names, tol=0.05):
        for name in names:
            aid = self.act[name]
            q = self.d.qpos[self.m.jnt_qposadr[int(self.m.actuator_trnid[aid, 0])]]
            if abs(q - self.targets[name]) > tol:
                return False
        return True

    def step(self):
        """Ramp ctrl toward targets so a new command doesn't jerk the balancer."""
        dt = self.m.opt.timestep
        for name, aid in self.act.items():
            delta = self.targets[name] - self.d.ctrl[aid]
            step = self.ramp[name] * dt
            self.d.ctrl[aid] += float(np.clip(delta, -step, step))
