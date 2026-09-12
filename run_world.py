"""Run the caregiver world: balancing robot that accepts (v, w) drive commands.

  python run_world.py                          viewer; arrow keys drive, space stops
  python run_world.py --goto kitchen           autonomous route to a room
  python run_world.py --goto bathroom --headless --seconds 90
  python run_world.py --joint rj1=0.4 --joint rj0=-0.3 --joint gripper_right=0.8

The balance law is the teammate's (run_balance.py), rewritten in the robot's
body frame so it works at any heading and adds differential torque for
steering. Everything upstream (voice -> intent -> nav) only ever calls
BalanceBase.command(v, w); the real Bracket Bot gets the same call.

Obstacle avoidance is reactive: nine simulated lidar beams (rangefinder
sensors added by build_world.py) slow the robot near things, steer it toward
the freer side, and stop-and-turn when the front is blocked. It is not a
planner; it gets past a crate in a hallway, not through a maze.

Arrow keys (viewer window must have focus):
  up/down     +/- 0.1 m/s forward speed      left/right   +/- 0.3 rad/s turn rate
  space       stop                            r            reset to spawn pose
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

import layout

WORLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "world.xml")
GLFW_KEYS = {"up": 265, "down": 264, "left": 263, "right": 262, "space": 32, "r": 82}


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
    (radians), plus 'gripper_right' / 'gripper_left' (0 closed .. 1 open),
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

    def step(self):
        """Ramp ctrl toward targets so a new command doesn't jerk the balancer."""
        dt = self.m.opt.timestep
        for name, aid in self.act.items():
            delta = self.targets[name] - self.d.ctrl[aid]
            step = self.ramp[name] * dt
            self.d.ctrl[aid] += float(np.clip(delta, -step, step))


class Lidar:
    """Reads the rangefinder fan added by build_world.py."""

    def __init__(self, model, data):
        self.m, self.d = model, data
        self.angles = np.radians(layout.RF_ANGLES_DEG)
        self.adr = [model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"rf_{i}")]
                    for i in range(len(layout.RF_ANGLES_DEG))]

    def ranges(self):
        r = np.array([self.d.sensordata[a] for a in self.adr])
        r[r < 0] = layout.RF_CUTOFF          # -1 means nothing within cutoff
        return r

    def sectors(self, front_deg=30):
        r = self.ranges()
        deg = np.degrees(self.angles)
        front = r[np.abs(deg) <= front_deg].min()
        left = r[deg > 0].min()
        right = r[deg < 0].min()
        return front, left, right


class Navigator:
    """Follows waypoints, with reactive obstacle avoidance from the lidar.

    States: NAV (P-controller to the waypoint), AVOID_TURN (front blocked:
    rotate toward the freer side until clear), AVOID_COMMIT (drive straight
    for a moment so we actually clear the obstacle before resuming NAV).
    """

    STOP, SLOW = 0.45, 0.9            # metres
    COMMIT_S = 1.5

    def __init__(self, waypoints, lidar=None):
        self.waypoints = list(waypoints)
        self.i = 0
        self.lidar = lidar
        self.state = "NAV"
        self.turn_dir = 1.0
        self.commit_until = 0.0

    @property
    def done(self):
        return self.i >= len(self.waypoints)

    def _nav_cmd(self, base):
        v, w, arrived = layout.step_toward(base.pose(), self.waypoints[self.i])
        if arrived:
            self.i += 1
            if self.done:
                return 0.0, 0.0
            v, w, _ = layout.step_toward(base.pose(), self.waypoints[self.i])
        return v, w

    def update(self, base, sim_time=0.0):
        if self.done:
            base.stop()
            return
        if self.lidar is None:
            base.command(*self._nav_cmd(base))
            return

        front, left, right = self.lidar.sectors()
        if self.state == "AVOID_TURN":
            if front > self.SLOW:
                self.state = "AVOID_COMMIT"
                self.commit_until = sim_time + self.COMMIT_S
            base.command(0.0, 0.5 * self.turn_dir)
            return
        if self.state == "AVOID_COMMIT":
            if front < self.STOP:
                self.state = "AVOID_TURN"
                self.turn_dir = 1.0 if left > right else -1.0
            elif sim_time > self.commit_until:
                self.state = "NAV"
            # drift toward the roomier side while committing, if there is one
            w = 0.3 * np.sign(left - right) if abs(left - right) > 0.3 else 0.0
            base.command(0.2, w)
            return

        v, w = self._nav_cmd(base)
        if front < self.STOP:
            self.state = "AVOID_TURN"
            self.turn_dir = 1.0 if left > right else -1.0
            base.command(0.0, 0.0)
            return
        if front < self.SLOW:
            k = (front - self.STOP) / (self.SLOW - self.STOP)      # 0 at STOP .. 1 at SLOW
            v = min(v, 0.05 + 0.25 * k)
            w += 0.5 * (1.0 - k) * np.sign(left - right)          # lean toward the freer side
        base.command(v, w)


def reset(model, data, base):
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    base.fallen = False
    base.stop()
    base.reset_targets()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", default=WORLD)
    ap.add_argument("--goto", metavar="ROOM", help=f"drive to one of {list(layout.ROOMS)}")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=60.0, help="headless run time")
    ap.add_argument("--joint", action="append", default=[], metavar="NAME=VALUE",
                    help="arm/lift/gripper target, repeatable (e.g. rj1=0.4, gripper_right=0.8)")
    ap.add_argument("--no-avoid", action="store_true", help="disable lidar obstacle avoidance")
    args = ap.parse_args()

    if sys.platform == "darwin" and not args.headless and os.environ.get("_CAREGIVER_MJPYTHON") != "1":
        launcher = shutil.which("mjpython")
        if launcher is None:
            ap.error("interactive MuJoCo viewing on macOS requires mjpython; install it with mujoco")
        env = os.environ.copy()
        env["_CAREGIVER_MJPYTHON"] = "1"
        os.execvpe(launcher, [launcher, os.path.abspath(__file__), *sys.argv[1:]], env)

    if args.world == WORLD and not os.path.isfile(WORLD):
        from build_world import build
        print("world.xml is missing; generating it from the robot and wheelchair models")
        build()

    model = mujoco.MjModel.from_xml_path(args.world)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    base = BalanceBase(model, data)
    arms = Arms(model, data)
    lidar = None if args.no_avoid else Lidar(model, data)

    for entry in args.joint:
        name, _, value = entry.partition("=")
        arms.set(name.strip(), float(value))

    nav = None
    if args.goto:
        room = layout.canonical_room(args.goto) or args.goto
        path = layout.route(base.pose()[:2], room)
        print(f"route to {room}: " + " -> ".join(f"({x:.1f}, {y:.1f})" for x, y in path))
        nav = Navigator(path, lidar)

    def control_tick():
        if not base.fallen:
            if nav is not None:
                nav.update(base, data.time)
            elif lidar is not None and base.v_cmd > 0:
                front, _, _ = lidar.sectors()
                if front < Navigator.STOP:           # teleop safety stop
                    base.command(0.0, base.w_cmd)
                    print("obstacle ahead: stopped")
        arms.step()
        base.step()

    if args.headless:
        n = int(args.seconds / model.opt.timestep)
        for _ in range(n):
            control_tick()
            if base.fallen or (nav is not None and nav.done):
                break
        x, y, yaw = base.pose()
        print(f"t={data.time:.1f}s pose=({x:.2f}, {y:.2f}, {math.degrees(yaw):.0f} deg) "
              f"pitch={math.degrees(base.pitch()):.1f} deg fallen={base.fallen} "
              f"{'arrived' if nav is not None and nav.done else ''}")
        return

    def on_key(keycode):
        nonlocal nav
        if keycode == GLFW_KEYS["up"]:
            base.command(base.v_cmd + 0.1, base.w_cmd)
        elif keycode == GLFW_KEYS["down"]:
            base.command(base.v_cmd - 0.1, base.w_cmd)
        elif keycode == GLFW_KEYS["left"]:
            base.command(base.v_cmd, base.w_cmd + 0.3)
        elif keycode == GLFW_KEYS["right"]:
            base.command(base.v_cmd, base.w_cmd - 0.3)
        elif keycode == GLFW_KEYS["space"]:
            nav = None
            base.stop()
        elif keycode == GLFW_KEYS["r"]:
            nav = None
            reset(model, data, base)
        else:
            return
        print(f"cmd v={base.v_cmd:+.1f} m/s  w={base.w_cmd:+.1f} rad/s")

    steps_per_frame = 10
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = base.base_id
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 4.5, -25, 135
        announced = False
        while viewer.is_running():
            t0 = time.monotonic()
            for _ in range(steps_per_frame):
                control_tick()
            viewer.sync()
            if base.fallen and not announced:
                print("fell over: press r to reset")
                announced = True
            if nav is not None and nav.done and not announced:
                print("arrived")
                announced = True
            remaining = steps_per_frame * model.opt.timestep - (time.monotonic() - t0)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
