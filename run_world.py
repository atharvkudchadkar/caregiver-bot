"""Run the caregiver world: balancing robot that accepts (v, w) drive commands.

  python run_world.py                          viewer; arrow keys drive, space stops
  python run_world.py --goto kitchen           explore for the kitchen label
  python run_world.py --goto bathroom --headless --seconds 90
  python run_world.py --joint rj1=0.4 --joint rj0=-0.3 --joint gripper_right=0.8

The balance law is the teammate's (run_balance.py), rewritten in the robot's
body frame so it works at any heading and adds differential torque for
steering. Everything upstream (voice -> intent -> nav) only ever calls
BalanceBase.command(v, w); the real Bracket Bot gets the same call.

Navigation consumes three RGB cameras, wheel encoders and IMU tilt. It builds
an observed free-space map and plans online, with no apartment coordinates or
simulator base pose passed to the navigator. Room names refer to visible floor
labels; unseen destinations require exploration.

Arrow keys (viewer window must have focus):
  up/down     +/- 0.1 m/s forward speed      left/right   +/- 0.3 rad/s turn rate
  space       stop                            r            reset to spawn pose
"""
import argparse
import math
import time

import mujoco
import mujoco.viewer
import numpy as np

from camera_rig import CameraRig, load_config
from vision_navigation import VisionNavigator

WORLD = "world.xml"
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

    def step(self):
        """Ramp ctrl toward targets so a new command doesn't jerk the balancer."""
        dt = self.m.opt.timestep
        for name, aid in self.act.items():
            delta = self.targets[name] - self.d.ctrl[aid]
            step = self.ramp[name] * dt
            self.d.ctrl[aid] += float(np.clip(delta, -step, step))


def reset(model, data, base):
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    base.fallen = False
    base.stop()
    base.reset_targets()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", default=WORLD)
    goals = ap.add_mutually_exclusive_group()
    goals.add_argument("--goto", metavar="ROOM", help="find a room by its visible floor label")
    goals.add_argument("--goal", nargs=2, type=float, metavar=("X", "Y"),
                       help="goal in metres relative to startup pose, x forward, y left")
    goals.add_argument("--explore", action="store_true", help="explore visible free space")
    ap.add_argument("--vision-config", help="camera and perception calibration JSON")
    ap.add_argument("--camera-preview", action="store_true", help="show all three RGB feeds and floor masks")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=60.0, help="headless run time")
    ap.add_argument("--joint", action="append", default=[], metavar="NAME=VALUE",
                    help="arm/lift/gripper target, repeatable")
    args = ap.parse_args()
    config = load_config(args.vision_config)
    room = args.goto.lower().strip().replace(" ", "_") if args.goto else None
    if room and room not in config["marker_rooms"].values():
        ap.error(f"unknown room {room!r}; labels: {list(config['marker_rooms'].values())}")
    if args.goal and not np.isfinite(args.goal).all():
        ap.error("goal coordinates must be finite")
    model = mujoco.MjModel.from_xml_path(args.world)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    base, arms = BalanceBase(model, data), Arms(model, data)
    for entry in args.joint:
        name, _, value = entry.partition("=")
        arms.set(name.strip(), float(value))
    rig = CameraRig(model, config)
    nav = VisionNavigator(config, room=room, goal=args.goal)
    autonomous = bool(room or args.goal is not None or args.explore)
    state = None
    print("RGB navigation: head + left hand + right hand; local wheel odometry; no preset routes")

    def control_tick():
        nonlocal state
        pose, frames = rig.sample(data)
        if frames is not None:
            nav.observe(frames, pose)
            if not base.fallen:
                if autonomous:
                    base.command(*nav.command(pose, data.time))
                else:
                    base.command(*nav.guard(pose, data.time, base.v_cmd, base.w_cmd))
            if autonomous and nav.state != state:
                state = nav.state
                print(f"t={data.time:.1f}s {state} odom=({pose[0]:.2f}, {pose[1]:.2f}) "
                      f"labels={list(nav.map.targets)}", flush=True)
            if args.camera_preview:
                import cv2
                previews = []
                for frame in frames:
                    rgb = frame.rgb.copy()
                    mask = nav.vision.masks.get(frame.name, np.zeros(rgb.shape[:2], np.uint8))
                    rgb[mask > 0] = (rgb[mask > 0].astype(float)*0.65 + np.array([0, 90, 0])).clip(0, 255)
                    cv2.putText(rgb, frame.name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    previews.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cv2.imshow("Robot RGB cameras - green: estimated floor", np.hstack(previews))
                cv2.waitKey(1)
        if data.time - nav.last_frame > 0.5:
            base.stop()
        arms.step()
        base.step()

    def on_key(keycode):
        nonlocal autonomous, nav
        if keycode in (GLFW_KEYS[k] for k in ("up", "down", "left", "right", "space", "r")):
            autonomous = False
        if keycode == GLFW_KEYS["up"]:
            base.command(base.v_cmd + 0.1, base.w_cmd)
        elif keycode == GLFW_KEYS["down"]:
            base.command(base.v_cmd - 0.1, base.w_cmd)
        elif keycode == GLFW_KEYS["left"]:
            base.command(base.v_cmd, base.w_cmd + 0.3)
        elif keycode == GLFW_KEYS["right"]:
            base.command(base.v_cmd, base.w_cmd - 0.3)
        elif keycode == GLFW_KEYS["space"]:
            base.stop()
        elif keycode == GLFW_KEYS["r"]:
            reset(model, data, base)
            rig.reset()
            nav = VisionNavigator(config)
        else:
            return
        print(f"cmd v={base.v_cmd:+.1f} m/s  w={base.w_cmd:+.1f} rad/s")

    try:
        if args.headless:
            for _ in range(int(args.seconds / model.opt.timestep)):
                control_tick()
                if base.fallen or (autonomous and (nav.done or nav.state == "BLOCKED")):
                    break
            x, y, yaw = rig.odom.pose
            print(f"t={data.time:.1f}s odom=({x:.2f}, {y:.2f}, {math.degrees(yaw):.0f} deg) "
                  f"pitch={math.degrees(base.pitch()):.1f} deg fallen={base.fallen} "
                  f"navigation={nav.state}")
            return
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
                remaining = steps_per_frame * model.opt.timestep - (time.monotonic() - t0)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        rig.close()
        if args.camera_preview:
            import cv2
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
