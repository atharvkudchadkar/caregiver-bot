"""Teleoperate the MuJoCo robot with Quest Browser WebXR controllers.

On the headset, open http://127.0.0.1:8765 after `adb reverse tcp:8765 tcp:8765`.
Runs its own simulation from world.xml; never starts or imports run_world.py.
Adapted from caregiver-bot vr-quest-teleop, commit 259acba. See VR_TELEOP.md.
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import shutil
import sys
import threading
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from push_wheelchair import stiffen_grasp_servos
from robot_base import Arms, BalanceBase


HERE = Path(__file__).resolve().parent
WEBXR_PAGE = HERE / "quest_controller.html"
TIMEOUT_S = 0.35
CONTROL_PERIOD_S = 0.02
MAX_HEAD_RELATIVE_REACH_M = 0.8
MAX_ABOVE_HEAD_M = 0.05  # physical arm reach above the head camera
VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS = 640, 360, 15
# WebXR: x right, y up, -z forward. Robot: x forward, y left, z up.
XR_TO_ROBOT = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])


def deadzone(value, threshold=0.15):
    value = float(np.clip(value, -1, 1))
    return 0.0 if abs(value) < threshold else math.copysign((abs(value) - threshold) / (1 - threshold), value)


class QuestInput:
    def __init__(self):
        self.lock = threading.Lock()
        self.packet = None
        self.received = 0.0

    def update(self, packet):
        if not isinstance(packet, dict) or not isinstance(packet.get("hands"), dict):
            raise ValueError("expected hands object")
        head = None
        if packet.get("active", True):
            raw_head = packet.get("head")
            if not isinstance(raw_head, dict):
                raise ValueError("headset pose is required")
            head_pos = np.asarray(raw_head.get("position"), dtype=float)
            head_quat = np.asarray(raw_head.get("orientation"), dtype=float)
            if (head_pos.shape != (3,) or head_quat.shape != (4,)
                    or not np.isfinite(head_pos).all() or not np.isfinite(head_quat).all()
                    or abs(np.linalg.norm(head_quat) - 1) > 0.1):
                raise ValueError("invalid headset pose")
            head = dict(position=head_pos, orientation=head_quat / np.linalg.norm(head_quat))
        hands = {}
        for side in ("left", "right"):
            raw = packet["hands"].get(side)
            if raw is None:
                continue
            if not isinstance(raw, dict):
                raise ValueError(f"expected {side} controller object")
            pos = np.asarray(raw.get("position"), dtype=float)
            quat = np.asarray(raw.get("orientation"), dtype=float)
            stick = np.asarray(raw.get("stick", [0, 0]), dtype=float)
            squeeze = float(raw.get("squeeze", 0))
            if (pos.shape != (3,) or quat.shape != (4,) or stick.shape != (2,)
                    or not np.isfinite(pos).all() or not np.isfinite(quat).all()
                    or not np.isfinite(stick).all() or not math.isfinite(squeeze)
                    or abs(np.linalg.norm(quat) - 1) > 0.1):
                raise ValueError(f"invalid {side} controller data")
            hands[side] = dict(position=pos, orientation=quat / np.linalg.norm(quat),
                               stick=np.clip(stick, -1, 1), squeeze=np.clip(squeeze, 0, 1))
        with self.lock:
            # Preserve an orientation-reset edge until the simulation consumes it.
            recenter = bool(packet.get("recenter")) or bool(self.packet and self.packet["recenter"])
            self.packet = dict(hands=hands, head=head, recenter=bool(packet.get("recenter")),
                               active=bool(packet.get("active", True)))
            self.packet["recenter"] = recenter
            self.received = time.monotonic()

    def latest(self):
        with self.lock:
            packet = self.packet
            if packet is not None and packet["recenter"]:
                packet = packet.copy()
                self.packet = {**packet, "recenter": False}
            return packet, time.monotonic() - self.received


class CameraFrames:
    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg = None
        self.published = 0.0

    def publish(self, image):
        with self.lock:
            self.jpeg = image
            self.published = time.monotonic()

    def latest(self):
        with self.lock:
            return self.jpeg if time.monotonic() - self.published < 0.75 else None


def make_handler(receiver, frames):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/frame.jpg":
                body = frames.latest()
                if body is None:
                    self.send_error(503, "camera warming up")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path != "/":
                self.send_error(404)
                return
            body = WEBXR_PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/input":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("packet too large or empty")
                receiver.update(json.loads(self.rfile.read(length)))
            except (ValueError, TypeError, OverflowError) as error:
                self.send_error(400, str(error))
                return
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_):
            pass

    return Handler


class ArmFollower:
    def __init__(self, model, data, arms, arm_scale=0.7):
        self.model, self.data, self.arms = model, data, arms
        self.arm_scale = arm_scale
        self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mobile_base")
        self.head_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
        self.sites = {side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_grasp")
                      for side in ("left", "right")}
        if self.head_camera_id < 0 or self.base_id < 0 or min(self.sites.values()) < 0:
            raise ValueError("World needs head_cam and both grasp sites; run python build_world.py")
        self.joints = {}
        for side, prefix in (("left", "lj"), ("right", "rj")):
            ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{i}")
                   for i in range(7)]
            self.joints[side] = (model.jnt_qposadr[ids], model.jnt_dofadr[ids],
                                 model.jnt_range[ids], [f"{prefix}{i}" for i in range(7)])
        self.origin = {}
        self.target_position = {}
        self.high_solution = {}
        self.trial = mujoco.MjData(model)

    def recenter(self):
        self.origin.clear()
        self.high_solution.clear()
        self.arms.hold_current()

    def follow(self, side, controller, head):
        m, d = self.model, self.data
        site_id = self.sites[side]
        base_rot = d.xmat[self.base_id].reshape(3, 3).copy()
        head_rot = Rotation.from_quat(head["orientation"]).as_matrix()
        forward = head_rot @ np.array([0., 0., -1.])
        heading = math.atan2(-forward[0], -forward[2])
        unyaw = Rotation.from_euler("y", -heading)
        relative_xr = unyaw.apply(
            controller["position"] - head["position"])
        relative_xr *= min(1., MAX_HEAD_RELATIVE_REACH_M / max(np.linalg.norm(relative_xr), 1e-9))
        relative_robot = self.arm_scale * (XR_TO_ROBOT @ relative_xr)
        relative_robot[2] = min(relative_robot[2], MAX_ABOVE_HEAD_M * self.arm_scale)
        target_pos = d.cam_xpos[self.head_camera_id] + base_rot @ relative_robot
        self.target_position[side] = target_pos.copy()
        vr_rot = unyaw.as_matrix() @ Rotation.from_quat(controller["orientation"]).as_matrix()
        if side not in self.origin:
            self.origin[side] = (vr_rot, base_rot.T @ d.site_xmat[site_id].reshape(3, 3))
        vr_rot0, hand_rot0 = self.origin[side]
        relative_rot = XR_TO_ROBOT @ (vr_rot @ vr_rot0.T) @ XR_TO_ROBOT.T
        target_rot = base_rot @ relative_rot @ hand_rot0

        qadr, dadr, limits, names = self.joints[side]
        self.trial.qpos[:] = d.qpos
        current_q = self.trial.qpos[qadr].copy()
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        weights = np.array([0.01, 1., 1., 1., 1., 1., 1.])
        root_weights = np.sqrt(weights)
        seeds = [current_q]
        if target_pos[2] > d.cam_xpos[self.head_camera_id, 2] - 0.2:
            elbow = np.zeros(7)
            seeds.append(np.clip(elbow, limits[:, 0], limits[:, 1]))
        best = (float("inf"), current_q)
        position_weights = (np.array([0.7, 0.7, 3.0])
                            if target_pos[2] >= d.cam_xpos[self.head_camera_id, 2]
                            else np.ones(3))
        for seed in seeds:
            q = seed.copy()
            for _ in range(30):
                self.trial.qpos[qadr] = q
                mujoco.mj_forward(m, self.trial)
                pos_error = target_pos - self.trial.site_xpos[site_id]
                if np.linalg.norm(pos_error) < 0.008:
                    break
                current_rot = self.trial.site_xmat[site_id].reshape(3, 3)
                rot_error = Rotation.from_matrix(target_rot @ current_rot.T).as_rotvec()
                mujoco.mj_jacSite(m, self.trial, jacp, jacr, site_id)
                jac = np.vstack((position_weights[:, None] * jacp[:, dadr],
                                 0.025 * jacr[:, dadr]))
                error = np.r_[position_weights * pos_error, 0.025 * rot_error]
                weighted_jac = jac * root_weights
                step = root_weights * (weighted_jac.T @ np.linalg.solve(
                    weighted_jac @ weighted_jac.T + 0.002 * np.eye(6), error))
                q = np.clip(q + np.clip(step, [-0.02] + [-0.12] * 6, [0.02] + [0.12] * 6),
                            limits[:, 0], limits[:, 1])
            self.trial.qpos[qadr] = q
            mujoco.mj_forward(m, self.trial)
            error = np.linalg.norm(position_weights * (target_pos - self.trial.site_xpos[site_id]))
            score = error + 0.005 * np.linalg.norm(q - current_q)
            if score < best[0]:
                best = score, q
        if target_pos[2] >= d.cam_xpos[self.head_camera_id, 2]:
            # A bounded position solve escapes elbow configurations in which
            # the fast Jacobian step leaves the hand below the head.
            seed = self.high_solution.get(side, current_q)
            seed = np.clip(seed, limits[:, 0] + 1e-6, limits[:, 1] - 1e-6)

            def residual(candidate):
                self.trial.qpos[qadr] = candidate
                mujoco.mj_forward(m, self.trial)
                return np.r_[position_weights * (self.trial.site_xpos[site_id] - target_pos),
                             0.02 * candidate[0]]

            def jacobian(candidate):
                self.trial.qpos[qadr] = candidate
                mujoco.mj_forward(m, self.trial)
                mujoco.mj_jacSite(m, self.trial, jacp, jacr, site_id)
                return np.vstack((position_weights[:, None] * jacp[:, dadr],
                                  np.array([0.02, 0., 0., 0., 0., 0., 0.])))

            candidate = least_squares(residual, seed, jac=jacobian,
                                      bounds=(limits[:, 0], limits[:, 1]),
                                      max_nfev=35).x
            candidate_error = np.linalg.norm(residual(candidate))
            if candidate_error > 0.05:
                # Both shoulders have a reachable raised-elbow branch. The
                # zero pose can lead the right arm into its opposite limit.
                raised_seed = np.zeros(7)
                raised_seed[1] = raised_seed[3] = 1.0
                raised_seed = np.clip(raised_seed, limits[:, 0] + 1e-6,
                                      limits[:, 1] - 1e-6)
                raised = least_squares(residual, raised_seed, jac=jacobian,
                                       bounds=(limits[:, 0], limits[:, 1]),
                                       max_nfev=50).x
                raised_error = np.linalg.norm(residual(raised))
                if raised_error < candidate_error:
                    candidate, candidate_error = raised, raised_error
            self.high_solution[side] = candidate
            score = candidate_error + 0.005 * np.linalg.norm(candidate - current_q)
            if score < best[0]:
                best = score, candidate
        else:
            self.high_solution.pop(side, None)
        q = best[1]
        for name, value in zip(names, q):
            self.arms.set(name, value)


class QuestTeleop:
    def __init__(self, model, data, receiver, arm_scale=0.7):
        self.model, self.data, self.receiver = model, data, receiver
        self.base = BalanceBase(model, data)
        self.arms = Arms(model, data)
        stiffen_grasp_servos(model, self.arms)
        self.arms.hold_current()
        self.follower = ArmFollower(model, data, self.arms, arm_scale)
        self.last_control = -1.0
        self.was_active = False
        self.tracked_hands = set()

    def hold_missing_hands(self, hands):
        for side in self.tracked_hands - set(hands):
            prefix = 'lj' if side == 'left' else 'rj'
            for name, aid in self.arms.act.items():
                if name.startswith((prefix, side + '_')):
                    adr = self.model.jnt_qposadr[self.model.actuator_trnid[aid, 0]]
                    self.arms.targets[name] = float(self.data.qpos[adr])
                    self.data.ctrl[aid] = self.arms.targets[name]
            self.follower.origin.pop(side, None)
        self.tracked_hands = set(hands)

    def tick(self):
        d = self.data
        if d.time - self.last_control >= CONTROL_PERIOD_S:
            self.last_control = d.time
            packet, age = self.receiver.latest()
            active = (packet is not None and packet["active"] and bool(packet["hands"])
                      and age < TIMEOUT_S and not self.base.fallen)
            if active:
                if packet["recenter"] or not self.was_active:
                    self.follower.recenter()
                hands = packet["hands"]
                self.hold_missing_hands(hands)
                for side, hand in hands.items():
                    self.follower.follow(side, hand, packet["head"])
                    self.arms.gripper(side, 0.0 if hand["squeeze"] >= 0.5 else 1.0)
                left = hands.get("left")
                right = hands.get("right")
                v = -0.25 * deadzone(left["stick"][1]) if left else 0.0
                w = -0.7 * deadzone(right["stick"][0]) if right else 0.0
                self.base.command(v, w)
            else:
                self.base.stop()
                if self.was_active:
                    self.follower.recenter()
                self.tracked_hands.clear()
            if active != self.was_active:
                print("VR connected: controller tracking active." if active else
                      "VR paused: stopping the base and holding the arms.", flush=True)
            self.was_active = active
        self.arms.step()
        self.base.step()
        mujoco.mj_forward(self.model, self.data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--world", type=Path, default=HERE / 'world.xml',
                        help="existing world file; this runner never rebuilds or overwrites it")
    parser.add_argument("--arm-scale", type=float, default=0.7,
                        help="metres of robot reach per metre of head-relative controller reach")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=60)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if not math.isfinite(args.arm_scale) or not 0 < args.arm_scale <= 2:
        parser.error("arm-scale must be greater than zero and at most 2")
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("seconds must be positive and finite")
    if not args.world.is_file():
        parser.error(f"World not found: {args.world}. Run python build_world.py first.")
    if sys.platform == "darwin" and not args.headless and os.environ.get("_CAREGIVER_MJPYTHON") != "1":
        launcher = shutil.which("mjpython")
        if launcher is None:
            parser.error("macOS MuJoCo viewing requires mjpython")
        env = os.environ.copy()
        env["_CAREGIVER_MJPYTHON"] = "1"
        os.execvpe(launcher, [launcher, os.path.abspath(__file__), *sys.argv[1:]], env)

    model = mujoco.MjModel.from_xml_path(str(args.world))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    receiver = QuestInput()
    frames = CameraFrames()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(receiver, frames))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Quest input: http://127.0.0.1:{args.port} (run adb reverse tcp:{args.port} tcp:{args.port})", flush=True)
    print("Independent VR simulation. Left stick: drive; right stick: turn; grips: close hands.", flush=True)
    print("Head-camera video is monoscopic. Ctrl+C or closing the viewer exits this runner.", flush=True)
    renderer = None
    last_capture = -1.0

    def capture_camera():
        nonlocal last_capture
        now = time.monotonic()
        if now - last_capture < 1 / VIDEO_FPS:
            return
        renderer.update_scene(data, camera="head_cam", scene_option=render_options)
        rgb = renderer.render()
        ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            frames.publish(jpeg.tobytes())
            last_capture = now

    try:
        teleop = QuestTeleop(model, data, receiver, args.arm_scale)
        renderer = mujoco.Renderer(model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)
        render_options = mujoco.MjvOption()
        render_options.geomgroup[3] = 0
        render_options.sitegroup[:] = 0
        capture_camera()
        if args.headless:
            deadline = time.monotonic() + args.seconds
            while time.monotonic() < deadline:
                start = time.monotonic()
                for _ in range(10):
                    teleop.tick()
                capture_camera()
                time.sleep(max(0., 10 * model.opt.timestep - (time.monotonic() - start)))
        else:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = teleop.base.base_id
                viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 4.0, -25, 135
                while viewer.is_running():
                    start = time.monotonic()
                    for _ in range(10):
                        teleop.tick()
                    capture_camera()
                    viewer.sync()
                    time.sleep(max(0., 10 * model.opt.timestep - (time.monotonic() - start)))
    except KeyboardInterrupt:
        pass
    finally:
        if renderer is not None:
            renderer.close()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
