"""Simulation sensor adapter. Navigation only receives RGB and proprioception.

Camera extrinsics are computed with joint encoders on a separate, local FK
state. World position/yaw, depth buffers and scene geometry are never passed
to perception. IMU attitude supplies roll/pitch; wheel encoders supply odometry.
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


def load_config(path=None):
    return json.loads(Path(path or Path(__file__).with_name("vision_config.json")).read_text())


@dataclass
class CameraFrame:
    name: str
    rgb: np.ndarray
    K: np.ndarray
    origin: np.ndarray          # local gravity-aligned base frame, metres
    rotation: np.ndarray        # OpenGL camera axes -> local base frame
    time: float


class WheelOdometry:
    def __init__(self, radius, track):
        self.radius, self.track = radius, track
        self.reset()

    def reset(self):
        self.pose = np.zeros(3)
        self.previous = None

    def update(self, angles):
        angles = np.asarray(angles, dtype=float)
        if self.previous is not None:
            dl, dr = (angles - self.previous) * self.radius
            distance, turn = (dl + dr) / 2, (dr - dl) / self.track
            yaw = self.pose[2] + turn / 2
            self.pose[:2] += distance * np.array([math.cos(yaw), math.sin(yaw)])
            self.pose[2] = (self.pose[2] + turn + math.pi) % (2 * math.pi) - math.pi
        self.previous = angles.copy()
        return self.pose.copy()


class CameraRig:
    def __init__(self, model, config):
        self.model, self.config = model, config
        self.names = [c["name"] for c in config["cameras"]]
        self.ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, n) for n in self.names]
        if len(self.ids) != 3 or min(self.ids) < 0:
            raise ValueError("World needs head_cam and both hand cameras; run python build_world.py")
        self.encoder_adr = [model.jnt_qposadr[model.joint(n).id] for n in
                            ("left_wheel_joint", "right_wheel_joint")]
        self.imu_adr = model.sensor("base_orientation").adr[0]
        self.base_qadr = model.jnt_qposadr[model.joint("base_free").id]
        self.local = mujoco.MjData(model)
        self.odom = WheelOdometry(config["wheel_radius"], config["wheel_track"])
        # Only scalar robot joints are copied into the local FK state.
        self.joint_adr = []
        for j in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if not name.startswith("wc_") and model.jnt_type[j] in (
                    mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                self.joint_adr.append(model.jnt_qposadr[j])
        self.renderer = mujoco.Renderer(model, height=config["height"], width=config["width"])
        self.options = mujoco.MjvOption()
        self.options.geomgroup[3] = 0  # hidden collision proxies, as in the viewer
        self.options.sitegroup[:] = 0
        self.reset()

    def reset(self):
        self.odom.reset()
        self.frames = []
        self.next_frame = 0.0

    def sample(self, data):
        pose = self.odom.update(data.qpos[self.encoder_adr])
        if data.time + 1e-8 < self.next_frame:
            return pose, None
        self.next_frame = float(data.time) + 1 / self.config["fps"]
        self.local.qpos[:] = self.model.qpos0
        self.local.qpos[self.joint_adr] = data.qpos[self.joint_adr]
        # Simulated IMU orientation. Strip global yaw; only gravity tilt is used.
        qw, qx, qy, qz = data.sensordata[self.imu_adr:self.imu_adr + 4]
        roll = math.atan2(2 * (qw*qx + qy*qz), 1 - 2 * (qx*qx + qy*qy))
        pitch = math.asin(float(np.clip(2 * (qw*qy - qz*qx), -1, 1)))
        cr, sr, cp, sp = math.cos(roll/2), math.sin(roll/2), math.cos(pitch/2), math.sin(pitch/2)
        a = self.base_qadr
        self.local.qpos[a:a+7] = [0, 0, self.config["base_height"], cp*cr, cp*sr, sp*cr, -sp*sr]
        mujoco.mj_kinematics(self.model, self.local)
        mujoco.mj_camlight(self.model, self.local)
        frames = []
        h, w = self.config["height"], self.config["width"]
        for name, cid in zip(self.names, self.ids):
            f = h / (2 * math.tan(math.radians(self.model.cam_fovy[cid]) / 2))
            K = np.array([[f, 0, (w-1)/2], [0, f, (h-1)/2], [0, 0, 1]])
            self.renderer.update_scene(data, camera=cid, scene_option=self.options)
            frames.append(CameraFrame(name, self.renderer.render().copy(), K,
                                      self.local.cam_xpos[cid].copy(),
                                      self.local.cam_xmat[cid].reshape(3, 3).copy(), float(data.time)))
        self.frames = frames
        return pose, frames

    def close(self):
        self.renderer.close()
