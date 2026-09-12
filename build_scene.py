"""Step 2: wrap the converted robot in a scene with a drivable base and rooms.

Run:   python build_scene.py          -> writes scene.xml and two test renders
View:  python -m mujoco.viewer --mjcf scene.xml

The URDF has no wheel joints and placeholder masses, so the robot is driven
KINEMATICALLY: we add planar joints (base_x, base_y, base_yaw) to the root body,
write joint positions directly, and call mj_forward. No dynamics, no balancing.
"""
import math
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Room targets for Stage 1 voice navigation: name -> (x, y), colour
ROOMS = {
    "kitchen": ([3.0, 0.0], [0.9, 0.3, 0.2]),
    "bedroom": ([0.0, 3.0], [0.2, 0.4, 0.9]),
    "living_room": ([-3.0, 0.0], [0.2, 0.8, 0.3]),
}


def build_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(os.path.join(HERE, "robot_converted.xml"))
    spec.modelname = "bracketbot_scene"
    spec.meshdir = "robot/meshes"
    spec.visual.global_.offwidth = 1280   # offscreen render size for camera images
    spec.visual.global_.offheight = 720

    wb = spec.worldbody
    wb.add_light(pos=[0, 0, 4], dir=[0, 0, -1], castshadow=False)
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=[10, 10, 0.1], rgba=[0.85, 0.85, 0.85, 1])
    for name, (xy, rgb) in ROOMS.items():
        wb.add_site(name=name, pos=[xy[0], xy[1], 0.01], size=[0.35, 0.005, 0],
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER, rgba=rgb + [0.6])
    wb.add_geom(name="wall", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[1.5, 1.2, 0.5],
                size=[0.05, 1.2, 0.5], rgba=[0.6, 0.6, 0.65, 1])
    wb.add_camera(name="overview", pos=[-2.5, -5.0, 4.0],
                  xyaxes=[1, -0.45, 0, 0.3, 0.6, 0.75])

    # Mobile base: planar joints on the URDF's root link (wheel-axle midpoint).
    root = spec.body("root")
    root.add_joint(name="base_x", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0])
    root.add_joint(name="base_y", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0])
    root.add_joint(name="base_yaw", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1])

    # Robot meshes are visual only (we never integrate contact forces).
    for g in spec.geoms:
        if g.name not in ("floor", "wall"):
            g.contype = 0
            g.conaffinity = 0

    add_joint_control(spec)
    return spec


BASE_JOINTS = ("base_x", "base_y", "base_yaw")
# Gripper: the URDF marks the second finger as a mimic of the first; MuJoCo
# ignores <mimic>, so we recreate it as an equality constraint.
GRIPPER_PAIRS = (("right_left_gripper", "right_right_gripper"),
                 ("left_left_gripper", "left_right_gripper"))


def add_joint_control(spec: mujoco.MjSpec, kp=30.0, kv=3.0):
    """Make the model stable under mj_step and give every joint a position servo.

    The URDF's inertials are placeholders (links of ~0.1 g), so without rotor
    inertia and damping the integrator explodes. Armature + damping fix that;
    position actuators let the viewer's Control sliders hold a pose.
    """
    followers = {b for _, b in GRIPPER_PAIRS}

    def set_dof(joint, attr, value):
        # Newer MuJoCo exposes joint damping/armature as 3-element arrays
        # (one per possible DOF); older versions use a scalar.
        try:
            setattr(joint, attr, np.full(3, float(value)))
        except TypeError:
            setattr(joint, attr, float(value))

    for j in spec.joints:
        if j.name in BASE_JOINTS:
            set_dof(j, "damping", 5.0)
            set_dof(j, "armature", 0.5)
            continue
        set_dof(j, "armature", 0.02)
        set_dof(j, "damping", 0.5)
        if j.name in followers:
            continue  # driven by the equality constraint below
        act = spec.add_actuator()
        act.name = f"act_{j.name}"
        act.target = j.name
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.gainprm[0] = kp
        act.biasprm[1] = -kp
        act.biasprm[2] = -kv
        act.ctrlrange[:] = j.range
        act.ctrllimited = True

    for leader, follower in GRIPPER_PAIRS:
        eq = spec.add_equality()
        eq.name = f"mimic_{follower}"
        eq.type = mujoco.mjtEq.mjEQ_JOINT
        eq.name1 = follower
        eq.name2 = leader
        eq.data[:5] = [0.0, 1.0, 0.0, 0.0, 0.0]  # follower = 0 + 1 * leader


def joint_qpos_index(model, name):
    return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]


def step_toward(pose, target, dt, v_max=0.5, w_max=1.0):
    """Unicycle P-controller. Returns (v, w, arrived). Reused by nav.py later."""
    x, y, yaw = pose
    dx, dy = target[0] - x, target[1] - y
    dist = math.hypot(dx, dy)
    if dist < 0.05:
        return 0.0, 0.0, True
    err = (math.atan2(dy, dx) - yaw + math.pi) % (2 * math.pi) - math.pi
    v = min(v_max, 0.8 * dist) if abs(err) < 0.5 else 0.0  # turn first, then drive
    w = max(-w_max, min(w_max, 2.0 * err))
    return v, w, False


def main():
    spec = build_spec()
    model = spec.compile()
    open(os.path.join(HERE, "scene.xml"), "w", encoding="utf-8").write(spec.to_xml())
    print("wrote scene.xml")

    data = mujoco.MjData(model)
    bx, by, byaw = (joint_qpos_index(model, n) for n in ("base_x", "base_y", "base_yaw"))

    # Smoke test: drive kinematically to the kitchen marker.
    target = ROOMS["kitchen"][0]
    dt = 0.02
    for k in range(1500):
        pose = (data.qpos[bx], data.qpos[by], data.qpos[byaw])
        v, w, arrived = step_toward(pose, target, dt)
        if arrived:
            break
        data.qpos[bx] += v * math.cos(pose[2]) * dt
        data.qpos[by] += v * math.sin(pose[2]) * dt
        data.qpos[byaw] += w * dt
        mujoco.mj_forward(model, data)
    print(f"arrived at kitchen after {k * dt:.1f}s sim time; "
          f"pose=({data.qpos[bx]:.2f}, {data.qpos[by]:.2f}, {math.degrees(data.qpos[byaw]):.0f} deg)")

    # Pose the arms/carriage a little so the render shows articulation.
    for name, val in {"rj2": 0.8, "rj3": -1.0, "lj2": -0.8, "lj3": 1.0, "rj0": -0.3, "lj0": -0.3}.items():
        data.qpos[joint_qpos_index(model, name)] = val
    mujoco.mj_forward(model, data)

    try:
        import PIL.Image
        renderer = mujoco.Renderer(model, height=540, width=960)
        renderer.update_scene(data, camera="overview")
        PIL.Image.fromarray(renderer.render()).save(os.path.join(HERE, "scene.png"))
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [data.qpos[bx], data.qpos[by], 0.9]
        cam.distance, cam.azimuth, cam.elevation = 3.2, 210, -15
        renderer.update_scene(data, camera=cam)
        PIL.Image.fromarray(renderer.render()).save(os.path.join(HERE, "closeup.png"))
        print("rendered scene.png and closeup.png")
    except Exception as e:  # rendering is optional
        print("render skipped:", e)


if __name__ == "__main__":
    main()
