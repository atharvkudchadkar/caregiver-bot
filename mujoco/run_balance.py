"""Two-wheel balancing and arm-control demo (simulation only).

macOS: mjpython run_balance.py --speed 0 --joint rj1=0.4
List joints: python3 run_balance.py --list-joints
"""
import argparse
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

MODEL = Path(__file__).with_name('balance_robot.xml')


def main():
    parser = argparse.ArgumentParser(description='Run the MuJoCo two-wheel balance demo.')
    parser.add_argument('--speed', type=float, default=0.0, help='Straight-line target speed in m/s; try 0 or 0.1.')
    parser.add_argument('--tilt-deg', type=float, default=0.0, help='Initial forward/backward tilt in degrees; try 3.')
    parser.add_argument('--headless', action='store_true', help='Run without opening the viewer.')
    parser.add_argument('--seconds', type=float, default=10.0, help='Duration in headless mode.')
    parser.add_argument('--list-joints', action='store_true', help='Show controllable arm and gripper joints.')
    parser.add_argument('--joint', action='append', default=[], metavar='NAME=VALUE',
                        help='Set an arm, lift, or gripper target; repeat for multiple joints.')
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    arm_controls = {}
    for actuator_id in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        if not actuator_name.endswith('_target'):
            continue
        joint_name = actuator_name.removesuffix('_target')
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        arm_controls[joint_name] = (actuator_id, joint_id)
    if args.list_joints:
        for name, (_, joint_id) in arm_controls.items():
            lo, hi = model.jnt_range[joint_id]
            unit = 'm' if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_SLIDE else 'rad'
            print(f'{name:24s} {lo:.3f} to {hi:.3f} {unit}')
        return
    requested = {name: 0.0 for name in arm_controls}
    for entry in args.joint:
        try:
            name, value_text = entry.split('=', 1)
            name = name.removesuffix('_target')
            value = float(value_text)
        except ValueError:
            parser.error(f'Invalid --joint {entry!r}; use NAME=VALUE, for example rj1=0.4')
        if name not in arm_controls:
            parser.error(f'Unknown joint {name!r}; run --list-joints to see valid names')
        joint_id = arm_controls[name][1]
        lo, hi = model.jnt_range[joint_id]
        if not math.isfinite(value) or value < lo or value > hi:
            parser.error(f'{name} must be between {lo:.3f} and {hi:.3f}')
        requested[name] = value
    data = mujoco.MjData(model)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'mobile_base')
    motor_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                 for name in ('left_motor', 'right_motor')]
    theta0 = math.radians(args.tilt_deg)
    data.qpos[3:7] = [math.cos(theta0 / 2), 0, math.sin(theta0 / 2), 0]
    target_x = 0.0
    fallen = False

    def step():
        nonlocal target_x, fallen
        mujoco.mj_forward(model, data)
        rotation = data.xmat[base_id].reshape(3, 3)
        pitch = math.atan2(rotation[0, 2], rotation[2, 2])
        if abs(pitch) > math.radians(20) or not np.isfinite(data.qpos).all():
            fallen = True
            data.ctrl[motor_ids] = 0
            return
        for name, (actuator_id, joint_id) in arm_controls.items():
            # Move targets gradually so a new arm command does not jerk the base.
            rate = 0.1 if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_SLIDE else 0.4
            delta = requested[name] - data.ctrl[actuator_id]
            data.ctrl[actuator_id] += float(np.clip(delta, -rate * model.opt.timestep,
                                                     rate * model.opt.timestep))
        target_x += args.speed * model.opt.timestep
        # Tuned for this uncalibrated simulation only. Positive torque drives beneath a forward lean.
        torque = (100 * pitch + 20 * data.qvel[4]
                  + 20 * (data.qpos[0] - target_x)
                  + 15 * (data.qvel[0] - args.speed))
        data.ctrl[motor_ids] = np.clip(torque, -20, 20)
        mujoco.mj_step(model, data)

    if args.headless:
        for _ in range(round(args.seconds / model.opt.timestep)):
            step()
            if fallen:
                break
        print(f'time={data.time:.2f}s x={data.qpos[0]:.3f}m '
              f'pitch={math.degrees(pitch_angle(model, data, base_id)):.2f}deg '
              f'fallen={fallen}')
        for name in sorted(set(entry.split('=', 1)[0].removesuffix('_target') for entry in args.joint)):
            joint_id = arm_controls[name][1]
            print(f'{name}: target={requested[name]:.3f}, '
                  f'actual={data.qpos[model.jnt_qposadr[joint_id]]:.3f}')
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.monotonic()
            if not fallen:
                step()
            viewer.sync()
            if fallen:
                print('Balance limit exceeded. Close the viewer and restart the script.')
                break
            remaining = model.opt.timestep - (time.monotonic() - start)
            if remaining > 0:
                time.sleep(remaining)
        if fallen:
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)


def pitch_angle(model, data, base_id):
    mujoco.mj_forward(model, data)
    rotation = data.xmat[base_id].reshape(3, 3)
    return math.atan2(rotation[0, 2], rotation[2, 2])


if __name__ == '__main__':
    main()
