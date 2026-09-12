"""Reproducible physics evaluation. Ground truth is used ONLY to score results.

Example: python verify_navigation.py --room bathroom --seconds 180 --output verification/bathroom
Repeat with the same --memory to test persistence, or change --spawn X Y YAW_DEG.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np

import layout  # evaluation only; never supplied to the navigation stack
from camera_rig import CameraRig, load_config
from run_world import Arms, BalanceBase
from vision_navigation import VisionNavigator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room", default="bathroom", choices=list(layout.ROOMS))
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--output", default="verification/run")
    parser.add_argument("--memory")
    parser.add_argument("--spawn", nargs=3, type=float)
    parser.add_argument("--world", default="world.xml")
    args = parser.parse_args()
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    config = load_config()
    model = mujoco.MjModel.from_xml_path(args.world)
    data = mujoco.MjData(model)
    if args.spawn:
        x, y, yaw = args.spawn
        data.qpos[:2] = [x, y]
        yaw = math.radians(yaw)
        data.qpos[3:7] = [math.cos(yaw/2), 0, 0, math.sin(yaw/2)]
    mujoco.mj_forward(model, data)
    base, arms = BalanceBase(model, data), Arms(model, data)
    rig = CameraRig(model, config)
    nav = VisionNavigator(config, room=args.room, memory_path=args.memory)
    robot_bodies = {base.base_id}
    for bid in range(model.nbody):
        if model.body_parentid[bid] in robot_bodies:
            robot_bodies.add(bid)
    contacts = set()
    trace, messages = [], []
    last_log, last_sample = -100., -100.
    try:
        for _ in range(int(args.seconds/model.opt.timestep)):
            pose, frames = rig.sample(data)
            if frames:
                nav.observe(frames, pose)
                base.command(*nav.command(pose, data.time))
                for event in nav.memory.drain_events():
                    messages.append(f"{data.time:.1f}: {event}")
                if data.time-last_log >= 10:
                    message = f"t={data.time:.1f} {nav.state}: {nav.explain(base.v_cmd, base.w_cmd)}"
                    print(message, flush=True)
                    messages.append(message)
                    last_log = data.time
            arms.step()
            base.step()
            for contact in data.contact[:data.ncon]:
                g1, g2 = int(contact.geom1), int(contact.geom2)
                first, second = model.geom_bodyid[g1] in robot_bodies, model.geom_bodyid[g2] in robot_bodies
                if first != second:
                    other = g2 if first else g1
                    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or str(other)
                    if name != "floor":
                        contacts.add(name)
            if data.time-last_sample >= 1:
                trace.append({"time": float(data.time), "truth": list(base.pose()), "estimate": nav.current_pose.tolist(), "state": nav.state})
                last_sample = data.time
            if base.fallen or nav.done or nav.state in ("BLOCKED", "LOCALIZATION_REQUIRED"):
                break
        x, y, _ = base.pose()
        room = layout.ROOMS[args.room]
        actually_inside = room["x"][0] < x < room["x"][1] and room["y"][0] < y < room["y"][1]
        report = {"room": args.room, "state": nav.state, "arrived": nav.done,
                  "actually_inside_room": actually_inside, "fallen": base.fallen,
                  "seconds": float(data.time), "world_pose": list(base.pose()),
                  "non_floor_contacts": sorted(contacts), "loaded_memory": nav.memory.loaded,
                  "localized": nav.memory.localized, "learned_rooms": nav.memory.rooms,
                  "landmarks": len(nav.memory.landmarks), "observed_cells": int(nav.map.seen.sum()),
                  "passed": bool(nav.done and actually_inside and not base.fallen and not contacts), "trace": trace}
        (directory/"report.json").write_text(json.dumps(report, indent=2))
        (directory/"terminal.txt").write_text("\n".join(messages))
        for frame in rig.frames:
            cv2.imwrite(str(directory/f"{frame.name}.png"), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
        view = np.full((*nav.map.seen.shape, 3), 65, np.uint8)
        view[nav.map.seen] = [170, 170, 170]
        view[nav.map.evidence > 0] = [60, 60, 230]
        view[nav.map.traversable()] = [80, 175, 80]
        for point in nav.path:
            cv2.circle(view, tuple(nav.map.cells(point)), 1, (255, 150, 0), -1)
        for target in nav.map.targets.values():
            cv2.circle(view, tuple(nav.map.cells(target)), 3, (0, 255, 255), -1)
        cv2.circle(view, tuple(nav.map.cells(nav.current_pose[:2])), 3, (255, 255, 255), -1)
        cv2.imwrite(str(directory/"map.png"), cv2.flip(cv2.resize(view, (900, 900), interpolation=cv2.INTER_NEAREST), 0))
        print(json.dumps({key: value for key, value in report.items() if key != "trace"}, indent=2), flush=True)
    finally:
        try:
            nav.memory.save()
        finally:
            rig.close()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
