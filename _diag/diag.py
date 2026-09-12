"""Diagnostic for test_reaches_local_goal: what do the cameras and the map see?

Run from the repo root:  python _diag\diag.py
Writes PNGs into _diag\ and prints the planner's decisions.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mujoco
import numpy as np

from build_world import build
from camera_rig import CameraRig, load_config
from run_world import Arms, BalanceBase
from vision_navigation import FloorVision, VisionNavigator, local_to_map

OUT = os.path.join("_diag")


def save_overlay(frames, vision, tag):
    for f in frames:
        rgb = f.rgb.copy()
        mask = vision.masks.get(f.name)
        if mask is not None:
            rgb[mask > 0] = (rgb[mask > 0].astype(float) * 0.6 + np.array([0, 90, 0])).clip(0, 255)
        if f.body is not None:
            rgb[f.body] = (rgb[f.body].astype(float) * 0.6 + np.array([120, 0, 0])).clip(0, 255)
        cv2.imwrite(os.path.join(OUT, f"{tag}_{f.name}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def save_map(nav, pose, goal, trajectory, tag, half=3.0):
    grid = nav.map
    cx, cy = grid.cells(pose[:2])
    n = int(half / grid.resolution)
    ys, xs = slice(cy - n, cy + n), slice(cx - n, cx + n)
    ev, seen = grid.evidence[ys, xs], grid.seen[ys, xs]
    free = grid.traversable()[ys, xs]
    img = np.full(ev.shape + (3,), 110, np.uint8)            # unseen: grey
    img[seen & (ev < 0)] = (60, 140, 60)                     # seen free: green
    img[free] = (90, 220, 90)                                # traversable: light green
    img[seen & (ev >= 0)] = (60, 60, 220)                    # obstacle: red (BGR)
    scale = 12
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    def px(p):
        c = grid.cells(p) - np.array([cx - n, cy - n])
        return int(c[0] * scale + scale / 2), int(c[1] * scale + scale / 2)

    for a, b in zip(nav.path, nav.path[1:]):
        cv2.line(img, px(a), px(b), (255, 200, 0), 2)
    for a, b in zip(trajectory, trajectory[1:]):
        cv2.line(img, px(a), px(b), (255, 255, 255), 1)
    if goal is not None:
        cv2.circle(img, px(goal), 6, (0, 255, 255), 2)
    cv2.circle(img, px(pose[:2]), 5, (255, 255, 255), -1)
    tip = pose[:2] + 0.4 * np.array([math.cos(pose[2]), math.sin(pose[2])])
    cv2.arrowedLine(img, px(pose[:2]), px(tip), (255, 255, 255), 2)
    img = cv2.flip(img, 0)   # +y up
    cv2.imwrite(os.path.join(OUT, f"{tag}_map.png"), img)


def run(model, position, yaw, tag):
    config = load_config()
    data = mujoco.MjData(model)
    data.qpos[:2] = position
    data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(model, data)
    rig = CameraRig(model, config)
    base, arms = BalanceBase(model, data), Arms(model, data)
    nav = VisionNavigator(config, goal=[1, 0], narrate=lambda s: print(f"   [{tag}] {s}"))
    print(f"\n=== {tag}: world pose {position} yaw {math.degrees(yaw):.0f} deg, goal 1 m ahead ===")
    trajectory = []
    first = True
    last_report = -1.0
    try:
        for _ in range(5000):
            pose, frames = rig.sample(data)
            if frames:
                nav.observe(frames, pose)
                if first:
                    first = False
                    save_overlay(frames, nav.vision, f"{tag}_t0")
                    # Per-camera contribution near the robot
                    for f in frames:
                        fr, bl, signs = FloorVision(config).observe([f])
                        near_b = bl[np.linalg.norm(bl, axis=1) < 1.5] if len(bl) else bl
                        print(f"   t0 {f.name:15s} free={len(fr):5d} blocked={len(bl):4d} "
                              f"blocked<1.5m={len(near_b):3d} signs={[(s['room'], s['side']) for s in signs]}")
                        if len(near_b):
                            print(f"      nearest blocked (robot frame x fwd, y left): "
                                  f"{np.round(near_b[np.argsort(np.linalg.norm(near_b, axis=1))[:8]], 2).tolist()}")
                    save_map(nav, pose, nav.goal, trajectory, f"{tag}_t0")
            v, w = nav.command(pose, data.time)
            base.command(v, w)
            if data.time - last_report >= 1.0:
                last_report = data.time
                trajectory.append(pose[:2].copy())
                tgt = nav.path[-1] if nav.path else None
                print(f"   t={data.time:4.1f} state={nav.state:11s} odom=({pose[0]:+.2f},{pose[1]:+.2f},"
                      f"{math.degrees(pose[2]):+4.0f}deg) v={v:+.2f} w={w:+.2f} path={len(nav.path):2d} "
                      f"end={None if tgt is None else np.round(tgt, 2).tolist()} "
                      f"explore={None if nav.explore_target is None else np.round(nav.explore_target, 2).tolist()}")
            arms.step()
            base.step()
            if base.fallen or nav.done:
                break
        print(f"   result: done={nav.done} state={nav.state} fallen={base.fallen} t={data.time:.1f}")
        save_overlay(rig.frames, nav.vision, f"{tag}_end")
        save_map(nav, pose, nav.goal, trajectory, f"{tag}_end")
        # Goal-cell status
        g = nav.map.cells(nav.goal)
        print(f"   goal cell seen={nav.map.seen[g[1], g[0]]} evidence={nav.map.evidence[g[1], g[0]]:.1f} "
              f"traversable={nav.map.traversable()[g[1], g[0]]}")
    finally:
        rig.close()


def main():
    os.makedirs(OUT, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(build())
    run(model, [-3.5, 2.6], -math.pi / 2, "bedroom")
    run(model, [3.5, 2.6], 0.0, "kitchen")
    print(f"\nimages written to {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
