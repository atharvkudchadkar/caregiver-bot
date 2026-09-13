"""One-off test: drive between every pair of rooms in one continuous run,
reusing the same memory across legs, to check the fix holds for genuine
room-to-room navigation (not just spawn->room)."""
import sys
import time

import mujoco

from camera_rig import CameraRig, load_config
from run_world import Arms, BalanceBase
from vision_navigation import VisionNavigator

MEMORY = "memory/combo_test.npz"
LEGS = [
    ("bedroom", "kitchen"),
    ("kitchen", "bathroom"),
    ("bathroom", "living_room"),
    ("living_room", "bedroom"),
    ("bedroom", "bathroom"),
    ("kitchen", "living_room"),
]
PER_LEG_SECONDS = 320.0

config = load_config()
model = mujoco.MjModel.from_xml_path("world.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
base, arms = BalanceBase(model, data), Arms(model, data)
rig = CameraRig(model, config)

import os
if os.path.exists(MEMORY):
    os.remove(MEMORY)

results = []
nav = None
for start, dest in LEGS:
    # Mirrors the fixed production pattern (voice "go to <room>" in
    # run_world.py): reuse the already-localized memory across legs within
    # this one continuous run, rather than reloading from disk and forcing a
    # fresh relocalization that can fail if no landmark is currently in view.
    if nav is None:
        nav = VisionNavigator(config, room=dest, memory_path=MEMORY)
    else:
        nav = VisionNavigator(config, room=dest, memory=nav.memory)
    t_leg_start = data.time
    outcome = "TIMEOUT"
    steps = int(PER_LEG_SECONDS / model.opt.timestep)
    for _ in range(steps):
        pose, frames = rig.sample(data)
        if frames is not None:
            nav.observe(frames, pose)
            if not base.fallen:
                base.command(*nav.command(pose, data.time))
        if data.time - nav.last_frame > 0.5:
            base.stop()
        arms.step()
        base.step()
        if base.fallen:
            outcome = "FELL"
            break
        if nav.done:
            outcome = "ARRIVED"
            break
        if nav.state == "BLOCKED":
            outcome = "BLOCKED"
            import numpy as np
            m = nav.map
            free = m.traversable()
            cx, cy = m.cells(nav.current_pose[:2])
            print(f"  DIAG pose={nav.current_pose} goal={nav.goal} "
                  f"room_visited={nav.memory.rooms.get(dest,{}).get('visited')} "
                  f"target={m.targets.get(dest)}")
            print(f"  DIAG own_cell_free={bool(free[cy,cx])} "
                  f"path_len={len(nav.path)} explore_target={nav.explore_target} "
                  f"memory_hint={nav.memory_hint}")
            direct = m.route(nav.current_pose[:2], nav.goal)
            print(f"  DIAG direct_route_len={len(direct)}")
            break
        if nav.state == "LOCALIZATION_REQUIRED":
            outcome = "LOCALIZATION_REQUIRED"
            break
    elapsed = data.time - t_leg_start
    print(f"[{start} -> {dest}] {outcome} after {elapsed:.1f}s (state={nav.state})", flush=True)
    results.append((start, dest, outcome, elapsed))
    nav.memory.save()

print("\n=== SUMMARY ===")
for start, dest, outcome, elapsed in results:
    print(f"{start:12s} -> {dest:12s} {outcome:10s} {elapsed:6.1f}s")

rig.close()
failed = [r for r in results if r[2] not in ("ARRIVED",)]
sys.exit(1 if failed else 0)
