"""Autonomous training, robot alone (no wheelchair): drive every unordered
room pair at least once, building up the real persistent memory used by
run_world.py (memory/robot_map.npz) rather than a scratch/throwaway file.

Not part of the app - a one-off training driver. Detects genuine stuck-loop
behaviour (confined to a small area for a long stretch, not just a normal
BLOCKED determination) and stops immediately so it can be reported rather
than silently burning through the rest of the legs.
"""
import sys

import mujoco

from camera_rig import CameraRig, load_config
from robot_base import Arms, BalanceBase
from vision_navigation import VisionNavigator

MEMORY = "memory/robot_map.npz"   # the real, persistent plain-robot map
LEGS = [
    ("bedroom", "kitchen"),
    ("kitchen", "bathroom"),
    ("bathroom", "living_room"),
    ("living_room", "bedroom"),
    ("bedroom", "bathroom"),
    ("kitchen", "living_room"),
]
PER_LEG_SECONDS = 320.0


class LoopWatch:
    """Flags "confined to a small area for a long stretch" - covers both a
    tight geometric loop and simple back-and-forth oscillation, which is what
    actually matters here: real forward progress isn't happening, regardless
    of which nav state is nominally active."""
    def __init__(self, window_s=90.0, sample_every=3.0, min_extent=0.8):
        self.window_s, self.sample_every, self.min_extent = window_s, sample_every, min_extent
        self.samples = []
        self.last_sample = -1e9

    def update(self, t, xy):
        if t - self.last_sample < self.sample_every:
            return False
        self.last_sample = t
        self.samples.append((t, xy[0], xy[1]))
        self.samples = [s for s in self.samples if t - s[0] <= self.window_s]
        if len(self.samples) < max(8, int(self.window_s / self.sample_every * 0.6)):
            return False
        xs = [s[1] for s in self.samples]
        ys = [s[2] for s in self.samples]
        return max(max(xs) - min(xs), max(ys) - min(ys)) < self.min_extent


config = load_config()
model = mujoco.MjModel.from_xml_path("world.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
base, arms = BalanceBase(model, data), Arms(model, data)
rig = CameraRig(model, config)

results = []
nav = None
stuck = False
for start, dest in LEGS:
    if nav is None:
        nav = VisionNavigator(config, room=dest, memory_path=MEMORY)
    else:
        nav = VisionNavigator(config, room=dest, memory=nav.memory)
    t_leg_start = data.time
    outcome = "TIMEOUT"
    watch = LoopWatch()
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
            break
        if nav.state == "LOCALIZATION_REQUIRED":
            outcome = "LOCALIZATION_REQUIRED"
            break
        if watch.update(data.time, nav.current_pose[:2] if nav.current_pose is not None else pose[:2]):
            outcome = "STUCK_LOOP"
            print(f"  STUCK_LOOP pose={nav.current_pose} goal={nav.goal} state={nav.state} "
                  f"path_len={len(nav.path)}", flush=True)
            stuck = True
            break
    elapsed = data.time - t_leg_start
    print(f"[{start} -> {dest}] {outcome} after {elapsed:.1f}s (state={nav.state})", flush=True)
    results.append((start, dest, outcome, elapsed))
    nav.memory.save()
    if outcome in ("FELL", "STUCK_LOOP"):
        break

print("\n=== ROBOT-ALONE TRAINING SUMMARY ===")
for start, dest, outcome, elapsed in results:
    print(f"{start:12s} -> {dest:12s} {outcome:10s} {elapsed:6.1f}s")
if stuck:
    print("PAUSED_FOR_STUCK_LOOP", flush=True)

rig.close()
sys.exit(2 if stuck else (1 if any(r[2] not in ("ARRIVED",) for r in results) else 0))
