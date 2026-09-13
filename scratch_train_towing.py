"""Train navigation while towing the wheelchair: attach once, then drive a
sequence of destinations chosen so every one of the 6 unordered room pairs
gets covered at least once (spawn is the bedroom, so the walk below starts
there), reusing the real persistent memory so the learned map/landmarks/room
targets carry over and pick up wherever the existing training left off.
Saves progress after every leg.

Not part of the app - a one-off training driver, mirroring the room-to-room
combination testing used earlier for plain (unattached) navigation.
"""
import os
import time

import mujoco

from camera_rig import CameraRig, load_config
from robot_base import Arms, BalanceBase
from vision_navigation import VisionNavigator
from push_wheelchair import PushController, ATTACH_STATES, HULL_SIDE_Y

class LoopWatch:
    """Flags "confined to a small area for a long stretch" - covers both a
    tight geometric loop and simple back-and-forth oscillation: real forward
    progress isn't happening, regardless of which nav state is nominally
    active."""
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


MEMORY = "memory/robot_map_tow.npz"   # dedicated to towing; never the plain-robot map
# Destination sequence (start is the bedroom spawn). Consecutive pairs:
# bedroom-kitchen, kitchen-bathroom, bathroom-living_room, living_room-bedroom,
# bedroom-bathroom, bathroom-kitchen (repeat, harmless), kitchen-living_room -
# all 6 distinct unordered pairs covered at least once in one continuous walk.
LEGS = ["kitchen", "bathroom", "living_room", "bedroom", "bathroom", "kitchen", "living_room"]
PER_LEG_SECONDS = 240.0

config = load_config()
model = mujoco.MjModel.from_xml_path("world.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
base, arms = BalanceBase(model, data), Arms(model, data)
push = PushController(model, data, base=base, arms=arms, start_state="drive")
rig = CameraRig(model, config)

results = []
nav = None
stuck = False
push.attach_chair()
print(f"[{data.time:.1f}s] attaching the wheelchair before training starts", flush=True)

for dest in LEGS:
    if nav is None:
        nav = VisionNavigator(config, room=dest, memory_path=MEMORY)
    else:
        nav = VisionNavigator(config, room=dest, memory=nav.memory)
    t_leg_start = data.time
    outcome = "TIMEOUT"
    watch = LoopWatch()
    # Budget covers the attach maneuver only on the very first leg.
    budget = PER_LEG_SECONDS + (30.0 if not push.linked else 0.0)
    steps = int(budget / model.opt.timestep)
    for _ in range(steps):
        pose, frames = rig.sample(data)
        front_x = push.tow_exclusion_front_x()
        nav.set_tow_footprint((0.15, front_x, max(HULL_SIDE_Y, 0.35)) if front_x is not None else None)
        if frames is not None:
            nav.observe(frames, pose)
            nav.set_towing(push.linked)
            if not base.fallen:
                if push.state in ATTACH_STATES:
                    push.decide()
                else:
                    base.command(*nav.command(pose, data.time))
        if data.time - nav.last_frame > 0.5:
            base.stop()
        push.step()
        if push.collided:
            nav.notify_collision(data.time)
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
        # Only watch for loops once actually driving - the lineup/reach/clamp
        # maneuver legitimately stays in a small area the whole time.
        if push.state not in ATTACH_STATES and watch.update(
                data.time, nav.current_pose[:2] if nav.current_pose is not None else pose[:2]):
            outcome = "STUCK_LOOP"
            print(f"  STUCK_LOOP pose={nav.current_pose} goal={nav.goal} state={nav.state} "
                  f"path_len={len(nav.path)} linked={push.linked} push.state={push.state}", flush=True)
            stuck = True
            break
    elapsed = data.time - t_leg_start
    print(f"[{dest}] {outcome} after {elapsed:.1f}s (state={nav.state}, "
          f"linked={push.linked}, push.state={push.state})", flush=True)
    results.append((dest, outcome, elapsed))
    nav.memory.save()
    if outcome in ("FELL", "STUCK_LOOP"):
        break

print("\n=== TOWING TRAINING SUMMARY ===")
for dest, outcome, elapsed in results:
    print(f"{dest:12s} {outcome:10s} {elapsed:6.1f}s")
if stuck:
    print("PAUSED_FOR_STUCK_LOOP", flush=True)

rig.close()
