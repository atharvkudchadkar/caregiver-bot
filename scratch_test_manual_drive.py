"""Check: attach the wheelchair, then simulate holding the UP arrow key
(manual v_cmd, guarded by nav.guard) for a sustained period, confirming it
doesn't roll a bit and stop due to the chair itself being seen as an
obstacle."""
import mujoco
import numpy as np

from camera_rig import CameraRig, load_config
from robot_base import Arms, BalanceBase
from vision_navigation import VisionNavigator
from push_wheelchair import PushController, ATTACH_STATES, HULL_SIDE_Y

config = load_config()
model = mujoco.MjModel.from_xml_path("world.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
base, arms = BalanceBase(model, data), Arms(model, data)
push = PushController(model, data, base=base, arms=arms, start_state="drive")
rig = CameraRig(model, config)
nav = VisionNavigator(config, memory_path="memory/robot_map_tow_manualtest.npz")

push.attach_chair()
manual_drive_start = None
last_print = -1e9
last_pre_print = -1e9
TARGET_CELL = (152, 138)  # known from a prior run's post-link DIAG output
for step in range(int(60 / model.opt.timestep)):
    pose, frames = rig.sample(data)
    front_x = push.tow_exclusion_front_x()
    nav.set_tow_footprint((0.15, front_x, max(HULL_SIDE_Y, 0.35)) if front_x is not None else None)
    if frames is not None:
        nav.observe(frames, pose)
        nav.set_towing(push.linked)
        if data.time - last_pre_print > 1.0:
            last_pre_print = data.time
            tx, ty = TARGET_CELL
            print(f"PRE t={data.time:5.1f} state={push.state:12s} front_x={front_x} "
                  f"footprint={nav.tow_footprint} target_ev={nav.map.evidence[ty,tx]:.2f} "
                  f"target_seen={nav.map.seen[ty,tx]}", flush=True)
        if not base.fallen:
            if push.state in ATTACH_STATES:
                push.decide()
            elif manual_drive_start is not None:
                # Simulate holding "up": v_cmd ramps toward 0.3 via repeated
                # "keypresses" (matching on_key's +0.1 increments), guarded.
                requested = min(base.v_cmd + 0.1, 0.3)
                gv, gw = nav.guard(pose, data.time, requested, 0.0)
                if data.time - manual_drive_start < 6.0:
                    import math as _m
                    mp = nav.memory.pose(pose)
                    dist = _m.copysign(0.35 + abs(requested)*0.5, requested)
                    end = mp[:2] + dist * np.array([_m.cos(mp[2]), _m.sin(mp[2])])
                    seg_ok = nav.map.safe_segment(mp[:2], end, data.time, strict=False)
                    cx, cy = nav.map.cells(mp[:2])
                    ex, ey = nav.map.cells(end)
                    print(f"    DIAG requested={requested:.2f} mp={mp} end={end} "
                          f"seg_ok={seg_ok} guard_v={gv} start_cell=({cx},{cy}) end_cell=({ex},{ey}) "
                          f"start_seen={nav.map.seen[cy,cx]} start_ev={nav.map.evidence[cy,cx]:.2f} "
                          f"end_seen={nav.map.seen[ey,ex]} end_ev={nav.map.evidence[ey,ex]:.2f}",
                          flush=True)
                base.command(gv, gw)
    if data.time - nav.last_frame > 0.5:
        base.stop()
    push.step()

    if push.linked and manual_drive_start is None:
        manual_drive_start = data.time
        print(f"t={data.time:.1f}s: linked, starting simulated manual drive", flush=True)

    if manual_drive_start is not None and data.time - last_print > 2.0:
        last_print = data.time
        rx, ry, ryaw = base.pose()
        print(f"  t={data.time - manual_drive_start:5.1f} v_cmd={base.v_cmd:+.2f} "
              f"last_frame_age={data.time - nav.last_frame:.2f} localized={nav.memory.localized} "
              f"tow_footprint={nav.tow_footprint} "
              f"robot=({rx:+.2f},{ry:+.2f},{ryaw:+.2f})", flush=True)

    if base.fallen:
        print("FELL"); break

rig.close()
