"""Run the caregiver world: balancing robot that accepts (v, w) drive commands.

  python run_world.py                          viewer; arrow keys drive, space stops
  python run_world.py --goto kitchen           explore for the kitchen label
  python run_world.py --goto bathroom --headless --seconds 90
  python run_world.py --joint rj1=0.4 --joint rj0=-0.3 --joint gripper_right=0.8
  python run_world.py --real-time              watch the viewer at 1x wall-clock speed
  python run_world.py --speed 4                run at a fixed 4x wall-clock speed
  python run_world.py --speed-slider           drag a live slider (0.1x-20x, or uncapped)
  python run_world.py --voice --voice-device 1 listen for spoken commands
  python run_world.py --voice --voice-device 1 --voice-debug   tune mic detection live

The viewer runs as fast as the machine can simulate/render by default (headless
mode always did). This only changes how quickly wall-clock time passes on
screen; the robot's commanded speed, torque limits and control gains are all
defined in simulated seconds and are untouched regardless of the multiplier.
--real-time caps the viewer to 1x, --speed N to a fixed multiplier, and
--speed-slider opens a small window with a live slider (see speed_control.py)
so you can drag the speed up or down, or check "uncapped", while it's running.

The balance law is the teammate's (run_balance.py), rewritten in the robot's
body frame so it works at any heading and adds differential torque for
steering. Everything upstream (voice -> intent -> nav) only ever calls
BalanceBase.command(v, w); the real Bracket Bot gets the same call.

Navigation consumes three RGB cameras, wheel encoders and IMU tilt. It builds
an observed free-space map and plans online, with no apartment coordinates or
simulator base pose passed to the navigator. Room names are learned from English wall signs; codes anchor persistent
localization memory, and unseen destinations require exploration.

Arrow keys (viewer window must have focus):
  up/down     +/- 0.1 m/s forward speed      left/right   +/- 0.3 rad/s turn rate
  space       stop                            r            reset to spawn pose
  g           line up on the wheelchair and clamp its handles (known pose, not vision)
  x           release the clamp

While G is lining up: arrows nudge the approach manually, Space resumes
automatic alignment. Once clamped, the chair drives as one body with the
robot - navigation (autonomous or teleop) drives both until X releases it.

Voice commands (--voice), seven in total, spoken through the chosen
microphone and transcribed offline (see voice_control.py):
  "go to <room>"            navigate to a learned room, exploring if needed
  "explore"                  autonomously explore visible free space
  "come home" / "go home"    return to the startup pose
  "stop"                      immediately stop and drop to manual control
  "reset"                     reset to the spawn pose
  "grab/attach the wheelchair"     line up and clamp the handles
  "let go/detach the wheelchair"   release the clamp
"""
import argparse
import math
import time

import mujoco
import mujoco.viewer
import numpy as np

from camera_rig import CameraRig, load_config
from robot_base import Arms, BalanceBase, wrap
from vision_navigation import VisionNavigator
from voice_control import VoiceListener
from push_wheelchair import PushController, ATTACH_STATES, LINEUP_STATES, HULL_SIDE_Y

WORLD = "world.xml"
GLFW_KEYS = {"up": 265, "down": 264, "left": 263, "right": 262, "space": 32, "r": 82,
             "g": 71, "x": 88}


def reset(model, data, base):
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    base.fallen = False
    base.stop()
    base.reset_targets()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", default=WORLD)
    goals = ap.add_mutually_exclusive_group()
    goals.add_argument("--goto", metavar="ROOM", help="navigate to a learned room, exploring if needed")
    goals.add_argument("--goal", nargs=2, type=float, metavar=("X", "Y"),
                       help="goal in metres relative to startup pose, x forward, y left")
    goals.add_argument("--explore", action="store_true", help="explore visible free space")
    ap.add_argument("--vision-config", help="camera and perception calibration JSON")
    ap.add_argument("--camera-preview", action="store_true", help="show all three RGB feeds and floor masks")
    ap.add_argument("--memory", default="memory/robot_map.npz", help="learned map file, loaded and saved automatically")
    ap.add_argument("--tow-memory", default="memory/robot_map_tow.npz",
                    help="separate learned map used whenever the wheelchair is attached - towing needs "
                         "a wider clearance margin, so routes learned for it are kept apart from the "
                         "plain-robot map instead of contaminating (or being constrained by) each other")
    ap.add_argument("--no-memory", action="store_true", help="run without loading or saving a map")
    ap.add_argument("--attach", action="store_true",
                    help="attach the wheelchair before starting (equivalent to pressing G at t=0); "
                         "useful for training/testing navigation while towing, headless or not")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=60.0, help="headless run time")
    speed = ap.add_mutually_exclusive_group()
    speed.add_argument("--real-time", action="store_true", help="cap the viewer to 1x wall-clock speed")
    speed.add_argument("--speed", type=float, metavar="X",
                       help="fixed wall-clock speed multiplier, e.g. 4 for 4x (default: uncapped)")
    speed.add_argument("--speed-slider", action="store_true",
                       help="open a window with a live speed slider (0.1x-20x, or an "
                            "'uncapped' checkbox) so you can change it while running")
    ap.add_argument("--voice", action="store_true",
                    help="listen for spoken commands: 'go to <room>', 'explore', "
                         "'come home', 'stop', 'reset', 'grab the wheelchair', "
                         "'let go of the wheelchair'")
    ap.add_argument("--voice-device", help="microphone device index or name substring "
                     "for --voice (default: system default mic)")
    ap.add_argument("--voice-model", default="small",
                    help="faster-whisper model size for --voice (default: small)")
    ap.add_argument("--voice-threshold", type=float,
                    help="manual microphone noise gate for --voice (default: auto-calibrated "
                         "from 1s of room noise at startup)")
    ap.add_argument("--voice-debug", action="store_true",
                    help="print a live mic level meter for --voice, to help tune detection")
    ap.add_argument("--joint", action="append", default=[], metavar="NAME=VALUE",
                    help="arm/lift/gripper target, repeatable")
    args = ap.parse_args()
    config = load_config(args.vision_config)
    room = args.goto.lower().strip().replace(" ", "_") if args.goto else None
    if room and room not in config["room_names"]:
        ap.error(f"unknown room {room!r}; labels: {config['room_names']}")
    if args.goal and not np.isfinite(args.goal).all():
        ap.error("goal coordinates must be finite")
    if args.speed is not None and args.speed <= 0:
        ap.error("--speed must be a positive number")
    model = mujoco.MjModel.from_xml_path(args.world)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    base, arms = BalanceBase(model, data), Arms(model, data)
    # Shares base/arms with navigation rather than owning its own, so the two
    # never fight over the same actuators; starts idle ("drive", unattached)
    # rather than the standalone demo's auto-attach-on-launch.
    push = PushController(model, data, base=base, arms=arms, start_state="drive")
    if args.attach:
        push.attach_chair()
    for entry in args.joint:
        name, _, value = entry.partition("=")
        arms.set(name.strip(), float(value))
    rig = CameraRig(model, config)
    memory_path = None if args.no_memory else args.memory
    tow_memory_path = None if args.no_memory else args.tow_memory
    try:
        nav = VisionNavigator(config, room=room, goal=args.goal, memory_path=memory_path)
    except ValueError as error:
        rig.close()
        ap.error(str(error))
    towing_nav_active = False   # which memory `nav` is currently pointed at
    autonomous = bool(room or args.goal is not None or args.explore)
    state = None
    last_narration = last_save = -100.0
    print("RGB navigation: head + left hand + right hand; local wheel odometry; no preset routes")
    print("Wheelchair: press G to line up and clamp the handles, X to release "
          "(or say 'grab the wheelchair' / 'let go of the wheelchair' with --voice)")

    voice_device = args.voice_device
    if voice_device is not None:
        try:
            voice_device = int(voice_device)
        except ValueError:
            pass
    voice = VoiceListener(config["room_names"], device=voice_device, model_size=args.voice_model,
                          energy_threshold=args.voice_threshold,
                          debug=args.voice_debug) if args.voice else None

    def do_reset():
        nonlocal nav, push, towing_nav_active, last_narration, last_save
        nav.memory.save()
        reset(model, data, base)
        rig.reset()
        nav = VisionNavigator(config, memory_path=memory_path)
        towing_nav_active = False
        # A physical reset snaps the chair back to its spawn pose too (part of
        # mj_resetData above); the grasp/hitch state has to restart to match.
        push = PushController(model, data, base=base, arms=arms, start_state="drive")
        last_narration = last_save = -100.0

    def sync_tow_memory(pose):
        """Swap `nav` onto the towing-specific memory the instant the chair
        links, and back onto the plain memory the instant it releases -
        towing needs a wider clearance margin (see set_towing), so routes
        learned for it are kept in their own persisted map rather than
        contaminating, or being constrained by, the plain-robot one.

        The robot hasn't physically moved, only which map/landmark database
        it should consult - so seed the new memory's alignment from the pose
        it already knows (current `pose`, in the *old* memory's frame) rather
        than forcing a blind re-localization turn, which would be risky this
        close to whatever it was just doing (e.g. right after clamping the
        handles, still near the wheelchair)."""
        nonlocal nav, towing_nav_active
        if push.linked == towing_nav_active:
            return
        known_pose = nav.current_pose
        nav.memory.save()
        towing_nav_active = push.linked
        path = tow_memory_path if towing_nav_active else memory_path
        nav = VisionNavigator(config, room=nav.room, goal=nav.local_goal, memory_path=path)
        nav.memory.seed_pose(known_pose, pose)
        which = "towing" if towing_nav_active else "plain"
        print(f"t={data.time:.1f}s Switched to the {which} memory ({path}); "
              f"{nav.map.seen.sum()} cells known there so far.", flush=True)

    def handle_voice_command(cmd):
        nonlocal autonomous, nav
        print(f"t={data.time:.1f}s [voice] heard {cmd.text!r} -> {cmd.kind}"
              + (f" {cmd.room}" if cmd.room else ""), flush=True)
        if cmd.kind == "stop":
            autonomous = False
            base.stop()
        elif cmd.kind == "reset":
            do_reset()
        elif cmd.kind == "explore":
            autonomous = True
            nav = VisionNavigator(config, memory=nav.memory)
        elif cmd.kind == "home":
            autonomous = True
            nav = VisionNavigator(config, goal=[0.0, 0.0], memory=nav.memory)
        elif cmd.kind == "goto":
            autonomous = True
            nav = VisionNavigator(config, room=cmd.room, memory=nav.memory)
        elif cmd.kind == "attach":
            push.attach_chair()
        elif cmd.kind == "detach":
            push.detach_chair()

    def control_tick():
        nonlocal state, last_narration, last_save
        if voice is not None:
            cmd = voice.poll()
            if cmd:
                handle_voice_command(cmd)
        pose, frames = rig.sample(data)
        sync_tow_memory(pose)
        # (rear, front, side) in the robot's own frame: rear starts just past
        # its own nose (real, still avoided), front reaches the chair's hull
        # edge - covers the arms (extended forward reaching for/gripping the
        # handles, from arm_pregrasp through drive-while-linked) as well as
        # the chair itself, both otherwise misread as a wall by the forward
        # camera (see set_tow_footprint / PushController.tow_exclusion_front_x).
        front_x = push.tow_exclusion_front_x()
        nav.set_tow_footprint((0.15, front_x, max(HULL_SIDE_Y, 0.35)) if front_x is not None else None)
        if frames is not None:
            nav.observe(frames, pose)
            nav.set_towing(push.linked)
            if not base.fallen:
                if push.state in ATTACH_STATES:
                    push.decide()   # lining up / reaching / clamping owns the base+arms
                elif autonomous:
                    base.command(*nav.command(pose, data.time))
                else:
                    base.command(*nav.guard(pose, data.time, base.v_cmd, base.w_cmd))
            for event in nav.memory.drain_events():
                print(f"t={data.time:.1f}s {event}", flush=True)
            shown = push.state if push.state in ATTACH_STATES else nav.state
            if shown != state or data.time-last_narration >= 5:
                state = shown
                last_narration = data.time
                if push.state in ATTACH_STATES:
                    description = f"Wheelchair: {push.state}."
                elif autonomous:
                    description = nav.explain(base.v_cmd, base.w_cmd)
                else:
                    description = "Manual control: learning the visible layout and checking commanded clearance."
                towing = " Towing the wheelchair." if push.linked else ""
                print(f"t={data.time:.1f}s [{state}] {description}{towing} "
                      f"Mapped {nav.map.seen.sum()*config['resolution']**2:.1f} square metres.", flush=True)
            if data.time-last_save >= 15:
                if nav.memory.save():
                    print(f"t={data.time:.1f}s Saved learned map to {nav.memory.path}.", flush=True)
                last_save = data.time
            if args.camera_preview:
                import cv2
                previews = []
                for frame in frames:
                    rgb = frame.rgb.copy()
                    mask = nav.vision.masks.get(frame.name, np.zeros(rgb.shape[:2], np.uint8))
                    rgb[mask > 0] = (rgb[mask > 0].astype(float)*0.65 + np.array([0, 90, 0])).clip(0, 255)
                    cv2.putText(rgb, frame.name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    previews.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cv2.imshow("Robot RGB cameras - green: estimated floor", np.hstack(previews))
                cv2.waitKey(1)
        if data.time - nav.last_frame > 0.5:
            base.stop()
        # The standalone grasp demo and the main app share exactly the same
        # physics, attachment, collision and transform update sequence.
        push.step()
        if push.collided:
            # The chair just actually hit something (push.step()'s
            # block_wall_clip rolled the pose back) - tell nav so a blind
            # in-place SCANNING turn tries the other direction next tick
            # instead of grinding against the same spot again.
            nav.notify_collision(data.time)

    def on_key(keycode):
        nonlocal autonomous, nav, last_narration, last_save
        if keycode == GLFW_KEYS["g"]:
            push.attach_chair()
            return
        if keycode == GLFW_KEYS["x"]:
            push.detach_chair()
            return
        if push.state in LINEUP_STATES:
            if keycode == GLFW_KEYS["space"]:
                push.accept_lineup()
                return
            if keycode in (GLFW_KEYS[k] for k in ("up", "down", "left", "right")):
                push.drive_lineup(keycode)
            return
        if push.state in ATTACH_STATES:
            return   # mid-reach/clamp; arrows/space don't apply here
        if keycode in (GLFW_KEYS[k] for k in ("up", "down", "left", "right", "space", "r")):
            autonomous = False
        if keycode == GLFW_KEYS["up"]:
            base.command(base.v_cmd + 0.1, base.w_cmd)
        elif keycode == GLFW_KEYS["down"]:
            base.command(base.v_cmd - 0.1, base.w_cmd)
        elif keycode == GLFW_KEYS["left"]:
            base.command(base.v_cmd, base.w_cmd + 0.3)
        elif keycode == GLFW_KEYS["right"]:
            base.command(base.v_cmd, base.w_cmd - 0.3)
        elif keycode == GLFW_KEYS["space"]:
            base.stop()
        elif keycode == GLFW_KEYS["r"]:
            do_reset()
        else:
            return
        print(f"cmd v={base.v_cmd:+.1f} m/s  w={base.w_cmd:+.1f} rad/s")

    if voice is not None:
        voice.start()
    slider = None
    try:
        if args.headless:
            for _ in range(int(args.seconds / model.opt.timestep)):
                control_tick()
                if base.fallen or (autonomous and (nav.done or nav.state in ("BLOCKED", "LOCALIZATION_REQUIRED"))):
                    break
            x, y, yaw = rig.odom.pose
            print(f"t={data.time:.1f}s odom=({x:.2f}, {y:.2f}, {math.degrees(yaw):.0f} deg) "
                  f"pitch={math.degrees(base.pitch()):.1f} deg fallen={base.fallen} "
                  f"navigation={nav.state}")
            return
        steps_per_frame = 10
        if args.speed_slider:
            from speed_control import SpeedSlider
            slider = SpeedSlider(initial=1.0)
            print("Speed slider window opened: drag it, or check 'Uncapped', while the sim runs.")
        fixed_speed = 1.0 if args.real_time else args.speed  # None means uncapped
        with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = base.base_id
            viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 4.5, -25, 135
            announced = False
            while viewer.is_running():
                t0 = time.monotonic()
                for _ in range(steps_per_frame):
                    control_tick()
                viewer.sync()
                if base.fallen and not announced:
                    print("fell over: press r to reset")
                    announced = True
                if slider is not None:
                    slider.pump()
                    multiplier = None if slider.closed else slider.multiplier
                else:
                    multiplier = fixed_speed
                if multiplier is not None:
                    remaining = steps_per_frame * model.opt.timestep / multiplier - (time.monotonic() - t0)
                    if remaining > 0:
                        time.sleep(remaining)
    except KeyboardInterrupt:
        base.stop()
        print("Interrupted; stopping and saving learned memory.", flush=True)
    finally:
        try:
            if nav.memory.save():
                print(f"Saved {nav.map.seen.sum()} learned cells, {len(nav.memory.landmarks)} localization landmarks "
                      f"and {len(nav.memory.rooms)} rooms to {nav.memory.path}.", flush=True)
        finally:
            rig.close()
            if voice is not None:
                voice.stop()
            if slider is not None and not slider.closed:
                slider.close()
            if args.camera_preview:
                import cv2
                cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
