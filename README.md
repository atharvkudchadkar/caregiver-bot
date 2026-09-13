# Bracket Bot: camera navigation with learned memory

A balancing two-wheel robot that navigates an apartment using only onboard RGB
cameras, wheel encoders and IMU tilt, plus a known-pose wheelchair grasp/tow
controller. There are two independent ways to run it: the main `run_world.py`
demo (autonomous navigation, voice control, manual driving) and a standalone
Meta Quest VR teleoperation mode. They share the same `world.xml` but run as
separate processes and never communicate with each other.

```powershell
python -m pip install -r requirements.txt
python build_world.py
```

Run that once (and again after any change to `layout.py` or `vision_config.json`)
before either mode below. Requires Python 3.12 or 3.13.

## 1. `run_world.py` demo

This is the main simulator: camera-based autonomous navigation, learned
persistent memory, optional voice control, and manual/teleop driving,
including towing the wheelchair.

```powershell
python run_world.py                            # viewer; arrow keys drive, space stops
python run_world.py --explore --camera-preview # autonomously explore and map
python run_world.py --goto kitchen             # navigate to a learned room
```

### Navigation quality: this needs training first

The robot has **no preloaded floor plan**. On an empty memory file it only
knows the small footprint disk under its own wheels - everything else starts
unseen, and unseen space is never treated as safe to drive through. To get
navigation that looks as smooth as a demo video, the map needs to already be
populated, in one of two ways:

- **Let it train itself.** Run with `--explore` (or `--goto <room>`) repeatedly
  so it wanders, scans for room signs, and builds up the occupancy grid and
  learned routes over time. This takes real wall-clock/simulated time - the
  first run through a new room is always the slowest and most hesitant one,
  since it's making the map as it goes rather than following it.
- **Drive an expert pass yourself.** Take manual control (arrow keys) and
  drive through the apartment once. `nav.observe()` runs identically no
  matter who's driving, so a human-driven pass populates the exact same
  occupancy grid, landmarks and room targets that autonomous exploration
  would - just faster and without any risk of it getting stuck on a route a
  person could see is fine. Autonomous runs afterward reuse that memory.

Either way, once a room/route has been driven at least once and saved to
memory, later runs navigate it far more directly - only genuinely new areas
still require the slower explore-and-learn behavior.

### Extra features (`run_world.py` flags)

Destination (pick at most one; default is manual control only):

| Flag | Effect |
| --- | --- |
| `--goto ROOM` | Navigate to a learned room by name, exploring first if it hasn't been visited yet |
| `--goal X Y` | Navigate to a fixed point, in metres, relative to the startup pose (X forward, Y left) |
| `--explore` | Autonomously explore visible free space with no fixed destination |

Voice control:

| Flag | Effect |
| --- | --- |
| `--voice` | Listen on the microphone for spoken commands (see below) |
| `--voice-device` | Microphone index or name substring (default: system default mic) |
| `--voice-model` | faster-whisper model size (default: `small`) |
| `--voice-threshold` | Manual noise-gate level (default: auto-calibrated from 1s of room noise at startup) |
| `--voice-debug` | Print a live mic level meter, to help tune detection |

Spoken commands, once `--voice` is on: *"go to \<room\>"*, *"explore"*,
*"come home"* / *"go home"*, *"stop"*, *"reset"*, *"grab/attach the
wheelchair"*, *"let go/detach the wheelchair"*.

Wheelchair towing:

| Flag | Effect |
| --- | --- |
| `--attach` | Attach the wheelchair before starting, equivalent to pressing **G** at t=0 |
| `--tow-memory PATH` | Separate learned map used only while towing (default `memory/robot_map_tow.npz`) - towing needs wider clearance, so its routes are kept apart from the plain-robot map |

Memory:

| Flag | Effect |
| --- | --- |
| `--memory PATH` | Learned map file, loaded and saved automatically (default `memory/robot_map.npz`) |
| `--no-memory` | Run without loading or saving a map at all |

Playback/viewer speed:

| Flag | Effect |
| --- | --- |
| `--real-time` | Cap the viewer to 1x wall-clock speed |
| `--speed X` | Fixed wall-clock speed multiplier, e.g. `--speed 4` (default: uncapped) |
| `--speed-slider` | Open a window with a live slider (0.1x-20x, or an "uncapped" checkbox) to change speed while it's running |

Other:

| Flag | Effect |
| --- | --- |
| `--camera-preview` | Show all three RGB feeds and the floor classifier mask |
| `--headless` / `--seconds N` | Run without a viewer for N wall-clock seconds (still needs an OpenGL context to render) |
| `--joint NAME=VALUE` | Set an arm/lift/gripper target directly, repeatable |
| `--vision-config PATH` | Alternate camera/perception calibration JSON |
| `--world PATH` | Alternate world XML (default `world.xml`) |

```powershell
python run_world.py --goto kitchen --headless --seconds 300
python run_world.py --goal 1 0 --headless --seconds 15 --no-memory
python run_world.py --joint rj1=0.4 --joint lj6=-0.3 --joint gripper_right=0.8
python run_world.py --voice --voice-device 1 --voice-debug
python run_world.py --attach --speed-slider
```

### Manual controls (viewer window must have focus)

| Key | Action |
| --- | --- |
| Up / Down | +/- 0.1 m/s forward speed |
| Left / Right | +/- 0.3 rad/s turn rate |
| Space | Stop |
| R | Reset to spawn pose (keeps learned memory) |
| G | Line up on the wheelchair and clamp its handles (known pose, not vision) |
| X | Release the wheelchair clamp |

Pressing any arrow key takes over from autonomous navigation immediately.
While **G** is lining up, arrows nudge the approach manually and Space resumes
automatic alignment. Once clamped, the chair drives as one body with the
robot until **X** releases it. After a reset/restart, the robot must
relocalize (see a known landmark) before it will trust the saved map enough
to translate on it - manual rotation to find a sign always still works.

### Memory, room signs and other details

- The default memory file is `memory/robot_map.npz` (or `robot_map_tow.npz`
  while towing). It's loaded automatically, autosaved every 15 simulated
  seconds, and saved on normal exit or Ctrl+C. A missing file starts empty,
  with no preloaded apartment layout. Generated memory, verification
  reports and marker textures are not committed to the repo.
- Each room has two signs by its entrance (one facing in, one facing the
  hallway), each with a distinct ArUco code, the English room name, and
  either INSIDE or ENTRANCE. Codes carry no room names or coordinates -
  `room_signs.py` estimates 3-D pose from pixels and PnP, and English text
  is read separately with a small vocabulary/font recognizer, not general OCR.
- The robot remembers observed free space/obstacles, exploration visits,
  motion-derived route connections, localization landmarks, and room
  names/targets. Room dimensions, doors and corridors are never passed to
  the navigator directly - only what the cameras actually saw.
- The floor classifier uses relative colour contrast and ground-plane
  projection, not learned depth; it's calibrated to the demo's neutral
  checker floor. `vision_config.json` holds camera mounts, resolution/FOV,
  marker size and grid settings - rebuild the world after changing it.
- The complete robot is built at 70% scale, the wheelchair at 60%; both are
  set in `layout.py` and matched in `vision_config.json`.

Tests: `python -m unittest discover -s tests -v`. Independent navigation
verification: `python verify_navigation.py --room bathroom --seconds 180
--output verification/bathroom --memory memory/test_map.npz`.

## 2. Meta Quest VR teleoperation

A separate, standalone simulation for driving the robot directly from a Quest
headset - its own process, own physics state, does not start, import, or
communicate with `run_world.py`, and doesn't read or write navigation memory.

```powershell
adb devices                     # confirm the headset shows as "device"
adb reverse tcp:8765 tcp:8765
python build_world.py           # only if world.xml doesn't exist yet
python quest_teleop.py
```

Enable Developer Mode on the Quest, connect it over USB, and accept USB
debugging in the headset first. Then in **Quest Browser on the headset**,
open `http://127.0.0.1:8765`, confirm the live head-camera preview is
updating, and select **Enter VR and connect**.

| Input | Robot action |
| --- | --- |
| Left thumbstick up/down | Drive forward/backward |
| Turn your head left/right | Swivel the robot to the same relative yaw angle |
| Controller position | Move the matching hand relative to the robot's head |
| Controller rotation | Rotate the matching gripper |
| Either rear grip, within 2m of the chair | Align to it and attach both hands to its handles |
| Rear grip again, while attached | Release the wheelchair |
| Either thumbstick click | Recalibrate head steering / gripper orientation |

The right thumbstick does not steer. The single head camera is shown to both
eyes as a flat, monoscopic view - no synthetic stereo. If input stops arriving
for 350 ms or tracking is lost, driving stops and the arms hold their pose.

If VR immediately closes, restart `quest_teleop.py`, reload the page, and
watch the terminal for lines starting with `Quest Browser:` - a successful
connection ends with `Immersive session, render layer, and local tracking are
ready.` Useful options: `--arm-scale`, `--headless --seconds N`, `--port`.
Full setup, troubleshooting and implementation notes: [VR_TELEOP.md](VR_TELEOP.md).
