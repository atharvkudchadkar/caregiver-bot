# Bracket Bot: camera navigation with learned memory

The active simulator is `build_world.py` + `run_world.py`. It uses the head and
both hand RGB cameras, wheel encoders and IMU tilt. The robot learns the visible
layout as it moves and saves it between runs. Joint and balancing controls are
unchanged.

## Run

For the independent Quest headset view, arm controls, and joystick driver, see
[VR_TELEOP.md](VR_TELEOP.md). Launch it with `python quest_teleop.py`; it uses
the same world file in its own process and leaves `run_world.py` unchanged.

From this directory with Python 3.12 or 3.13:

```powershell
python -m pip install -r requirements.txt
python build_world.py
python run_world.py --explore --camera-preview
python run_world.py --goto bathroom --camera-preview
```

## Meta Quest VR mode

VR runs as a separate simulation and does not start or communicate with
`run_world.py`. Enable Developer Mode for the Quest, connect it with a USB data
cable, accept USB debugging in the headset, and verify that `adb devices` lists
it as `device`. Then run:

```powershell
adb reverse tcp:8765 tcp:8765
python build_world.py
python quest_teleop.py
```

Open `http://127.0.0.1:8765` in Quest Browser, confirm the live head-camera
preview, and select **Enter VR and connect**. The left thumbstick drives forward
and backward. Turning your head swivels the robot through the same relative yaw
angle; the right thumbstick does not steer. Moving each controller moves the
matching arm. Press either rear grip within 2 metres of the wheelchair to align
with its orientation and attach both hands to its handles; press again to
release it. Every arm link collides with the complete wheelchair in VR mode.

If VR immediately closes, restart `quest_teleop.py`, reload the page, and watch
the terminal for lines beginning with `Quest Browser:`. A successful connection
ends with `Immersive session, render layer, and local tracking are ready.` See
[VR_TELEOP.md](VR_TELEOP.md) for full setup, controls, and troubleshooting.

The default memory file is `memory/robot_map.npz`. It is loaded automatically,
autosaved every 15 simulated seconds, and saved on normal exit or Ctrl+C. Use
`--memory memory/another_building.npz` for a different environment. A missing
file starts with an empty map except the robot's current footprint; there is
no preloaded apartment layout. `--no-memory` disables both loading and saving.
Generated memory, verification reports and marker textures are not committed.

The terminal explains the robot's current action at state changes and every
five simulated seconds: searching for a room, rotating to inspect an area,
following a learned route, replanning after an obstacle, waiting for camera
coverage, or stopping. It also reports newly learned localization landmarks,
read English signs, mapped area, successful relocalization and memory saves.

Arrow keys drive/turn, Space stops, and R resets the physical robot while
preserving learned memory. Pressing an arrow key takes over from autonomous
navigation. After reset/restart, the robot must relocalize before translating
on the saved map. Manual rotation remains available for finding a code.

```powershell
python run_world.py --goto kitchen --headless --seconds 300
python run_world.py --goal 1 0 --headless --seconds 15 --no-memory
python run_world.py --joint rj1=0.4 --joint lj6=-0.3 --joint gripper_right=0.8
python build_world.py --obstacle
```

`--goal X Y` uses startup-local coordinates: X forward, Y left, metres. The
human's overview camera is never a navigation input. Headless mode still needs
an OpenGL context for RGB rendering. On macOS use `mjpython` for the interactive
viewer. The older `mujoco/run_balance.py` and `build_scene.py` are separate demos.

## Robot size and wheelchair attachment

The complete robot is built at 70% of its original dimensions (meshes, joints,
grippers and collision shapes); the wheelchair remains at 60%. The camera
mounts and wheel odometry calibration in `vision_config.json` match this size.
Rebuild with `python build_world.py` before launching an updated simulation.

Press **G** to approach and grasp the wheelchair, then use the arrow keys to
drive the attached pair; **X** releases it. `python run_world.py --attach`
starts the grasp immediately. This grasp controller currently uses the
simulator's known wheelchair pose. After both hands verify handle contacts,
a planar simulated hitch maintains the chair's relative position and heading
while allowing the robot to balance. The same hitch update runs in the main
app and `push_wheelchair.py`; it is a simulated attachment, not a model of load
transfer through the hands. Navigation still uses the camera-based learned map.

Run `python -m unittest discover -s tests -p test_wheelchair.py -v` to check
whole-robot scaling, grasping from the normal spawn, forward/reverse towing,
turning, and release.

## Doorway signs

Each room has two signs at the front wall beside its entrance: one facing the
room, one facing the hallway. Each contains a distinct ArUco code, the English
room name, and either INSIDE or ENTRANCE. There are no floor-code mats.

Codes carry **no room names or coordinates**. `room_signs.py` estimates their
3-D position/orientation from pixels, camera calibration and the printed marker
size. English text is recognized separately after perspective rectification.
The room name and side line must both be readable, and repeated readings are
required before remembering a semantic label. An obscured line is not inferred
from the code. This is a small vocabulary/font template recognizer for the
provided signs, not unrestricted OCR.

## What the robot remembers

- Camera-observed free space, obstacles and unknown areas in an occupancy grid.
- Exploration visits, so already examined areas can be deprioritized.
- Motion-derived route connections: cells the robot actually traveled through,
  recorded separately from exploration scores. These guide return trips when
  noisy occupancy observations temporarily disconnect a remembered doorway.
- Localization landmarks and their observed poses in the learned map frame.
- English room names, the observed signs associated with them, and room-side
  floor or previously visited destination positions.

Doorway approaches are derived from observed free floor on the interior side of
a sign's wall. Room dimensions, door coordinates and corridor routes are never
passed to the navigator. Dijkstra planning runs on the learned grid with base
clearance; occupied and unknown space are not traversable. Exploration aims for
visible free/unknown boundaries, not unknown space hidden behind a wall.

On a later run, the robot loads its memory but **does not assume the same start
position**. It waits for consistent observations of a saved landmark, aligns the
new odometry frame with the saved map, then navigates using memory and live
cameras. Codes can leave view after this alignment. Later sightings provide
bounded odometry-drift corrections. If a saved landmark cannot be found, the
robot stops after a scan and leaves the stored map untouched. This does not
provide global visual SLAM or relocalization in a completely unfamiliar part of
the building; a recognizable saved landmark is needed to initialize a reused map.

Memory uses a versioned, non-pickle NPZ archive with JSON metadata and atomic
file replacement. Corrupt/incompatible archives produce an error rather than
silently discarding learned data. Keep separate files for different buildings
or materially changed sign arrangements.

## Sensor and implementation boundary

`vision_navigation.py`, `navigation_memory.py` and `room_signs.py` do not import
MuJoCo or the apartment's `layout.py`. They receive RGB, camera calibration and
local odometry, not world positions, room geometry, LiDAR, depth buffers,
collision/contact data or simulator segmentation. `layout.py` is used only by
the world builder and the independent verification evaluator.

`camera_rig.py` renders exactly three robot RGB feeds. Hand cameras are attached
to their hand bodies. Articulated camera transforms are calculated from joint
encoders in a separate local FK state with global translation/yaw removed. IMU
tilt supplies gravity alignment; wheel encoders supply navigation odometry.
The balance controller independently uses simulator state to keep the base upright.

## Calibration and limits

`vision_config.json` sets the camera mounts, 480?360 RGB at 8 Hz, 80? vertical
FOV, printed marker size, wheel dimensions, floor classifier and grid settings.
After changing mounts/FOV, rebuild the world with the same `--vision-config`.
The head looks toward the wall signs while the hand cameras provide nearer
floor coverage. Mount positions remain approximate until hardware measurements
are supplied.

The floor classifier is calibrated to the demo's neutral checker floor. It uses
relative colour contrast to distinguish it from shaded walls, projects visible
ground to a flat plane, and uses bottom silhouettes for obstacles. This is not
learned monocular depth. Similar-coloured obstacles, transparency, difficult
lighting, slopes and drop-offs require stronger perception. A 0.28 m base
clearance does not model the swept volume of extended arms. The map spans
30?30 m at 0.1 m resolution. Encoder slip and imperfect landmark pose estimates
can still affect mapping; memory is not a guarantee that old space stays clear.
Live camera observations continue to update obstacles and trigger replanning.

## Verification

```powershell
python -m unittest discover -s tests -v
python verify_navigation.py --room bathroom --seconds 180 --output verification/bathroom --memory memory/test_map.npz
python verify_navigation.py --room bedroom --spawn -3.5 2.2 -90 --output verification/restart --memory memory/test_map.npz
```

The evaluator saves a JSON result, trajectory, terminal narration, three camera
images and a learned-map image. Ground truth is used only in this evaluator to
check whether the robot actually entered the requested room, stayed upright,
and avoided non-floor contacts. A controller ARRIVED message alone is not a pass.
Unit/integration tests also cover memory round trips, relocalization from a
changed starting position, ID-independent English text, unreadable signs,
corrupt archives, articulated cameras, dynamic obstacles and local goal motion.

Remembered motion is only a planning guide: every local segment is still checked
against the camera-updated occupancy grid. It does not override a newly observed
obstacle. On first arrival the robot scans to learn inside-facing landmarks for
future restarts. The supplied default memory was learned during the verification
runs; use a different, nonexistent `--memory` filename to watch learning from scratch.

Implementation references: [OpenCV marker pose estimation](https://docs.opencv.org/4.10.0/d5/d1f/calib3d_solvePnP.html)
and [MuJoCo rendering](https://mujoco.readthedocs.io/en/stable/programming/visualization.html).
