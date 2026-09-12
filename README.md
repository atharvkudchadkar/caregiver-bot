# Bracket Bot: three-camera navigation with a learned map memory

The active simulator is `build_world.py` + `run_world.py`. Navigation uses RGB
images from the head and both hands, wheel-encoder odometry and IMU tilt. Arm,
lift, gripper and balancing controls remain available. Nothing about the
apartment is given to the navigator: it learns the floor plan by driving, keeps
that memory between runs, and uses door signs only to recognise rooms and to
work out where it is inside its own memory.

## Run

From this directory (Python 3.12 or 3.13):

```
python -m pip install -r requirements.txt
python build_world.py
python run_world.py --explore --camera-preview
```

Arrow keys drive/turn, Space stops, and R resets the robot (the memory is kept).
An arrow key also takes over from autonomous navigation. Camera preview shows
all three RGB feeds with the estimated floor in green. The overview camera is
for the human viewer and is never a navigation input.

```
python run_world.py --goto kitchen --camera-preview      find the kitchen by its sign / from memory
python run_world.py --goal 1 0 --headless --seconds 15   local goal: 1 m ahead of the start pose
python run_world.py --explore --forget                    start from a blank memory
python run_world.py --goto bathroom --no-memory           neither load nor save memory
python run_world.py --joint rj1=0.4 --joint lj6=-0.3 --joint gripper_right=0.8
python build_world.py --obstacle                          crate in the hallway
```

The terminal narrates what the robot is doing: what it is looking for, which
sign it can see, when it has placed itself in the saved memory, where it is
exploring, when it replans, and what it saved at the end.

## Door signs

Two signs hang on the wall beside every doorway at 0.95 m, each 1.25 m to the
right of the door as you face it: one on the hallway face (the room's
*outside* code) and one on the room face (*inside*). Each sign carries a 4x4
ArUco code and the room's English name. `vision_config.json` maps code ->
room + side and holds the signage convention (`marker_offset`,
`sign_door_offset`). The head camera detects codes; `solvePnP` on the known
0.34 m code gives the sign's position and facing, and the convention turns that
into the door centre and "into the room" direction.

If the room names appear upside down in the camera preview, set
`"sign_image_flip": true` in `vision_config.json` and rebuild the world.

## Memory (`memory/apartment_memory.npz`)

`MapMemory` in `vision_navigation.py` holds what the robot has learned:

* an occupancy grid (30 m x 30 m, 0.1 m) built only from what the cameras saw,
* door signs it has seen: position and facing direction,
* per room: door centre, inward direction, and an entry point 1.2 m inside.

The memory frame is the odometry frame of the very first run. A new run starts
in a fresh local map; the moment the robot sees a sign it remembers, it
computes the rigid transform from this run's odometry into the memory frame,
replays what it saw so far into the memory, and from then on plans over the
remembered map. `--goto kitchen` therefore drives straight there on the second
run, without exploring. Known signs also nudge the transform to cancel slow
odometry drift. The memory is saved every 10 s and at exit; a run that never
saw a remembered sign is not saved, because it cannot be aligned.

## Input boundary

```
head RGB + left-hand RGB + right-hand RGB
              + camera calibration and joint encoders / IMU tilt
                               |
             floor masks + door-sign poses (PnP)
                               |
wheel encoders -> local odometry --(sign match)--> memory frame
                               |
       learned occupancy map + landmarks (saved between runs)
                               |
        inflated free space + Dijkstra / online frontier search
                               |
                     replanned (v, w) commands
                               |
                   balance controller + wheel motors
```

`vision_navigation.py` imports neither MuJoCo nor `layout`. It receives RGB
frames, calibrated camera transforms and local odometry, not body positions,
scene geometry, room coordinates, contact data, depth images or simulator
segmentation. Goals do not make unseen space traversable. Missing/stale or
unusable images stop movement. A full unsuccessful scan or 45 s without
translation reports `BLOCKED` rather than claiming arrival.

`camera_rig.py` isolates simulation sensor access. Hand mounts are attached to
the hand bodies; their transforms are recomputed from joint encoders on a
separate FK state with world translation/yaw removed. A simulated IMU provides
roll/pitch; navigation yaw and displacement come from wheel encoders. The
balancing controller still uses MuJoCo state to keep the robot upright; the
navigator cannot access that state.

## Calibration and limits

`vision_config.json` is the calibration/configuration source: 480x360 RGB at
10 Hz, 80° vertical field of view, camera mounts, floor colour thresholds, sign
catalogue and sizes, memory path. After changing camera mounts, FOV or signs,
rebuild the world.

This is a classical CV baseline with explicit assumptions:

- Floor perception: a colour classifier calibrated to the demo's neutral
  checker floor; visible ground and obstacle bottom silhouettes are projected
  onto a flat floor. Not learned depth or general scene understanding.
- Localization: wheel encoders and IMU tilt are assumed on the hardware. Sign
  sightings correct drift; there is no visual SLAM or loop closure beyond that.
- Coverage: unseen floor stays unknown. A fixed 0.28 m navigation radius
  describes the base clearance, not the swept volume of extended arms.
- Memory: the grid is cleared with `--forget`; obstacles that move are
  overwritten by later clear observations.

Hardware integration needs measured intrinsics/distortion and mount extrinsics,
encoder/IMU interfaces, and a perception model suited to the real floors and
lighting. Replace the simulation `CameraRig` with a hardware adapter producing
the same `CameraFrame` and odometry inputs; printed ArUco signs by the doors
replace the rendered ones.

## Verification

```
python -m unittest discover -s tests -v
```

The tests cover camera articulation, independence from world pose, absence of
rangefinders, wall-sign detection with position/facing/height, room entry
points from signs, odometry-frame localization from a remembered sign, memory
save/load, encoder odometry, dynamic obstacle replanning, unknown-space
handling, stale/black-image stops, and reaching local goals from two different
simulator positions. A rendered crate also blocks the direct path in the
camera-derived map.
