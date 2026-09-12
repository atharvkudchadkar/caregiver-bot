# Bracket Bot: three-camera navigation

The active simulator is `build_world.py` + `run_world.py`. Navigation now uses
RGB images from the head and both hands, with wheel-encoder odometry and IMU
tilt. Arm, lift, gripper and balancing controls remain available.

## Run

From this directory (Python 3.12 or 3.13):

```powershell
python -m pip install -r requirements.txt
python build_world.py
python run_world.py --camera-preview
```

Arrow keys drive/turn, Space stops, and R resets the robot and navigation map.
An arrow key also takes over from autonomous navigation. Reverse translation
is allowed only through previously observed clearance. Camera preview shows
all three RGB feeds, with the estimated floor in green. The overview camera
is for the human viewer and is never a navigation input.

```powershell
python run_world.py --goal 1 0 --headless --seconds 15
python run_world.py --explore --camera-preview
python run_world.py --goto kitchen --camera-preview
python run_world.py --joint rj1=0.4 --joint lj6=-0.3 --joint gripper_right=0.8
python build_world.py --obstacle
```

`--goal X Y` is in a local frame at startup: X forward, Y left, metres.
`--goto` accepts a room label from `vision_config.json`. Room labels are
physical ArUco floor mats rendered into the images; this is **label recognition,
not general visual recognition of kitchens or bathrooms**. The robot must
explore to find an unseen label. Room discovery and exploration are experimental;
they can stop at `BLOCKED` before reaching a destination.
Room arrival means stopping within 0.6 m of the observed label; local coordinate
goals use a 0.3 m tolerance. Named-room runs first scan for nearby labels.

Headless means no viewer; RGB rendering still requires an OpenGL context.
On macOS, use `mjpython` for the interactive viewer. `mujoco/run_balance.py`
is the older standalone balance demo; `build_scene.py` is a legacy kinematic
scene builder. Neither is the camera-navigation entry point.

## What changed

Previously, `layout.route()` constructed a waypoint graph from room centres,
door positions and hallway coordinates. `Navigator` followed those routes
using the exact simulator base pose, with nine rangefinders for avoidance.
Those routing functions, the LiDAR adapter, rangefinder sites/sensors and the
avoidance bypass have been removed from the active world/controller.

The new input boundary is:

```text
head RGB + left-hand RGB + right-hand RGB
                 + camera calibration and joint encoders / IMU tilt
                                  |
                          floor masks + visible labels
                                  |
wheel encoders -> local odometry -> observed occupancy map
                                  |
                   inflated free space + online frontier search
                                  |
                         replanned (v, w) commands
                                  |
                        balance controller + wheel motors
```

`vision_navigation.py` imports neither MuJoCo nor `layout`. It receives RGB
frames, calibrated camera transforms and local odometry, not body positions,
scene geometry, room coordinates, contact data, depth images or simulator
segmentation. Goals do not make unseen space traversable. Dijkstra search uses
only the observed map, with obstacles and unknown space inflated by the robot
radius. A persistent exploration target avoids changing direction every frame.
New obstacle observations interrupt previously clear paths. Missing/stale or
unusable images stop movement. A full unsuccessful scan or 45 seconds without
translation reports `BLOCKED`, rather than claiming arrival.

`camera_rig.py` isolates simulation sensor access. Hand mounts are attached to
their respective hand bodies. Their changing transforms are recomputed from
joint encoders on a separate robot FK state with world translation/yaw removed.
A simulated IMU provides roll/pitch; navigation yaw and displacement come from
wheel encoders. The balancing controller still uses MuJoCo state to keep the
robot upright; the navigator cannot access that state.

The three RGB cameras use the standard [MuJoCo renderer](https://mujoco.readthedocs.io/en/stable/programming/visualization.html).
Room labels use [OpenCV ArUco detection](https://docs.opencv.org/4.10.0/d2/d1a/classcv_1_1aruco_1_1ArucoDetector.html).

## Calibration and limits

`vision_config.json` is the calibration/configuration source. The default is
320×240 RGB at 10 Hz, 80° vertical field of view. Mount positions and angles
are approximate because the physical cameras have not been specified. Mount
offsets are in the zero-pose base axes relative to each parent body origin;
the builder converts them to link-local coordinates. After changing camera
mounts or FOV, rebuild the world with the same configuration:

```powershell
python build_world.py --vision-config vision_config.json
python run_world.py --vision-config vision_config.json --camera-preview
```

This implementation is a classical CV baseline with explicit assumptions:

- **Floor perception:** a configurable colour classifier is calibrated to the
  demo's neutral checker floor. It projects visible ground and the bottom
  silhouettes of obstacles onto a flat floor. This is not learned monocular
  depth or general scene understanding. Similar-coloured obstacles, shadows,
  reflective/transparent objects, slopes and drop-offs need a better perception
  model. Floor beyond an occluding wall is excluded from free-space evidence.
- **Localization:** wheel encoders and IMU tilt are assumed available on the
  hardware. Slip causes odometry drift; there is no visual SLAM, loop closure or
  global relocalization. The three articulated cameras are not treated as a
  fixed stereo rig.
- **Coverage:** hand motion and occlusion change visibility. Unseen floor is
  kept unknown. A fixed 0.28 m navigation radius describes the base clearance,
  not the swept volume of extended arms. Physics collisions remain enabled.
- **Mapping:** the odometry map spans 30 m × 30 m at 0.1 m resolution and is
  cleared on reset. There is no prior apartment map, saved route or coordinate
  shortcut for named destinations. Room label IDs/names and camera/floor
  calibration are configuration, not navigation routes.

Hardware integration needs measured intrinsics/distortion and hand/head mount
extrinsics, encoder/IMU interfaces, and a perception/localization model suitable
for the actual floors and lighting. Replace the simulation `CameraRig` with a
hardware adapter producing the same `CameraFrame` and odometry inputs.

## Verification

```powershell
python -m unittest discover -s tests -v
```

The 14 tests cover camera articulation, independence from world pose, absence of
rangefinders, RGB marker recognition, metric ground projection, encoder odometry,
dynamic obstacle replanning, unknown-space handling, stale/black-image stops,
and reaching local goals from two different simulator positions/headings. A
rendered crate also blocks the direct path in the camera-derived map. A separate
headless `--goto bedroom` run detected the label and reported arrival at 16.7 s
without falling. Longer room exploration has detected multiple room labels and
traversed the hallway, but complete room-to-room reliability is not established
by these checks.
