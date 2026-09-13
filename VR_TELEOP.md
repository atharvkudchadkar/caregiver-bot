# Quest VR teleoperation

This ports only `quest_teleop.py` and `quest_controller.html` from
[`vr-quest-teleop`, commit 259acba](https://github.com/atharvkudchadkar/caregiver-bot/tree/259acba).
It runs **its own simulation** using the current `world.xml`, including the
scaled robot and existing room layout. It does not import, launch, modify, or
communicate with `run_world.py`. You can run both programs, but their robot
states are independent. The VR runner does not load or write navigation memory.

## Connect a Quest

1. Install the project's Python dependencies as usual. Install Android
   [SDK Platform Tools](https://developer.android.com/tools/releases/platform-tools)
   separately so `adb` is available in your terminal.
2. Enable developer mode on the Quest, connect it over USB, and accept USB
   debugging in the headset. In PowerShell:

   ```powershell
   adb devices
   adb reverse tcp:8765 tcp:8765
   python quest_teleop.py
   ```

   The headset must appear as `device`, rather than `unauthorized`. If several
   devices are connected, use `adb -s YOUR_QUEST_SERIAL reverse tcp:8765 tcp:8765`.
   Run `python build_world.py` once first if `world.xml` does not yet exist;
   the VR runner deliberately never rebuilds or overwrites it.
3. In **Quest Browser on the headset**, open `http://127.0.0.1:8765`. Check that
   the camera preview is updating, then select **Enter VR and connect**.
4. Hold both controllers in front of you. The desktop MuJoCo viewer shows the
   same simulation being controlled by this VR process.

The HTTP server listens only on the computer's loopback interface. ADB forwards
the headset's own loopback port to it; do not substitute the PC's LAN address.
WebXR needs a secure context, which includes loopback origins. This connection
does not use Quest Link. See [ADB documentation](https://developer.android.com/tools/adb)
and the [WebXR specification](https://www.w3.org/TR/webxr/).

If entering VR immediately returns to the page, restart `quest_teleop.py`, reload
the page in Quest Browser, and try again. The page now prints each WebXR startup
stage and the exact browser error in the Python terminal. A successful startup
prints `Immersive session, render layer, and local tracking are ready.` If the
session ends without a JavaScript error, keep Quest Browser in the foreground,
confirm the headset is awake, and update Quest Browser/headset software.

## Controls

| Input | Robot action |
| --- | --- |
| Left thumbstick up/down | Drive forward/backward, up to 0.25 m/s |
| Turn your head left/right | Swivel the robot to the same relative yaw angle |
| Controller position | Move the matching hand relative to the robot's head |
| Controller rotation | Rotate the matching gripper relative to its calibrated orientation |
| Either rear grip/squeeze press within 2 m of chair | Align to its heading and attach both hands to the handles |
| Rear grip/squeeze press while attached | Release the wheelchair and resume free arm tracking |
| Either thumbstick click | Recalibrate head steering and gripper orientation references |
| Exit VR / Stop and exit VR | Stop driving and hold the arms |

The mapping follows the [WebXR `xr-standard` layout](https://www.w3.org/TR/webxr-gamepads-module-1/).
Arm reach is scaled by 0.7 for the current smaller robot, bounded by joint limits,
and applied through the existing arm servos. `--arm-scale` can tune the mapping.
An unreachable free-arm target is approximated within joint limits. The VR-only
auto-attach action aligns the robot at the chair's rear using the chair's actual
position and orientation, solves both arms onto the two handle sites, and locks
that relative pose for towing. The attached arms remain fixed on the handles.
Every arm link collides with every wheelchair component in this VR process.
These runtime collision changes do not alter `world.xml` or `run_world.py`.

The single **head_cam** image is shown to both eyes as a flat video view. It is
monoscopic, with no synthetic stereo or 360-degree view. Head yaw controls the
mobile base's target heading; pitch and roll do not steer. This is based on
the source branch. Hand cameras and autonomous navigation are not part of this
VR process; room collision geometry remains active.

If input stops arriving for 350 ms, tracking disappears, or VR exits, driving
stops and the arms hold their current pose on the next control update. A missing
controller holds its own arm. The browser also pauses input when video is stale
or the headset session loses visibility. If the robot falls, restart this VR
runner to reset its simulation.

## Options and checks

```powershell
python quest_teleop.py --world world.xml --arm-scale 0.7
python quest_teleop.py --headless --seconds 60
python quest_teleop.py --port 8766
adb reverse tcp:8766 tcp:8766
python -m unittest discover -s tests -p test_quest_teleop.py -v
node tests/test_quest_controller.cjs
```

Headless runs still render camera video and need OpenGL; `--seconds` is wall-clock
duration. Normal runs stop when you close their viewer or press Ctrl+C. On macOS
the viewer relaunches with `mjpython` if installed. Remove forwarding afterwards
with `adb reverse --remove tcp:8765` if desired.

Integration fixes include the current `head_cam` camera name, shared controllers
from `robot_base.py`, proportional arm reach, and exactly one physics step per
control tick (the upstream runner stepped twice). Tests use synthetic WebXR
packets and real MuJoCo physics/rendering. They cannot verify headset optics,
USB connectivity, or physical Quest tracking without the headset.
