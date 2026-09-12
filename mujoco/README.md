# Two-wheel robot balance demo (MuJoCo)

A simulation of the robot from `chopped_urdf_v2`, with two driven wheels, a Python balance controller, and scripted targets for both arms and grippers. The model and mesh files are included; no external mesh paths are needed.

## Set up

Use Python 3.13. From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

On macOS, launch the interactive simulation with MuJoCo's `mjpython`:

```bash
mjpython run_balance.py --speed 0
```

On Ubuntu or another Linux desktop, use `python3 run_balance.py --speed 0`. The script must run the simulation: opening `balance_robot.xml` by itself will not activate the balance controller.

## Control the robot

```bash
mjpython run_balance.py --list-joints
mjpython run_balance.py --speed 0.1
mjpython run_balance.py --speed -0.1 --joint rj1=0.4 --joint lj1=-0.4
mjpython run_balance.py --joint rj0=-0.1 --joint right_left_gripper=0.5
```

`--speed` is a straight-line target in metres per second. Use `--joint NAME=VALUE` multiple times to command multiple joints. `rj` names refer to the right arm, `lj` to the left; `rj0` and `lj0` are vertical lifts measured in metres. Rotating joint targets use radians. Unspecified joints hold their zero position. Targets ramp gradually. Close the viewer and restart the command to change command-line targets.

The script writes all actuator controls every step, so the viewer's Controls sliders will be overwritten.

## Check without a viewer

```bash
python3 run_balance.py --headless --seconds 10 --tilt-deg 3 --joint rj1=0.4
```

The run prints elapsed time, base position, pitch, fall status, and commanded joint results.

## Model scope

This is an early simulation, not a controller for the physical robot. It has no virtual blue support pads and balances near upright using wheel torque. The wheel and arm meshes come from the supplied URDF, but body mass, collision geometry, wheel traction, and motor limits have not been calibrated against hardware. It currently handles straight driving; it has no steering, wheelchair model, speech control, navigation, obstacle detection, or emergency-stop hardware interface. Test those separately before any real-world use with a wheelchair or person.
