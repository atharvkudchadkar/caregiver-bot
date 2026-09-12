"""Step 1: load the Bracket Bot URDF into MuJoCo and save a native MJCF copy.

Run once:  python convert.py
Expects the URDF package at ./robot (urdf/ and meshes/ side by side).
Writes    ./robot/urdf/chopped_urdf_v2_mj.urdf  (URDF + MuJoCo compiler hints)
          ./robot_converted.xml                 (native MJCF, used by build_scene.py)
"""
import os
import re
import sys

import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "robot", "urdf", "chopped_urdf_v2.urdf")
OUT_URDF = os.path.join(HERE, "robot", "urdf", "chopped_urdf_v2_mj.urdf")
OUT_MJCF = os.path.join(HERE, "robot_converted.xml")

if not os.path.exists(SRC):
    sys.exit(f"missing {SRC} - copy the chopped_urdf_v2 folder to ./robot first")

src = open(SRC, encoding="utf-8").read()
# MuJoCo reads URDF directly if we give it compiler hints:
#  meshdir       -> where the STLs live, relative to the urdf file
#  strippath     -> turns package://chopped_urdf_v2/meshes/X.stl into X.stl
#  balanceinertia-> tolerates the placeholder inertials in this export
hint = (
    r'\1\n  <mujoco><compiler meshdir="../meshes" strippath="true" '
    r'balanceinertia="true" discardvisual="false" fusestatic="false"/></mujoco>'
)
src = re.sub(r'(<robot name="[^"]+">)', hint, src, count=1)
open(OUT_URDF, "w", encoding="utf-8").write(src)

model = mujoco.MjModel.from_xml_path(OUT_URDF)
print(f"loaded: nbody={model.nbody} njnt={model.njnt} nmesh={model.nmesh}")
for j in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    print(f"  {name:22s} range={model.jnt_range[j]}")

mujoco.mj_saveLastXML(OUT_MJCF, model)
print(f"saved {OUT_MJCF}")
