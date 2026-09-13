Original organizer URDF (untouched):

  robot/urdf/chopped_urdf_v2.urdf

This file is already in git. To restore it after any experiment:

  git checkout -- robot/urdf/chopped_urdf_v2.urdf
  git checkout -- robot/urdf/chopped_urdf_v2_mj.urdf

Do not treat this URDF as the live simulation model.
push_wheelchair.py / build_world.py load mujoco/balance_robot.xml.
Changing the URDF has no effect until someone reconverts and replaces
that MuJoCo file. The finger links also have no <collision> meshes —
that is why build_world.py adds jaw pads in the composed world.
