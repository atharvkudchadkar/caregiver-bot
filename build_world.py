"""Compose world.xml: four-room apartment + wheelchair + the balancing robot.

Run:   python build_world.py
Then:  python run_world.py                 (viewer, arrow-key teleop)
       python run_world.py --goto kitchen  (drive to a room)
       python -m mujoco.viewer --mjcf world.xml   (passive look, no controller)

Sources merged:
  balance_robot_with_cameras.xml   robot with wheel motors, arm servos, and cameras
  assets/wheelchair.xml      teammate's wheelchair (renamed with a wc_ prefix)
  layout.py                  room/hallway geometry (walls generated here)

Everything is written in radians. The wheelchair file is in degrees, so its
euler attributes are converted; the robot file is already in radians.
"""
import copy
import math
import os
import xml.etree.ElementTree as ET

import layout

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT_XML = os.path.join(HERE, "balance_robot_with_cameras.xml")
WHEELCHAIR_XML = os.path.join(HERE, "assets", "wheelchair.xml")
OUT = os.path.join(HERE, "world.xml")
WC_PREFIX = "wc_"


def fmt(*vals):
    return " ".join(f"{v:.6g}" for v in vals)


def yaw_quat(yaw_deg):
    h = math.radians(yaw_deg) / 2
    return fmt(math.cos(h), 0, 0, math.sin(h))


def prefix_names(elem, prefix, attrs=("name", "joint", "body", "site", "geom")):
    for el in elem.iter():
        for a in attrs:
            if a in el.attrib:
                el.set(a, prefix + el.attrib[a])
    return elem


def eulers_to_radians(elem):
    for el in elem.iter():
        if "euler" in el.attrib:
            el.set("euler", fmt(*(math.radians(float(v)) for v in el.attrib["euler"].split())))
    return elem


def make_robot_solid(base):
    """Give the robot collision geometry.

    Teammate's file has contype=0 on everything except the tyres. We:
      * turn on mesh collision for every arm link, but only against world
        geoms (layout.COL_ARM), so the arms never collide with the mast or
        each other and the placeholder inertials don't fight spurious contacts;
      * add two hidden, massless primitives to the base body (box + mast
        capsule) that collide with the world. Primitives instead of the cover
        meshes so nothing dips below the tyres and lifts the robot.
    """
    wheels = {"left_wheel", "right_wheel"}
    for body in base.iter("body"):
        if body is base or body.get("name") in wheels:
            continue
        for g in body.findall("geom"):
            g.attrib.update(layout.COL_ARM)
    common = dict(group="3", density="0", rgba="1 0.3 0.3 0.25", **layout.COL_BASE)
    ET.SubElement(base, "geom", name="col_base", type="box",
                  size="0.13 0.2 0.08", pos="0.01 0 0.17", **common)
    ET.SubElement(base, "geom", name="col_mast", type="capsule",
                  fromto="0.01 0 0.25 0.01 0 1.55", size="0.06", **common)


def add_lidar(base, sensor):
    """Nine rangefinder beams fanned across the front at layout.RF_HEIGHT.
    A rangefinder ignores geoms of the body its site sits on (the base), so
    the beams see walls, the wheelchair and obstacles, not the robot."""
    for i, deg in enumerate(layout.RF_ANGLES_DEG):
        a = math.radians(deg)
        ET.SubElement(base, "site", name=f"rf_{i}", pos=fmt(0.12, 0, layout.RF_HEIGHT),
                      zaxis=fmt(math.cos(a), math.sin(a), 0), size="0.01",
                      rgba="0 1 0 0.3", group="3")
        ET.SubElement(sensor, "rangefinder", name=f"rf_{i}", site=f"rf_{i}",
                      cutoff=fmt(layout.RF_CUTOFF))


def build(obstacle=False):
    robot = ET.parse(ROBOT_XML).getroot()
    wc = ET.parse(WHEELCHAIR_XML).getroot()

    root = ET.Element("mujoco", model="caregiver_world")
    # Robot mesh entries are written as file="meshes/X.stl",
    # so meshdir points at the folder that contains "meshes/".
    ET.SubElement(root, "compiler", angle="radian", meshdir="mujoco", autolimits="true")
    ET.SubElement(root, "option", timestep="0.002")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="720")
    ET.SubElement(visual, "headlight", ambient="0.45 0.45 0.45", diffuse="0.6 0.6 0.6")

    # Defaults the wheelchair file relied on, scoped to a class.
    default = ET.SubElement(root, "default")
    wcd = ET.SubElement(default, "default", {"class": "wheelchair"})
    ET.SubElement(wcd, "joint", damping="0.5", armature="0.02")
    ET.SubElement(wcd, "geom", density="300", friction="0.8 0.1 0.05", condim="3", **layout.COL_WORLD)
    ET.SubElement(wcd, "motor", ctrllimited="true", ctrlrange="-1 1")
    wall = ET.SubElement(default, "default", {"class": "wall"})
    ET.SubElement(wall, "geom", type="box", rgba="0.82 0.82 0.86 1", **layout.COL_WORLD)

    # Assets: robot meshes + a checker floor.
    asset = ET.SubElement(root, "asset")
    for mesh in robot.find("asset"):
        asset.append(copy.deepcopy(mesh))
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker",
                  rgb1="0.78 0.78 0.78", rgb2="0.68 0.68 0.7", width="512", height="512")
    ET.SubElement(asset, "material", name="grid", texture="grid", texrepeat="24 24", reflectance="0.05")

    wb = ET.SubElement(root, "worldbody")
    ET.SubElement(wb, "light", pos="0 0 10", dir="0 0 -1", directional="true", castshadow="false")
    ET.SubElement(wb, "geom", name="floor", type="plane", size="12 12 0.05",
                  material="grid", friction="1.2 0.005 0.0001", **layout.COL_WORLD)
    ET.SubElement(wb, "camera", name="overview", pos="0 -14 12", xyaxes="1 0 0 0 0.65 0.76")

    if obstacle:
        ob = layout.OBSTACLE
        ET.SubElement(wb, "geom", {"class": "wall"}, name="obstacle_crate",
                      pos=fmt(*ob["pos"]), size=fmt(*ob["size"]), rgba="0.55 0.35 0.2 1")

    # Walls from layout.py
    for name, x0, y0, x1, y1 in layout.wall_segments():
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if abs(x1 - x0) < 1e-9:   # wall running north-south
            size = (layout.WALL_HALF_T, abs(y1 - y0) / 2 + layout.WALL_HALF_T, layout.WALL_HALF_H)
        else:                      # wall running east-west
            size = (abs(x1 - x0) / 2 + layout.WALL_HALF_T, layout.WALL_HALF_T, layout.WALL_HALF_H)
        ET.SubElement(wb, "geom", {"class": "wall"}, name=f"wall_{name}",
                      pos=fmt(cx, cy, layout.WALL_HALF_H), size=fmt(*size))

    # Room markers (sites double as navigation targets you can read from the model).
    for name, r in layout.ROOMS.items():
        cx, cy = layout.room_center(name)
        ET.SubElement(wb, "site", name=name, type="cylinder", pos=fmt(cx, cy, 0.01),
                      size="0.3 0.005", rgba=fmt(*r["color"], 0.7))

    # Robot: the whole mobile_base subtree, including its head and hand cameras.
    base = copy.deepcopy(robot.find("worldbody/body[@name='mobile_base']"))
    sp = layout.SPAWN["robot"]
    base.set("pos", fmt(*sp["pos"]))
    base.set("quat", yaw_quat(sp["yaw_deg"]))
    make_robot_solid(base)
    sensor = ET.Element("sensor")          # appended to root below
    add_lidar(base, sensor)
    wb.append(base)

    # Wheelchair: prefixed so its joint names don't collide with the robot's wheels.
    chair = copy.deepcopy(wc.find("worldbody/body[@name='wheelchair']"))
    prefix_names(eulers_to_radians(chair), WC_PREFIX)
    sp = layout.SPAWN["wheelchair"]
    chair.set("pos", fmt(*sp["pos"]))
    chair.set("quat", yaw_quat(sp["yaw_deg"]))
    chair.set("childclass", "wheelchair")
    wb.append(chair)

    # Constraints / actuators / sensors from both sources.
    equality = ET.SubElement(root, "equality")
    for eq in robot.find("equality"):
        equality.append(copy.deepcopy(eq))

    actuator = ET.SubElement(root, "actuator")
    for act in robot.find("actuator"):
        actuator.append(copy.deepcopy(act))
    for act in wc.find("actuator"):
        a = prefix_names(copy.deepcopy(act), WC_PREFIX)
        a.set("class", "wheelchair")
        actuator.append(a)

    for s in wc.find("sensor"):
        sensor.append(prefix_names(copy.deepcopy(s), WC_PREFIX))
    root.append(sensor)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(OUT, encoding="utf-8", xml_declaration=True)
    return OUT


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Compose world.xml")
    ap.add_argument("--obstacle", action="store_true",
                    help="drop a crate in the hallway to test obstacle avoidance")
    args = ap.parse_args()

    out = build(obstacle=args.obstacle)
    import mujoco
    model = mujoco.MjModel.from_xml_path(out)
    print(f"wrote {out}" + ("  (with hallway obstacle)" if args.obstacle else ""))
    print(f"  bodies={model.nbody} joints={model.njnt} geoms={model.ngeom} "
          f"actuators={model.nu} sensors={model.nsensor} cameras={model.ncam}")
    for name in ("mobile_base", f"{WC_PREFIX}wheelchair"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        print(f"  {name:16s} body id {bid}")
    print("  rooms:", ", ".join(layout.ROOMS))


if __name__ == "__main__":
    main()
