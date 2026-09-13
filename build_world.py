"""Compose world.xml: four-room apartment + wheelchair + the balancing robot.

Run:   python build_world.py
Then:  python run_world.py                 (viewer, arrow-key teleop)
       python run_world.py --goto kitchen  (drive to a room)
       python -m mujoco.viewer --mjcf world.xml   (passive look, no controller)

Sources merged:
  mujoco/balance_robot.xml   teammate's robot with wheel motors + arm servos
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
from camera_rig import load_config

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT_XML = os.path.join(HERE, "mujoco", "balance_robot.xml")
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


def add_cameras(base, config):
    """Calibrate home mounts into each link's local frame; wrists then articulate."""
    import mujoco
    import numpy as np
    model = mujoco.MjModel.from_xml_path(ROBOT_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for camera in config["cameras"]:
        body = base if camera["body"] == "mobile_base" else base.find(f".//body[@name='{camera['body']}']")
        bid = model.body(camera["body"]).id
        rotation = data.xmat[bid].reshape(3, 3)
        yaw, pitch = map(math.radians, (camera["yaw_deg"], camera["pitch_deg"]))
        right = np.array([math.sin(yaw), -math.cos(yaw), 0])
        up = np.array([math.sin(pitch)*math.cos(yaw), math.sin(pitch)*math.sin(yaw), math.cos(pitch)])
        ET.SubElement(body, "camera", name=camera["name"],
                      pos=fmt(*(rotation.T @ camera["offset"])),
                      xyaxes=fmt(*(rotation.T @ right), *(rotation.T @ up)), fovy=str(config["fovy"]))


def add_room_labels(asset, wb, config):
    """Wall signs on both sides of each entrance; geometry stays in the builder."""
    import cv2
    from room_signs import sign_image
    directory = os.path.join(HERE, "assets", "markers")
    os.makedirs(directory, exist_ok=True)
    for index, (room, bounds) in enumerate(layout.ROOMS.items()):
        inward = 1 if layout.is_north(room) else -1
        wall_y = bounds["y"][0] if inward > 0 else bounds["y"][1]
        for side, face in enumerate(("inside", "entrance")):
            marker_id = index * 2 + side
            normal_y = inward if face == "inside" else -inward
            name = f"sign_{room}_{face}"
            filename = f"{name}.png"
            cv2.imwrite(os.path.join(directory, filename), sign_image(marker_id, room, face))
            ET.SubElement(asset, "texture", name=name, type="2d", file=f"assets/markers/{filename}")
            ET.SubElement(asset, "material", name=name, texture=name, texrepeat="1 1", texuniform="false")
            ET.SubElement(wb, "geom", name=name, type="plane",
                          pos=fmt(bounds["door_x"] + layout.DOOR_W/2 + .45,
                                  wall_y + normal_y*(layout.WALL_HALF_T+.004), .84),
                          xyaxes=fmt(-normal_y, 0, 0, 0, 0, 1),
                          size=fmt(.4*config["marker_size"]/.3, .3*config["marker_size"]/.3, .001),
                          material=name, contype="0", conaffinity="0")


def add_corridor_labels(asset, wb, config):
    """Two ArUco-only codes per bridge, flanking its doorway on the west
    room's side. Their printed text ("CORRIDOR"/"START") matches no name in
    config["room_names"], so SignReader never resolves them to a room - they
    exist purely as localization landmarks, the same mechanism as the room
    signs (see room_signs.SignReader)."""
    import cv2
    from room_signs import sign_image
    directory = os.path.join(HERE, "assets", "markers")
    os.makedirs(directory, exist_ok=True)
    for key, bridge in layout.BRIDGES.items():
        for marker_id, (x, y) in zip(bridge["marker_ids"], layout.bridge_marker_positions(key)):
            name = f"sign_corridor_{marker_id}"
            filename = f"{name}.png"
            cv2.imwrite(os.path.join(directory, filename), sign_image(marker_id, "corridor", "start"))
            ET.SubElement(asset, "texture", name=name, type="2d", file=f"assets/markers/{filename}")
            ET.SubElement(asset, "material", name=name, texture=name, texrepeat="1 1", texuniform="false")
            ET.SubElement(wb, "geom", name=name, type="plane",
                          pos=fmt(x - (layout.WALL_HALF_T+.004), y, .84),
                          xyaxes=fmt(0, -1, 0, 0, 0, 1),
                          size=fmt(.4*config["marker_size"]/.3, .3*config["marker_size"]/.3, .001),
                          material=name, contype="0", conaffinity="0")


def add_furniture(wb):
    """Static props: boxes/cylinders only, collidable like walls."""
    for room, items in layout.furniture().items():
        for item in items:
            ET.SubElement(wb, "geom", name=f"{room}_{item['name']}", type=item["type"],
                          pos=fmt(*item["pos"]), size=fmt(*item["size"]),
                          rgba=fmt(*item["rgba"]), **layout.COL_WORLD)


def build(obstacle=False, config_path=None):
    config = load_config(config_path)
    robot = ET.parse(ROBOT_XML).getroot()
    wc = ET.parse(WHEELCHAIR_XML).getroot()

    root = ET.Element("mujoco", model="caregiver_world")
    # Mesh entries in balance_robot.xml are written as file="meshes/X.stl",
    # so meshdir points at the folder that contains "meshes/".
    ET.SubElement(root, "compiler", angle="radian", meshdir="mujoco", autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", iterations="80",
                  tolerance="1e-8", integrator="implicitfast")
    ET.SubElement(root, "size", nconmax="400", njmax="2000")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="720")
    ET.SubElement(visual, "headlight", ambient="0.45 0.45 0.45", diffuse="0.6 0.6 0.6")

    # Defaults the wheelchair file relied on, scoped to a class.
    default = ET.SubElement(root, "default")
    wcd = ET.SubElement(default, "default", {"class": "wheelchair"})
    ET.SubElement(wcd, "joint", damping="0.5", armature="0.02")
    # Chair geoms accept floor/base/arm contacts but do not collide with one
    # another (the source caster wheels otherwise intersect the footrests).
    ET.SubElement(wcd, "geom", density="300", friction="0.8 0.1 0.05", condim="3",
                  contype="8", conaffinity="5")
    ET.SubElement(wcd, "motor", ctrllimited="true", ctrlrange="-1 1")
    wall = ET.SubElement(default, "default", {"class": "wall"})
    ET.SubElement(wall, "geom", type="box", rgba="0.82 0.82 0.86 1", **layout.COL_WORLD)

    # Assets: robot meshes + a checker floor.
    asset = ET.SubElement(root, "asset")
    for mesh in robot.find("asset"):
        asset.append(copy.deepcopy(mesh))
    from assets.make_aruco_marker import generate as generate_wheelchair_marker
    generate_wheelchair_marker()
    ET.SubElement(asset, "texture", name="wc_aruco_texture", type="2d",
                  file="assets/wheelchair_aruco_4x4_50_id0.png")
    ET.SubElement(asset, "material", name="wc_aruco_material", texture="wc_aruco_texture",
                  emission="1", specular="0", shininess="0")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker",
                  rgb1="0.78 0.78 0.78", rgb2="0.68 0.68 0.7", width="512", height="512")
    ET.SubElement(asset, "material", name="grid", texture="grid", texrepeat="24 24", reflectance="0")

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

    add_room_labels(asset, wb, config)
    add_corridor_labels(asset, wb, config)
    add_furniture(wb)

    # Robot: the whole mobile_base subtree, plus a forward-looking head camera.
    base = copy.deepcopy(robot.find("worldbody/body[@name='mobile_base']"))
    sp = layout.SPAWN["robot"]
    base.set("pos", fmt(*sp["pos"]))
    base.set("quat", yaw_quat(sp["yaw_deg"]))
    add_cameras(base, config)
    make_robot_solid(base)
    for hand_name, side, grasp_pos in (
        ("hand__hand", "right", "0.021785 -0.094351 -0.020721"),
        ("l_hand__hand", "left", "0.002393 -0.095113 0.020571"),
    ):
        hand = base.find(f".//body[@name='{hand_name}']")
        ET.SubElement(hand, "site", name=f"{side}_grasp", pos=grasp_pos,
                      size="0.008", rgba="1 0.4 0 1")
    # Thin rubber pads on the inner jaw faces. These, not the finger meshes,
    # are the rigid clamp surfaces (box-on-box against the flat handle).
    pads = (
        ("right", "lower", "left_finger__left_finger", "-0.0422 -0.0107 -0.0295",
         "-0.70157 -0.02082 0.71229 0.70952 0.07235 0.70096"),
        ("right", "upper", "right_finger__right_finger", "0.0437 0.0053 0.0313",
         "0.70165 -0.01815 -0.71229 0.70979 -0.06966 0.70096"),
        ("left", "lower", "l_left_finger__left_finger", "0.0430 -0.0067 0.0295",
         "0.65333 -0.25651 -0.71229 0.68089 -0.21224 0.70096"),
        ("left", "upper", "l_right_finger__right_finger", "-0.0423 0.0123 -0.0313",
         "-0.63809 0.29238 0.71229 0.62533 -0.34295 0.70096"),
    )
    pad_contact = dict(type="box", size="0.022 0.016 0.006",
                       rgba="0.10 0.10 0.12 1", mass="0.001",
                       friction="2.2 0.12 0.02", condim="3",
                       solref="0.006 1", solimp="0.98 0.99 0.001",
                       **layout.COL_FINGER)
    for side, jaw, body_name, pos, xyaxes in pads:
        finger = base.find(f".//body[@name='{body_name}']")
        ET.SubElement(finger, "geom", name=f"{side}_{jaw}_pad", pos=pos,
                      xyaxes=xyaxes, **pad_contact)
        ET.SubElement(finger, "site", name=f"{side}_{jaw}_pad_site", pos=pos,
                      xyaxes=xyaxes, size="0.003", rgba="1 0.6 0 1")
    sensor = ET.Element("sensor")          # appended to root below
    ET.SubElement(sensor, "framequat", name="base_orientation", objtype="body", objname="mobile_base")
    wb.append(base)

    # Wheelchair: prefixed so its joint names don't collide with the robot's wheels.
    chair = copy.deepcopy(wc.find("worldbody/body[@name='wheelchair']"))
    prefix_names(eulers_to_radians(chair), WC_PREFIX)
    sp = layout.SPAWN["wheelchair"]
    chair.set("pos", fmt(*sp["pos"]))
    chair.set("quat", yaw_quat(sp["yaw_deg"]))
    chair.set("childclass", "wheelchair")
    # Jaw pads (bit 16) meet the flat handle plates (bit 8). Handles also
    # accept world contacts (bit 1) so the chair cannot ghost through a wall.
    # The rest of the chair keeps the class mask (world/base/arms, not itself).
    handle_names = {f"wc_{side}_{part}" for side in ("left", "right")
                    for part in ("push_handle", "handle_grip", "handle_tip")}
    for geom in chair.iter("geom"):
        if geom.get("name") in handle_names:
            geom.set("contype", "8")
            geom.set("conaffinity", "17")
            geom.set("friction", "2.2 0.12 0.02")
            geom.set("condim", "3")
            geom.set("solref", "0.006 1")
            geom.set("solimp", "0.98 0.99 0.001")
    # Rigid body hull for walls/floor/base only (conaffinity bit 1). Covers
    # seat, wheels and footrests. Stops short of the handle tips (x=-0.41)
    # and below handle height so the arms can still pinch.
    ET.SubElement(chair, "geom", name="wc_hull", type="box",
                  pos="0.22 0 0.46", size="0.41 0.39 0.40",
                  rgba="0.2 0.2 0.25 0.0", group="3", contype="8", conaffinity="1",
                  friction="1.6 0.12 0.02", condim="3", margin="0.004",
                  solref="0.002 1", solimp="0.95 0.99 0.001")
    # The visible footrests reach the hull's front edge. These world-only
    # clearance shapes stop the hitch before a pedal can enter a wall.
    for side, y in (("left", 0.14), ("right", -0.14)):
        ET.SubElement(chair, "geom", name=f"wc_{side}_footrest_clearance",
                      type="box", pos=fmt(0.48, y, 0.17), size="0.18 0.11 0.025",
                      rgba="0 0 0 0", group="3", density="0",
                      contype="8", conaffinity="1", margin="0.003")
    for joint in chair.iter("joint"):
        name = joint.get("name") or ""
        if "base_free" in name:
            continue
        joint.set("damping", "8")
    wb.append(chair)

    # Constraints / actuators / sensors from both sources.
    equality = ET.SubElement(root, "equality")
    for eq in robot.find("equality"):
        equality.append(copy.deepcopy(eq))
    for side in ("right", "left"):
        ET.SubElement(equality, "connect", name=f"{side}_wheelchair_grasp",
                      site1=f"{side}_grasp", site2=f"wc_{side}_handle_grasp",
                      active="false")
    ET.SubElement(equality, "weld", name="chair_hitch",
                  body1="mobile_base", body2="wc_wheelchair",
                  active="false", relpose="0 0 0 1 0 0 0",
                  solref="0.004 1", solimp="0.9 0.95 0.001")

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
    ap.add_argument("--vision-config", help="camera calibration and room label configuration")
    args = ap.parse_args()

    out = build(obstacle=args.obstacle, config_path=args.vision_config)
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
