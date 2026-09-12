"""Compose world.xml: four-room apartment + wheelchair + the balancing robot.

Run:  python build_world.py
Then: python run_world.py --camera-preview        (viewer, arrow-key teleop)
      python run_world.py --goto kitchen          (find the kitchen by its door sign)
      python -m mujoco.viewer --mjcf world.xml    (passive look, no controller)

Sources merged:
  balance_robot_with_cameras.xml   robot with wheel motors, arm servos, and cameras
  assets/wheelchair.xml      teammate's wheelchair (renamed with a wc_ prefix)
  layout.py                  room/hallway geometry (walls generated here)
  vision_config.json         cameras + the sign catalogue (id -> room, side)

Every doorway gets two signs on the wall beside it, each hung to the right of
the door as you face it: one on the hallway face (the room's "outside" code)
and one on the room face ("inside"). Each sign carries an ArUco code and the
room's English name. The robot only knows which code means which room and the
signage convention (sign_door_offset); it has to see a sign to learn where it is.

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
ROBOT_XML = os.path.join(HERE, "balance_robot_with_cameras.xml")
WHEELCHAIR_XML = os.path.join(HERE, "assets", "wheelchair.xml")
SIGN_DIR = os.path.join(HERE, "assets", "markers")
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
        """Add the front rangefinder beams used by obstacle avoidance."""
        for i, deg in enumerate(layout.RF_ANGLES_DEG):
                angle = math.radians(deg)
                ET.SubElement(base, "site", name=f"rf_{i}", pos=fmt(0.12, 0, layout.RF_HEIGHT),
                                            zaxis=fmt(math.cos(angle), math.sin(angle), 0), size="0.01",
                                            rgba="0 1 0 0.3", group="3")
                ET.SubElement(sensor, "rangefinder", name=f"rf_{i}", site=f"rf_{i}",
                                            cutoff=fmt(layout.RF_CUTOFF))

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
        up = np.array([math.sin(pitch) * math.cos(yaw), math.sin(pitch) * math.sin(yaw), math.cos(pitch)])
        ET.SubElement(body, "camera", name=camera["name"],
                      pos=fmt(*(rotation.T @ camera["offset"])),
                      xyaxes=fmt(*(rotation.T @ right), *(rotation.T @ up)), fovy=str(config["fovy"]))


def sign_image(marker_id, room, config):
    """White sign, 1 px = 1 mm: ArUco code on the left, English room name on the right."""
    import cv2
    import numpy as np
    width, height = (int(round(v * 1000)) for v in config["sign_size"])
    marker = int(round(config["marker_size"] * 1000))
    pad = (height - marker) // 2
    # Perception undoes this code-vs-sign-centre offset; keep the config honest.
    offset = (pad + marker / 2 - width / 2) / 1000
    if abs(offset - config.get("marker_offset", 0.0)) > 0.005:
        raise ValueError(f"vision_config marker_offset should be {offset:.3f} for this sign layout")
    img = np.full((height, width, 3), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    img[pad:pad + marker, pad:pad + marker] = cv2.aruco.generateImageMarker(dictionary, marker_id, marker)[..., None]
    text = room.replace("_", " ").upper()
    font, scale, thick = cv2.FONT_HERSHEY_DUPLEX, 3.4, 9
    x0 = 2 * pad + marker
    avail = width - x0 - pad
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    while tw > avail and scale > 0.8:
        scale -= 0.2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.putText(img, text, (x0 + (avail - tw) // 2, (height + th) // 2), font, scale, (0, 0, 0), thick, cv2.LINE_AA)
    if config.get("sign_image_flip"):
        img = cv2.rotate(img, cv2.ROTATE_180)
    return img


def add_door_signs(asset, wb, config):
    """Two signs per doorway, beside the door: 'outside' faces the hall, 'inside' the room."""
    import cv2
    os.makedirs(SIGN_DIR, exist_ok=True)
    half_w, half_h = (v / 2 for v in config["sign_size"])
    lateral = float(config["sign_door_offset"])   # sign centre is this far to the viewer's right of the door
    for sid, info in config["signs"].items():
        room, side = info["room"], info["side"]
        if room not in layout.ROOMS:
            continue
        filename = f"sign_{sid}.png"
        cv2.imwrite(os.path.join(SIGN_DIR, filename), sign_image(int(sid), room, config))
        name = f"sign_{sid}"
        ET.SubElement(asset, "texture", name=name, type="2d", file=f"assets/markers/{filename}")
        ET.SubElement(asset, "material", name=name, texture=name, texrepeat="1 1", texuniform="false")
        dx, wy = layout.door_center(room)
        nx, ny = layout.door_normal_out(room)
        if side == "inside":
            nx, ny = -nx, -ny                           # face into the room
        # Plane normal (+z) toward the viewer; x axis = viewer's right = up x normal.
        rx, ry = -ny, nx
        gap = layout.WALL_HALF_T + 0.002
        ET.SubElement(wb, "geom", name=name, type="plane",
                      pos=fmt(dx + nx * gap + rx * lateral, wy + ny * gap + ry * lateral, layout.SIGN_Z),
                      size=fmt(half_w, half_h, 0.05), xyaxes=fmt(rx, ry, 0, 0, 0, 1),
                      material=name, contype="0", conaffinity="0")


def build(obstacle=False, config_path=None):
    config = load_config(config_path)
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
    # Chair geoms accept floor/base/arm contacts but do not collide with one
    # another (the source caster wheels otherwise intersect the footrests).
    ET.SubElement(wcd, "geom", density="300", friction="0.8 0.1 0.05", condim="3",
                  contype="8", conaffinity="5")
    ET.SubElement(wcd, "motor", ctrllimited="true", ctrlrange="-1 1")
    wall = ET.SubElement(default, "default", {"class": "wall"})
    # Clearly tinted so the floor-colour classifier never mistakes a lit wall for floor.
    ET.SubElement(wall, "geom", type="box", rgba="0.74 0.78 0.9 1", **layout.COL_WORLD)

    # Assets: robot meshes + a checker floor.
    asset = ET.SubElement(root, "asset")
    for mesh in robot.find("asset"):
        asset.append(copy.deepcopy(mesh))
    # Both checker colours are neutral grey: the floor detector keys on equal RGB
    # channels, and any tint would flicker in and out under shading/filtering.
    ET.SubElement(asset, "texture", name="wc_aruco_texture", type="2d",
                  file="assets/wheelchair_aruco_4x4_50_id0.png")
    ET.SubElement(asset, "material", name="wc_aruco_material", texture="wc_aruco_texture",
                      emission="1", specular="0", shininess="0")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker",
                  rgb1="0.78 0.78 0.78", rgb2="0.66 0.66 0.66", width="512", height="512")
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
        else:                     # wall running east-west
            size = (abs(x1 - x0) / 2 + layout.WALL_HALF_T, layout.WALL_HALF_T, layout.WALL_HALF_H)
        ET.SubElement(wb, "geom", {"class": "wall"}, name=f"wall_{name}",
                      pos=fmt(cx, cy, layout.WALL_HALF_H), size=fmt(*size))

    add_door_signs(asset, wb, config)

    # Robot: the whole mobile_base subtree, plus its cameras and collision proxies.
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
    pads = (
        ("right", "lower", "left_finger__left_finger", "-0.0422 -0.0107 -0.0295",
         "-0.70157 -0.02082 0.71229 0.70952 0.07235 0.70096"),
        ("right", "upper", "right_finger__right_finger", "0.0437 0.0053 0.0313",
         "0.70165 -0.01815 -0.71229 0.70979 -0.06966 0.70096"),
        ("left", "lower", "l_left_finger__left_finger", "0.0536 -0.0087 0.0201",
         "0.57682 -0.33612 -0.74451 0.36424 -0.70996 0.60273"),
        ("left", "upper", "l_right_finger__right_finger", "-0.0249 0.0066 -0.0031",
         "-0.64231 0.18204 0.74451 0.76395 0.23045 0.60273"),
    )
    for side, jaw, body_name, pos, xyaxes in pads:
        finger = base.find(f".//body[@name='{body_name}']")
        ET.SubElement(finger, "geom", name=f"{side}_{jaw}_pad", type="box",
                      pos=pos, xyaxes=xyaxes, size="0.022 0.013 0.006",
                      rgba="0.10 0.10 0.12 1", mass="0.001",
                      friction="1.8 0.08 0.01", condim="3",
                      contype="16", conaffinity="8")
        ET.SubElement(finger, "site", name=f"{side}_{jaw}_pad_site", pos=pos,
                      xyaxes=xyaxes, size="0.003", rgba="1 0.6 0 1")
    sensor = ET.Element("sensor")   # appended to root below
    ET.SubElement(sensor, "framequat", name="base_orientation", objtype="body", objname="mobile_base")
    add_lidar(base, sensor)
    wb.append(base)

    # Wheelchair: prefixed so its joint names don't collide with the robot's wheels.
    chair = copy.deepcopy(wc.find("worldbody/body[@name='wheelchair']"))
    prefix_names(eulers_to_radians(chair), WC_PREFIX)
    sp = layout.SPAWN["wheelchair"]
    chair.set("pos", fmt(*sp["pos"]))
    chair.set("quat", yaw_quat(sp["yaw_deg"]))
    chair.set("childclass", "wheelchair")
    # Let the arm meshes interact with the handle tubes, while the chair's
    # broad seat/backrest stay out of the path of the prescribed reach.
    handle_names = {f"wc_{side}_{part}" for side in ("left", "right")
                    for part in ("push_handle", "handle_grip")}
    for geom in chair.iter("geom"):
        if geom.get("name") in handle_names:
            geom.set("contype", "8")
            geom.set("conaffinity", "16")
            geom.set("friction", "1.8 0.08 0.01")
            geom.set("condim", "3")
        else:
            geom.set("conaffinity", "1")
    wb.append(chair)

    # Constraints / actuators / sensors from both sources.
    equality = ET.SubElement(root, "equality")
    for eq in robot.find("equality"):
        equality.append(copy.deepcopy(eq))
    for side in ("right", "left"):
        ET.SubElement(equality, "connect", name=f"{side}_wheelchair_grasp",
                      site1=f"{side}_grasp", site2=f"wc_{side}_handle_grasp",
                      active="false")

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
    ap.add_argument("--vision-config", help="camera calibration and room sign configuration")
    args = ap.parse_args()

    out = build(obstacle=args.obstacle, config_path=args.vision_config)
    import mujoco
    model = mujoco.MjModel.from_xml_path(out)
    print(f"wrote {out}" + (" (with hallway obstacle)" if args.obstacle else ""))
    print(f"  bodies={model.nbody} joints={model.njnt} geoms={model.ngeom} "
          f"actuators={model.nu} sensors={model.nsensor} cameras={model.ncam}")
    for name in ("mobile_base", f"{WC_PREFIX}wheelchair"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        print(f"  {name:16s} body id {bid}")
    print("  rooms:", ", ".join(layout.ROOMS))
    print(f"  door signs: {len([g for g in range(model.ngeom) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or '').startswith('sign_')])} "
          f"(images in {os.path.relpath(SIGN_DIR, HERE)})")


if __name__ == "__main__":
    main()
