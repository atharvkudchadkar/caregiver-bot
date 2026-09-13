"""Apartment geometry for world construction, never imported by navigation.

Units: metres, world frame, z up. Originally derived from the teammate's
four_room_wheelchair_map_varied.xml, rebuilt so that every room has a real
doorway onto the hallway (in the original wall coordinates the hallway walls
were continuous and the "doorway" segments overlapped, so no room was
reachable), then enlarged (rooms, hallway, doorways) for easier towing
clearance.
"""

WALL_HALF_T = 0.075 * 1.5   # 15 cm thick walls, scaled up 1.5x
WALL_HALF_H = 0.6 * 1.5     # 1.2 m tall (scaled 1.5x): see into rooms from the overview camera
DOOR_W = 1.6   # widened from 1.2 for easier towing clearance
HALL_X = (-6.5, 7.0)      # hallway runs east-west along y = 0; matches the west/east
                          # rooms' own outer walls exactly (bedroom/bathroom's x0,
                          # kitchen/living_room's x1) so the corridor's end-cap walls
                          # sit flush with the building's true exterior - previously
                          # left at the old (-6.0, 6.0) after the rooms grew, which put
                          # the end caps *inside* the rooms' footprints instead of
                          # flush with them, leaving an actual 0.5m gap in the outer
                          # wall at each end that the robot could drive straight through
HALL_Y = (-1.25, 1.25)    # widened from +-0.75 for easier towing/turning clearance

# Uniform geometric scale of the complete robot and wheelchair. Robot camera
# offsets, wheel dimensions and clearance in vision_config.json must match.
ROBOT_SCALE = 0.7
CHAIR_SCALE = 0.6

# Direct pathways between a west room and an east room, in addition to the
# main hallway - alternate routes the robot can learn and, once it knows
# both, should prefer whichever is actually shorter (see VisionNavigator).
# Each bridge gets its own pair of localization-only codes (see
# bridge_marker_positions) flanking the doorway on the west room's side.
BRIDGE_W = 1.8   # wider than a plain doorway: these are corridors, not doorways
BRIDGES = {
    "bedroom_kitchen":        dict(west="bedroom",  east="kitchen",     y=2.75,  marker_ids=(8, 9)),
    "bathroom_living_room":   dict(west="bathroom", east="living_room", y=-3.3,  marker_ids=(10, 11)),
}

# Rooms north of the hallway have y[0] == HALL_Y[1]; rooms south have y[1] == HALL_Y[0].
# door_x is the centre of the doorway on the hallway wall.
ROOMS = {
    "bedroom":     dict(x=(-6.5, -1.5), y=(1.25, 5.25),    door_x=-3.5,  color=(0.2, 0.4, 0.9)),
    "kitchen":     dict(x=(1.0, 7.0),   y=(1.25, 5.75),    door_x=3.5,   color=(0.9, 0.3, 0.2)),
    "bathroom":    dict(x=(-6.5, -2.0), y=(-6.25, -1.25),  door_x=-3.75, color=(0.2, 0.8, 0.8)),
    "living_room": dict(x=(1.0, 7.0),   y=(-6.75, -1.25),  door_x=3.5,   color=(0.2, 0.8, 0.3)),
}

# Initial poses. Robot starts in the bedroom facing its door (south); the
# wheelchair sits out of the door lane, facing south (rear toward the north
# wall) rather than east (rear toward the west wall - not enough room for the
# robot to reach the standoff point behind it and grasp the handles; facing
# south leaves more room, and roughly faces the robot's own spawn instead of
# requiring a loop around the chair). Both shifted 1.1m east of their
# original position (same relative offset between them, so the grasp
# approach/standoff geometry is unaffected) after towing testing kept
# wedging the chair's hull against the west wall during early post-attach
# exploration, before the map is built up enough to route around it -
# 0.79m of clearance there wasn't enough; this leaves 1.89m.
SPAWN = {
    "robot":      dict(pos=(-2.4, 2.6, 0.005), yaw_deg=-90.0),
    "wheelchair": dict(pos=(-3.5, 2.2, -0.055), yaw_deg=-90.0),
}

# Optional test obstacle (build_world.py --obstacle): a 0.5 m crate in the
# hallway leaving a 0.8 m gap on its south side.
OBSTACLE = dict(pos=(0.0, 0.3, 0.3), size=(0.25, 0.25, 0.3))

# Collision bitmasks. World geoms (walls, floor, wheelchair, obstacle) accept
# both the base (bit 1) and the arms (bit 4); the arms only collide with world
# geoms, never with the robot itself.
COL_WORLD = dict(contype="1", conaffinity="5")
COL_BASE = dict(contype="1", conaffinity="1")
COL_ARM = dict(contype="4", conaffinity="0")
# Finger meshes collide with the wheelchair handles (bit 8/16), not with each
# other or the rest of the robot. conaffinity must stay 0: the two finger
# hulls overlap at rest, and 16&8 would make them explode on the first step.
COL_FINGER = dict(contype="16", conaffinity="0")


def is_north(room):
    return ROOMS[room]["y"][0] >= HALL_Y[1] - 1e-6


def room_center(room):
    r = ROOMS[room]
    return ((r["x"][0] + r["x"][1]) / 2, (r["y"][0] + r["y"][1]) / 2)


def door_wall_y(room):
    """Y of the wall shared with the hallway, i.e. the wall the door sits in."""
    r = ROOMS[room]
    return r["y"][0] if is_north(room) else r["y"][1]



# Two floor-mat markers per room, flanking the doorway instead of sitting in
# the middle of it, so the same room label is visible whether the robot is
# arriving (entrance) or leaving (exit) without blocking the opening itself.
MARKER_INSET = 0.4                  # how far into the room from the door wall
MARKER_SIDE_GAP = 0.3                # clearance beyond the door frame edge


def marker_positions(room):
    r = ROOMS[room]
    wall_y = door_wall_y(room)
    y = wall_y + MARKER_INSET if is_north(room) else wall_y - MARKER_INSET
    dx = DOOR_W / 2 + MARKER_SIDE_GAP
    return [(r["door_x"] - dx, y), (r["door_x"] + dx, y)]


def bridge_marker_positions(key, inset=0.35, wall_margin=0.1):
    """Two wall-mounted positions flanking bridge `key`'s doorway on its west
    room's side, on the solid wall segments to either side of the opening,
    facing into that room so they're seen on approach.

    Clamped to the west room's own y-range: if the doorway sits close to one
    of that room's other walls (as bathroom's does, for the
    bathroom_living_room bridge - its hallway-facing wall is only ~0.15m past
    the doorway edge), a fixed inset would overshoot straight through that
    wall and land the marker outside the room entirely (observed: floating
    inside the main hallway instead of mounted on the bathroom wall)."""
    b = BRIDGES[key]
    room = ROOMS[b["west"]]
    y0, y1 = room["y"]
    x = room["x"][1]
    gy0, gy1 = b["y"] - BRIDGE_W / 2, b["y"] + BRIDGE_W / 2
    lo = max(gy0 - inset, y0 + wall_margin)
    hi = min(gy1 + inset, y1 - wall_margin)
    return [(x, lo), (x, hi)]


def wall_segments():
    """Axis-aligned wall centre-lines as (name, x0, y0, x1, y1)."""
    segs = []
    hx0, hx1 = HALL_X
    hy0, hy1 = HALL_Y
    north_doors = sorted(r["door_x"] for n, r in ROOMS.items() if is_north(n))
    south_doors = sorted(r["door_x"] for n, r in ROOMS.items() if not is_north(n))
    for tag, y, doors in (("hall_north", hy1, north_doors), ("hall_south", hy0, south_doors)):
        x = hx0
        for k, dx in enumerate(doors):
            segs.append((f"{tag}_{k}", x, y, dx - DOOR_W / 2, y))
            x = dx + DOOR_W / 2
        segs.append((f"{tag}_{len(doors)}", x, y, hx1, y))
    segs.append(("hall_west", hx0, hy0, hx0, hy1))
    segs.append(("hall_east", hx1, hy0, hx1, hy1))
    # Each bridge punches a doorway-width gap in its west room's east wall and
    # its east room's west wall, keyed by room name so a room in two bridges
    # (not currently the case, but the code doesn't assume otherwise) would
    # just get two gaps.
    gaps = {}   # room -> {"east": (gy0, gy1)} and/or {"west": (gy0, gy1)}
    for bridge in BRIDGES.values():
        gy0, gy1 = bridge["y"] - BRIDGE_W / 2, bridge["y"] + BRIDGE_W / 2
        gaps.setdefault(bridge["west"], {})["east"] = (gy0, gy1)
        gaps.setdefault(bridge["east"], {})["west"] = (gy0, gy1)
    for name, r in ROOMS.items():
        x0, x1 = r["x"]
        y0, y1 = r["y"]
        room_gaps = gaps.get(name, {})
        if "east" in room_gaps:
            gy0, gy1 = room_gaps["east"]
            segs.append((f"{name}_east_0", x1, y0, x1, gy0))
            segs.append((f"{name}_east_1", x1, gy1, x1, y1))
        else:
            segs.append((f"{name}_east", x1, y0, x1, y1))
        if "west" in room_gaps:
            gy0, gy1 = room_gaps["west"]
            segs.append((f"{name}_west_0", x0, y0, x0, gy0))
            segs.append((f"{name}_west_1", x0, gy1, x0, y1))
        else:
            segs.append((f"{name}_west", x0, y0, x0, y1))
        far_y = y1 if is_north(name) else y0
        segs.append((f"{name}_far", x0, far_y, x1, far_y))
    # Corridor walls joining each bridge's two doorways directly.
    for key, bridge in BRIDGES.items():
        gy0, gy1 = bridge["y"] - BRIDGE_W / 2, bridge["y"] + BRIDGE_W / 2
        bx0, bx1 = ROOMS[bridge["west"]]["x"][1], ROOMS[bridge["east"]]["x"][0]
        segs.append((f"{key}_north", bx0, gy1, bx1, gy1))
        segs.append((f"{key}_south", bx0, gy0, bx1, gy0))
    return segs
