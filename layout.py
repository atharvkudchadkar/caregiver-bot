"""Apartment geometry for world construction, never imported by navigation.

Units: metres, world frame, z up. Derived from the teammate's
four_room_wheelchair_map_varied.xml (same four room sizes, same 1.5 m
east-west hallway), rebuilt so that every room has a real 1.2 m doorway
onto the hallway. In the original wall coordinates the hallway walls were
continuous and the "doorway" segments overlapped, so no room was reachable.
"""

WALL_HALF_T = 0.075 * 1.5   # 15 cm thick walls, scaled up 1.5x
WALL_HALF_H = 0.6 * 1.5     # 1.2 m tall (scaled 1.5x): see into rooms from the overview camera
DOOR_W = 1.2
HALL_X = (-6.0, 6.0)      # hallway runs east-west along y = 0
HALL_Y = (-0.75, 0.75)

# Direct pathways between a west room and an east room, in addition to the
# main hallway - alternate routes the robot can learn and, once it knows
# both, should prefer whichever is actually shorter (see VisionNavigator).
# Each bridge gets its own pair of localization-only codes (see
# bridge_marker_positions) flanking the doorway on the west room's side.
BRIDGE_W = DOOR_W
BRIDGES = {
    "bedroom_kitchen":        dict(west="bedroom",  east="kitchen",     y=2.0,  marker_ids=(8, 9)),
    "bathroom_living_room":   dict(west="bathroom", east="living_room", y=-1.8, marker_ids=(10, 11)),
}

# Rooms north of the hallway have y[0] == HALL_Y[1]; rooms south have y[1] == HALL_Y[0].
# door_x is the centre of the doorway on the hallway wall.
ROOMS = {
    "bedroom":     dict(x=(-5.5, -1.5), y=(0.75, 3.75),   door_x=-3.5,  color=(0.2, 0.4, 0.9)),
    "kitchen":     dict(x=(1.0, 6.0),   y=(0.75, 4.25),   door_x=3.5,   color=(0.9, 0.3, 0.2)),
    "bathroom":    dict(x=(-5.5, -2.0), y=(-4.75, -0.75), door_x=-3.75, color=(0.2, 0.8, 0.8)),
    "living_room": dict(x=(1.0, 6.0),   y=(-5.25, -0.75), door_x=3.5,   color=(0.2, 0.8, 0.3)),
}

# Initial poses. Robot starts in the bedroom facing its door (south); the
# wheelchair sits out of the door lane, facing south (rear toward the north
# wall) rather than east (rear toward the west wall, only 0.9m clear - not
# enough room for the robot to reach the standoff point behind it and grasp
# the handles; facing south leaves 1.55m, and roughly faces the robot's own
# spawn instead of requiring a loop around the chair).
SPAWN = {
    "robot":      dict(pos=(-3.5, 2.6, 0.005), yaw_deg=-90.0),
    "wheelchair": dict(pos=(-4.6, 2.2, -0.055), yaw_deg=-90.0),
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


def far_wall_y(room):
    """Y of the room's outer wall, opposite the door."""
    r = ROOMS[room]
    return r["y"][1] if is_north(room) else r["y"][0]


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


# --- Furniture ---------------------------------------------------------
# Simple primitive props (boxes/cylinders), kept well clear of the doorway,
# the flanking markers above, and the robot/wheelchair spawns. All positions
# are derived from ROOMS/SPAWN so they track any future layout edits.

FURN_COLOR = dict(
    wood=(0.55, 0.4, 0.25, 1), wood_dark=(0.4, 0.28, 0.18, 1),
    counter=(0.75, 0.68, 0.56, 1), fabric=(0.32, 0.38, 0.55, 1),
    fabric_dark=(0.25, 0.3, 0.45, 1), screen=(0.05, 0.05, 0.05, 1),
    # porcelain/linen are light on purpose, but kept a good margin (>20/255
    # channel spread) from the near-neutral-gray floor colour classifier in
    # vision_navigation.py, so they never get mistaken for walkable floor.
    porcelain=(0.95, 0.92, 0.83, 1), linen=(0.92, 0.87, 0.75, 1),
)


def _box(name, pos, size, color):
    return dict(name=name, type="box", pos=tuple(pos), size=tuple(size), rgba=FURN_COLOR[color])


def _cyl(name, pos, size, color):
    return dict(name=name, type="cylinder", pos=tuple(pos), size=tuple(size), rgba=FURN_COLOR[color])


def furniture():
    """name -> list of primitive geom specs (pos/size are half-extents, metres)."""
    items = {}

    # Bedroom: a bed against the far (north) wall, east of the door and both
    # spawns, plus a nightstand at its head. Sized/placed to stay clear of the
    # bedroom/kitchen bridge doorway (bedroom_kitchen bridge, on the east wall).
    r = ROOMS["bedroom"]
    bx = r["x"][1] - 0.7                       # east side, clear of the door lane
    by = far_wall_y("bedroom") - WALL_HALF_T - 0.32 - 0.05
    items["bedroom"] = [
        _box("bed_frame", (bx, by, 0.12), (0.4, 0.32, 0.12), "wood_dark"),
        _box("bed_mattress", (bx, by, 0.32), (0.35, 0.28, 0.08), "linen"),
        _box("bed_pillow", (bx, by + 0.2, 0.45), (0.25, 0.08, 0.05), "linen"),
        _box("nightstand", (bx - 0.75, by + 0.1, 0.25), (0.2, 0.2, 0.25), "wood"),
    ]

    # Kitchen: three counter/cabinet units along the far wall, a table with
    # legs set back from the door far enough to clear both markers.
    r = ROOMS["kitchen"]
    cy = far_wall_y("kitchen") - 0.075 - 0.3 - 0.01
    items["kitchen"] = [
        _box("cabinet_1", (r["x"][0] + 1.0, cy, 0.45), (0.65, 0.3, 0.45), "counter"),
        _box("cabinet_2", (r["door_x"],     cy, 0.45), (0.65, 0.3, 0.45), "counter"),
        _box("cabinet_3", (r["x"][1] - 1.0, cy, 0.45), (0.65, 0.3, 0.45), "counter"),
        _box("table_top", (r["door_x"], door_wall_y("kitchen") + 1.25, 0.4), (0.5, 0.35, 0.03), "wood"),
        *[_cyl(f"table_leg_{i}",
               (r["door_x"] + sx * 0.45, door_wall_y("kitchen") + 1.25 + sy * 0.3, 0.2),
               (0.025, 0.2), "wood")
          for i, (sx, sy) in enumerate([(-1, -1), (1, -1), (-1, 1), (1, 1)])],
    ]

    # Living room: a couch against the far (south) wall facing the door, a
    # wall-mounted TV on the west wall so it doesn't compete for floor space.
    r = ROOMS["living_room"]
    sofa_y = far_wall_y("living_room") + 0.075 + 0.08 + 0.02   # backrest, against the wall
    seat_y = sofa_y + 0.08 + 0.4
    items["living_room"] = [
        _box("couch_back", (r["door_x"], sofa_y, 0.35), (1.0, 0.08, 0.35), "fabric_dark"),
        _box("couch_seat", (r["door_x"], seat_y, 0.2), (1.0, 0.4, 0.2), "fabric"),
        _box("tv", (r["x"][0] + 0.115, -3.0, 1.3), (0.03, 0.5, 0.3), "screen"),
    ]

    # Bathroom: a toilet (tank + bowl + seat) against the far wall, off to
    # one side.
    r = ROOMS["bathroom"]
    tx = r["x"][1] - 0.5
    tank_y = far_wall_y("bathroom") + 0.075 + 0.1 + 0.02
    bowl_y = tank_y + 0.1 + 0.19
    items["bathroom"] = [
        _box("toilet_tank", (tx, tank_y, 0.5), (0.18, 0.1, 0.2), "porcelain"),
        _cyl("toilet_bowl", (tx, bowl_y, 0.19), (0.19, 0.19), "porcelain"),
        _box("toilet_seat", (tx, bowl_y, 0.4), (0.19, 0.22, 0.02), "porcelain"),
    ]
    return items


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
