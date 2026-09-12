"""Apartment geometry for world construction, never imported by navigation.

Units: metres, world frame, z up. Derived from the teammate's
four_room_wheelchair_map_varied.xml (same four room sizes, same 1.5 m
east-west hallway), rebuilt so that every room has a real 1.2 m doorway
onto the hallway. In the original wall coordinates the hallway walls were
continuous and the "doorway" segments overlapped, so no room was reachable.
"""

WALL_HALF_T = 0.075       # 15 cm thick walls
WALL_HALF_H = 0.6         # 1.2 m tall: you can see into rooms from the overview camera
DOOR_W = 1.2
HALL_X = (-6.0, 6.0)      # hallway runs east-west along y = 0
HALL_Y = (-0.75, 0.75)

# Rooms north of the hallway have y[0] == HALL_Y[1]; rooms south have y[1] == HALL_Y[0].
# door_x is the centre of the doorway on the hallway wall.
ROOMS = {
    "bedroom":     dict(x=(-5.5, -1.5), y=(0.75, 3.75),   door_x=-3.5,  color=(0.2, 0.4, 0.9)),
    "kitchen":     dict(x=(1.0, 6.0),   y=(0.75, 4.25),   door_x=3.5,   color=(0.9, 0.3, 0.2)),
    "bathroom":    dict(x=(-5.5, -2.0), y=(-4.75, -0.75), door_x=-3.75, color=(0.2, 0.8, 0.8)),
    "living_room": dict(x=(1.0, 6.0),   y=(-5.25, -0.75), door_x=3.5,   color=(0.2, 0.8, 0.3)),
}

# Initial poses. Robot starts in the bedroom facing its door (south), the
# wheelchair sits against the west wall out of the door lane.
SPAWN = {
    "robot":      dict(pos=(-3.5, 2.6, 0.005), yaw_deg=-90.0),
    "wheelchair": dict(pos=(-4.6, 2.2, -0.055), yaw_deg=0.0),
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


def is_north(room):
    return ROOMS[room]["y"][0] >= HALL_Y[1] - 1e-6


def room_center(room):
    r = ROOMS[room]
    return ((r["x"][0] + r["x"][1]) / 2, (r["y"][0] + r["y"][1]) / 2)


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
    for name, r in ROOMS.items():
        x0, x1 = r["x"]
        y0, y1 = r["y"]
        segs.append((f"{name}_west", x0, y0, x0, y1))
        segs.append((f"{name}_east", x1, y0, x1, y1))
        far_y = y1 if is_north(name) else y0
        segs.append((f"{name}_far", x0, far_y, x1, far_y))
    return segs
