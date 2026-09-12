"""Apartment layout: single source of truth for build_world.py and navigation.

Units: metres, world frame, z up. Derived from the teammate's
four_room_wheelchair_map_varied.xml (same four room sizes, same 1.5 m
east-west hallway), rebuilt so that every room has a real 1.2 m doorway
onto the hallway. In the original wall coordinates the hallway walls were
continuous and the "doorway" segments overlapped, so no room was reachable.
"""
import heapq
import math

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

# Aliases the voice pipeline may produce -> canonical room name.
ROOM_ALIASES = {
    "living room": "living_room", "lounge": "living_room", "couch": "living_room",
    "toilet": "bathroom", "washroom": "bathroom", "restroom": "bathroom",
    "bed": "bedroom", "my room": "bedroom",
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

# Simulated lidar on the robot: one rangefinder per angle (degrees, 0 = forward).
RF_ANGLES_DEG = list(range(-60, 61, 15))
RF_HEIGHT = 0.30
RF_CUTOFF = 4.0

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


def canonical_room(text):
    """Map free text ('the living room', 'kitchen please') to a room name or None."""
    t = text.lower().replace("_", " ").strip()
    for alias, name in ROOM_ALIASES.items():
        if alias in t:
            return name
    for name in ROOMS:
        if name.replace("_", " ") in t:
            return name
    return None


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


# --------------------------------------------------------------------------- #
# Waypoint graph for navigation
# --------------------------------------------------------------------------- #
def graph():
    """Nodes: room centres, a point just inside each door, a hallway point
    outside each door. Returns (nodes: name -> (x, y), edges: name -> [names])."""
    nodes, edges = {}, {}

    def link(a, b):
        edges.setdefault(a, []).append(b)
        edges.setdefault(b, []).append(a)

    hall_nodes = []
    for name, r in ROOMS.items():
        cx, cy = room_center(name)
        inside_y = HALL_Y[1] + 0.9 if is_north(name) else HALL_Y[0] - 0.9
        nodes[name] = (cx, cy)
        nodes[f"{name}_door"] = (r["door_x"], inside_y)
        nodes[f"{name}_hall"] = (r["door_x"], 0.0)
        link(name, f"{name}_door")
        link(f"{name}_door", f"{name}_hall")
        hall_nodes.append(f"{name}_hall")
    # Chain the hallway points west -> east (kitchen and living room share x=3.5;
    # that zero-length edge is harmless and keeps the graph connected).
    hall_nodes.sort(key=lambda n: nodes[n][0])
    for a, b in zip(hall_nodes, hall_nodes[1:]):
        link(a, b)
    return nodes, edges


def region_of(xy):
    """Room name, 'hall', or None for a point."""
    x, y = xy
    for name, r in ROOMS.items():
        if r["x"][0] <= x <= r["x"][1] and r["y"][0] <= y <= r["y"][1]:
            return name
    if HALL_X[0] <= x <= HALL_X[1] and HALL_Y[0] <= y <= HALL_Y[1]:
        return "hall"
    return None


def route(start_xy, room):
    """List of (x, y) waypoints from start_xy to the centre of `room`."""
    if room not in ROOMS:
        raise KeyError(f"unknown room {room!r}; choose from {list(ROOMS)}")
    nodes, edges = graph()
    region = region_of(start_xy)
    if region == "hall":
        candidates = [n for n in nodes if n.endswith("_hall")]
    elif region in ROOMS:
        candidates = [region, f"{region}_door"]
    else:
        candidates = list(nodes)
    start = min(candidates, key=lambda n: math.dist(nodes[n], start_xy))

    dist = {start: 0.0}
    prev = {}
    pq = [(0.0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == room:
            break
        if d > dist.get(u, math.inf):
            continue
        for v in edges.get(u, []):
            nd = d + math.dist(nodes[u], nodes[v])
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    path, n = [], room
    while n != start:
        path.append(nodes[n])
        n = prev[n]
    path.append(nodes[start])
    path.reverse()
    # Drop the first waypoint if we are already on top of it.
    if len(path) > 1 and math.dist(path[0], start_xy) < 0.3:
        path = path[1:]
    return path


def step_toward(pose, target, v_max=0.3, w_max=0.6, tol=0.15):
    """Unicycle P-controller. pose=(x, y, yaw). Returns (v, w, arrived)."""
    x, y, yaw = pose
    dx, dy = target[0] - x, target[1] - y
    dist = math.hypot(dx, dy)
    if dist < tol:
        return 0.0, 0.0, True
    err = (math.atan2(dy, dx) - yaw + math.pi) % (2 * math.pi) - math.pi
    v = min(v_max, 0.6 * dist) if abs(err) < 0.4 else 0.0   # turn first, then drive
    w = max(-w_max, min(w_max, 1.5 * err))
    return v, w, False


if __name__ == "__main__":
    for s in wall_segments():
        print(s)
    print(route(SPAWN["robot"]["pos"][:2], "kitchen"))
