"""RGB floor mapping, door-sign localization and persistent map memory.

Independent of MuJoCo and of the world builder. Inputs are RGB frames with
calibrated camera transforms, wheel odometry, and the sign catalogue from
vision_config.json (sign id -> room, side). Distances to the floor are
ground-plane projections; distances to signs come from PnP on the known
marker size. Neither is learned depth.

Frames:
  odom  wheel-encoder frame of the current run, (0, 0, 0) at start-up
  map   frame of the saved memory: the odom frame of the first run ever
A door sign that is both visible now and stored in memory gives the rigid
transform odom -> map. Until then the robot works in a fresh local map.
"""
import heapq
import json
import math
import os

import cv2
import numpy as np


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def local_to_map(points, pose):
    """Robot-relative (x forward, y left) points -> the frame `pose` is expressed in."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return np.asarray(points) @ np.array([[c, s], [-s, c]]) + pose[:2]


class Frame2D:
    """Rigid 2D transform: p_map = R(theta) p_odom + t."""

    def __init__(self, theta=0.0, t=(0.0, 0.0)):
        self.theta = float(theta)
        self.t = np.asarray(t, float)

    def R(self):
        c, s = math.cos(self.theta), math.sin(self.theta)
        return np.array([[c, -s], [s, c]])

    def apply(self, points):
        return np.asarray(points, float) @ self.R().T + self.t

    def rotate(self, v):
        return self.R() @ np.asarray(v, float)

    def apply_pose(self, pose):
        x, y = self.apply(np.asarray(pose[:2], float)[None])[0]
        return np.array([x, y, wrap(pose[2] + self.theta)])

    @classmethod
    def from_landmark(cls, obs_pos, obs_normal, mem_pos, mem_normal):
        """Transform that carries an observed sign (odom) onto its stored copy (map)."""
        theta = wrap(math.atan2(mem_normal[1], mem_normal[0]) - math.atan2(obs_normal[1], obs_normal[0]))
        f = cls(theta)
        f.t = np.asarray(mem_pos, float) - f.R() @ np.asarray(obs_pos, float)
        return f

    def blend(self, other, alpha):
        theta = wrap(self.theta + alpha * wrap(other.theta - self.theta))
        return Frame2D(theta, self.t + alpha * (other.t - self.t))


def ground_points(frame, pixels):
    """Intersect calibrated RGB pixel rays with z=0, rejecting the horizon."""
    pixels = np.asarray(pixels).reshape(-1, 2)
    rays = np.column_stack(((pixels[:, 0] - frame.K[0, 2]) / frame.K[0, 0],
                            -(pixels[:, 1] - frame.K[1, 2]) / frame.K[1, 1],
                            -np.ones(len(pixels)))) @ frame.rotation.T
    valid = (rays[:, 2] < -0.05) & (frame.origin[2] > 0.1)
    scale = np.divide(-frame.origin[2], rays[:, 2], out=np.zeros(len(rays)), where=valid)
    points = frame.origin + rays * scale[:, None]
    return points[:, :2], valid


class SignDetector:
    """ArUco door signs -> position (floor-projected), facing normal, room and side."""

    def __init__(self, config):
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())
        self.signs = {int(k): v for k, v in config["signs"].items()}
        self.marker_size = float(config["marker_size"])
        # The code sits left of the room name, so the code centre is offset from
        # the sign (= door) centre along the viewer's right axis.
        self.marker_offset = float(config.get("marker_offset", 0.0))
        # Signage convention: the sign hangs this far to the viewer's right of the door.
        self.sign_door_offset = float(config.get("sign_door_offset", 0.0))
        self.max_range = float(config["max_range"])
        half = self.marker_size / 2
        # ArUco corner order is TL, TR, BR, BL; IPPE_SQUARE expects the same.
        self.objp = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], np.float64)

    def detect(self, frame):
        out = []
        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY))
        if ids is None:
            return out
        for corner, marker_id in zip(corners, ids.ravel()):
            info = self.signs.get(int(marker_id))
            if info is None:
                continue
            ok, rvec, tvec = cv2.solvePnP(self.objp, corner[0].astype(np.float64),
                                          frame.K.astype(np.float64), None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.ravel()
            n = R[:, 2]
            if np.dot(n, t) > 0:          # the sign's face must point at the camera
                n = -n
            # OpenCV camera axes (x right, y down, z forward) -> OpenGL (x right, y up, z back)
            p = frame.origin + frame.rotation @ np.array([t[0], -t[1], -t[2]])
            nb = frame.rotation @ np.array([n[0], -n[1], -n[2]])
            horiz = float(np.linalg.norm(nb[:2]))
            if horiz < 0.5 or p[2] < 0.5:     # door signs hang on walls and face sideways
                continue
            normal = nb[:2] / horiz
            right = np.array([-normal[1], normal[0]])          # viewer's right = up x normal
            # code centre -> sign centre -> door centre, all along the wall
            centre = p[:2] - (self.marker_offset + self.sign_door_offset) * right
            dist = float(np.linalg.norm(centre))
            if not (0.3 < dist < self.max_range):
                continue
            out.append(dict(id=int(marker_id), room=info["room"], side=info["side"],
                            position=centre, height=float(p[2]), normal=normal,
                            distance=dist, camera=frame.name))
        return out


class FloorVision:
    def __init__(self, config):
        self.config = config
        self.signs = SignDetector(config)
        self.masks = {}

    def observe(self, frames):
        """-> (free points, blocked points, sign observations), all robot-relative."""
        free, blocked, signs = [], [], []
        for frame in frames:
            rgb = frame.rgb
            signs.extend(self.signs.detect(frame))
            # No simulator segmentation or depth buffers. This colour classifier
            # is intentionally explicit and configurable, not semantic room AI.
            body = frame.body if frame.body is not None else np.zeros(rgb.shape[:2], bool)
            mask = ((np.ptp(rgb.astype(np.int16), axis=2) <= self.config["floor_channel_spread"])
                    & (rgb.min(axis=2) >= self.config["floor_min_value"]) & ~body).astype(np.uint8)
            # Restrict to substantial connected patches; discard speckles/highlights.
            count, components, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            keep = np.zeros(count, np.uint8)
            keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= 70)
            mask = keep[components]
            # A high-mounted camera can see floor OVER a short wall. Keep only
            # the uninterrupted ground run from the image bottom in each column;
            # distant floor behind an occluding object must not clear that object.
            # The robot's own arms/grippers/wheels in view are neither floor nor an
            # obstacle: the run passes through them, but they never count as floor.
            passable = np.flip(np.logical_and.accumulate(np.flip(mask.astype(bool) | body, axis=0), axis=0), axis=0)
            mask = (passable & ~body & mask.astype(bool)).astype(np.uint8)
            self.masks[frame.name] = mask
            h, w = mask.shape
            yy, xx = np.mgrid[0:h:2, 0:w:2]
            pixels = np.column_stack((xx.ravel(), yy.ravel()))
            points, valid = ground_points(frame, pixels)
            distance = np.linalg.norm(points, axis=1)
            valid &= (distance >= 0.3) & (distance <= self.config["max_range"])
            floor = mask[yy, xx].ravel().astype(bool)
            free.append(points[valid & floor])
            # Ground contact is the bottom silhouette, not every vertical edge
            # of an object (those edges would project false obstacles into doors).
            run = passable.sum(axis=0)
            columns = np.flatnonzero((run > 0) & (run < h))
            contact = np.column_stack((columns, h - run[columns] - 1))
            hits, hit_valid = ground_points(frame, contact)
            hit_distance = np.linalg.norm(hits, axis=1)
            hit_valid &= (hit_distance >= 0.3) & (hit_distance <= self.config["max_range"])
            blocked.append(hits[hit_valid])
        return (np.concatenate(free) if free else np.empty((0, 2)),
                np.concatenate(blocked) if blocked else np.empty((0, 2)), signs)


class ObservedMap:
    """Fixed-size occupancy grid. Unknown space is never traversable."""

    def __init__(self, config):
        self.resolution = config["resolution"]
        self.size = int(math.ceil(config["map_size"] / self.resolution))
        self.offset = self.size // 2
        self.radius = config["robot_radius"]
        self.evidence = np.zeros((self.size, self.size), np.float32)
        self.seen = np.zeros_like(self.evidence, bool)
        self.visits = np.zeros_like(self.evidence)
        self._traversable = None
        # The robot's own footprint at the frame origin is free by construction.
        yy, xx = np.mgrid[:self.size, :self.size]
        footprint = np.hypot(xx - self.offset, yy - self.offset) * self.resolution <= self.radius + 0.1
        self.evidence[footprint] = -3
        self.seen[footprint] = True

    def cells(self, points):
        return np.floor(np.asarray(points) / self.resolution).astype(int) + self.offset

    def xy(self, cell):
        return (np.asarray(cell) - self.offset + 0.5) * self.resolution

    def inside(self, cell):
        return 0 <= cell[0] < self.size and 0 <= cell[1] < self.size

    def known_fraction(self):
        return float(self.seen.mean())

    def integrate(self, free, blocked, pose):
        """free/blocked are robot-relative; pose is the robot in this grid's frame."""
        self._traversable = None
        # Count a cell once per observation, not once per pixel. Obstacles win
        # conflicting observations; later unobstructed observations can clear them.
        for points, value in ((free, -1.0), (blocked, 2.5)):
            if not len(points):
                continue
            cells = np.unique(self.cells(local_to_map(points, pose)), axis=0)
            cells = cells[((cells >= 0) & (cells < self.size)).all(axis=1)]
            x, y = cells.T
            old = self.evidence[y, x]
            self.evidence[y, x] = np.clip((np.maximum(old, 0) if value > 0 else old) + value, -4, 5)
            self.seen[y, x] = True
        x, y = self.cells(pose[:2])
        if self.inside((x, y)):
            self.visits[y, x] += 1

    def traversable(self):
        if self._traversable is not None:
            return self._traversable
        occupied = ((self.evidence >= 0) | ~self.seen).astype(np.uint8)
        radius = int(math.ceil(self.radius / self.resolution))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        inflated = cv2.dilate(occupied, kernel).astype(bool)
        self._traversable = self.seen & (self.evidence < 0) & ~inflated
        return self._traversable

    def route(self, start_xy, goal_xy=None):
        """Dijkstra over observed free cells; otherwise choose a reachable frontier."""
        start = tuple(self.cells(start_xy))
        free = self.traversable()
        if not self.inside(start) or not free[start[1], start[0]]:
            return []
        goal = tuple(self.cells(goal_xy)) if goal_xy is not None else None
        frontier_width = 2 * (int(math.ceil(self.radius / self.resolution)) + 2) + 1
        unknown_near = cv2.dilate((~self.seen).astype(np.uint8),
                                  np.ones((frontier_width, frontier_width), np.uint8)) > 0
        frontier = free & unknown_near
        cost, previous = {start: 0.0}, {}
        queue = [(0.0, start)]
        best, best_score = None, -math.inf
        while queue:
            distance, u = heapq.heappop(queue)
            if distance != cost[u]:
                continue
            if goal is not None and u == goal:
                best = u
                break
            x, y = u
            if frontier[y, x] and distance > 0.5:
                # Prefer nearby unexplored views; avoid repeatedly selecting
                # already visited viewpoints. A known goal biases exploration.
                score = -distance - 0.5 * self.visits[y, x]
                if goal_xy is not None:
                    score -= 2 * np.linalg.norm(self.xy(u) - goal_xy)
                if score > best_score:
                    best, best_score = u, score
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                v = (x + dx, y + dy)
                if not self.inside(v) or not free[v[1], v[0]]:
                    continue
                if dx and dy and (not free[y, x + dx] or not free[y + dy, x]):
                    continue
                candidate = distance + self.resolution * math.hypot(dx, dy)
                if candidate < cost.get(v, math.inf):
                    cost[v], previous[v] = candidate, u
                    heapq.heappush(queue, (candidate, v))
        if best is None:
            return []
        path = [best]
        while path[-1] != start:
            path.append(previous[path[-1]])
        return [self.xy(cell) for cell in reversed(path)]

    def safe_segment(self, start, end):
        free = self.traversable()
        n = max(2, int(np.linalg.norm(np.asarray(end) - start) / self.resolution * 2) + 1)
        cells = self.cells(np.linspace(start, end, n))
        if not ((cells >= 0) & (cells < self.size)).all():
            return False
        return bool(free[cells[:, 1], cells[:, 0]].all())


class MapMemory:
    """What the robot has learned about the building, kept between runs.

    * grid       occupancy evidence in the map frame (learned by driving)
    * landmarks  door signs: id -> room, side, position, facing normal
    * rooms      per room: door centre, inward direction, entry point
    Nothing here is authored; it is only ever written from observations.
    Saved as a single .npz (arrays + a JSON metadata string).
    """

    VERSION = 1

    def __init__(self, config, path=None):
        self.config = config
        self.path = path
        self.grid = ObservedMap(config)
        self.landmarks = {}
        self.rooms = {}
        self.runs = 0
        self.loaded = False
        if path and os.path.exists(path):
            self.load()

    # -- persistence --------------------------------------------------------
    def load(self):
        with np.load(self.path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            compatible = (meta.get("version") == self.VERSION
                          and tuple(z["evidence"].shape) == self.grid.evidence.shape
                          and abs(meta.get("resolution", -1) - self.grid.resolution) < 1e-9)
            if not compatible:
                raise ValueError(f"{self.path} was saved with a different map size/resolution; "
                                 "delete it or run with --forget")
            self.grid.evidence[:] = z["evidence"]
            self.grid.seen[:] = z["seen"]
            self.grid.visits[:] = z["visits"] * 0.5     # old viewpoints matter less each run
            self.grid._traversable = None
        self.landmarks = {int(k): dict(v, pos=np.array(v["pos"], float), normal=np.array(v["normal"], float))
                          for k, v in meta["landmarks"].items()}
        self.rooms = {k: {kk: (np.array(vv, float) if isinstance(vv, list) else vv) for kk, vv in v.items()}
                      for k, v in meta["rooms"].items()}
        self.runs = int(meta.get("runs", 0))
        self.loaded = True

    def save(self):
        if not self.path:
            return False
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        meta = dict(version=self.VERSION, resolution=self.grid.resolution,
                    map_size=self.config["map_size"], runs=self.runs + 1,
                    landmarks={str(k): dict(v, pos=[float(x) for x in v["pos"]],
                                            normal=[float(x) for x in v["normal"]])
                               for k, v in self.landmarks.items()},
                    rooms={k: {kk: ([float(x) for x in vv] if isinstance(vv, np.ndarray) else vv)
                               for kk, vv in v.items()} for k, v in self.rooms.items()})
        np.savez_compressed(self.path, evidence=self.grid.evidence, seen=self.grid.seen,
                            visits=self.grid.visits, meta=json.dumps(meta))
        return True

    # -- learning -----------------------------------------------------------
    def update_landmark(self, sign_id, room, side, pos, normal):
        """Running average of a sign's map-frame pose. Returns True if it was new."""
        lm = self.landmarks.get(sign_id)
        if lm is None:
            self.landmarks[sign_id] = dict(room=room, side=side, pos=np.asarray(pos, float).copy(),
                                           normal=np.asarray(normal, float).copy(), count=1)
            return True
        k = min(lm["count"], 20)
        lm["pos"] = (lm["pos"] * k + pos) / (k + 1)
        n = lm["normal"] * k + normal
        lm["normal"] = n / max(np.linalg.norm(n), 1e-9)
        lm["count"] += 1
        return False

    def update_room(self, room):
        """Door centre / inward direction / entry point from that room's signs."""
        signs = [lm for lm in self.landmarks.values() if lm["room"] == room]
        if not signs:
            return
        door = np.mean([lm["pos"] for lm in signs], axis=0)
        inward = np.sum([(-lm["normal"] if lm["side"] == "outside" else lm["normal"]) for lm in signs], axis=0)
        inward /= max(np.linalg.norm(inward), 1e-9)
        entry = door + inward * float(self.config["entry_depth"])
        self.rooms[room] = dict(door=door, inward=inward, entry=entry)

    def summary(self):
        rooms = ", ".join(sorted(self.rooms)) or "none yet"
        return (f"{100 * self.grid.known_fraction():.1f}% of a {self.config['map_size']:.0f} m map known, "
                f"{len(self.landmarks)} door signs, rooms remembered: {rooms}")


class Narrator:
    """Prints what the robot is trying to do, without repeating itself."""

    def __init__(self, sink=None):
        self.sink = sink or (lambda text: None)
        self.last_text = {}
        self.last_time = {}

    def say(self, key, text, now=0.0, every=None):
        if every is not None:
            if now - self.last_time.get(key, -math.inf) < every:
                return
        elif self.last_text.get(key) == text:
            return
        self.last_text[key] = text
        self.last_time[key] = now
        self.sink(text)


class VisionNavigator:
    def __init__(self, config, room=None, goal=None, memory=None, narrate=None):
        self.config = config
        self.rooms = sorted({s["room"] for s in config["signs"].values()})
        if room is not None and room not in self.rooms:
            raise ValueError(f"unknown room {room!r}; signs exist for {self.rooms}")
        self.room = room
        self.goal_local = None if goal is None else np.asarray(goal, float)   # odom frame
        self.vision = FloorVision(config)
        self.memory = memory if memory is not None else MapMemory(config)
        self.say = Narrator(narrate)
        # With no remembered signs we cannot relate this run to any earlier
        # one, so the current odom frame becomes (or stays) the map frame.
        if self.memory.landmarks:
            self.localized, self.T = False, None
            self.map = ObservedMap(config)
            self.buffer, self.pending = [], []
        else:
            self.localized, self.T = True, Frame2D()
            self.map = self.memory.grid
            self.buffer, self.pending = [], []
        self.state = "OBSERVING"
        self.done = False
        self.last_frame = -math.inf
        self.last_plan = -math.inf
        self.path = []
        self.scan_angle = 0.0
        self.last_yaw = None
        self.explore_target = None
        self.progress_pose = None
        self.progress_time = None
        self.initial_scan = room is not None or not self.localized

    # -- frames -------------------------------------------------------------
    def to_map(self, pose):
        return self.T.apply_pose(pose) if self.T is not None else np.asarray(pose, float)

    @property
    def goal(self):
        """Current goal in the map frame, or None if not known yet."""
        if self.room is not None:
            info = self.memory.rooms.get(self.room) if self.localized else None
            return None if info is None else info["entry"]
        if self.goal_local is not None:
            return self.T.apply(self.goal_local[None])[0] if self.T is not None else self.goal_local
        return None

    # -- perception ---------------------------------------------------------
    def observe(self, frames, pose):
        # Reject missing/stale cameras rather than treating unavailable pixels as clear.
        if len(frames) != 3 or {f.name for f in frames} != {c["name"] for c in self.config["cameras"]}:
            return
        if not all(f.rgb.shape == (self.config["height"], self.config["width"], 3)
                   and np.isfinite(f.rgb).all() for f in frames):
            return
        now = min(f.time for f in frames)
        free, blocked, signs = self.vision.observe(frames)
        for sign in signs:
            self._handle_sign(sign, pose, now)
        # Stop only if no camera sees usable floor (the head view may be filled
        # by a wall or sign at close range while the hand cameras still see floor).
        if len(free) < 50 or max(m.mean() for m in self.vision.masks.values()) < 0.02:
            self.last_frame = -math.inf
            return
        self.map.integrate(free, blocked, self.to_map(pose))
        if not self.localized:
            self.buffer.append((free, blocked, np.asarray(pose, float).copy()))
            del self.buffer[:-600]
        self.last_frame = now

    def _handle_sign(self, sign, pose, now):
        pos_o = local_to_map(sign["position"][None], pose)[0]
        c, s = math.cos(pose[2]), math.sin(pose[2])
        n_o = np.array([c * sign["normal"][0] - s * sign["normal"][1],
                        s * sign["normal"][0] + c * sign["normal"][1]])
        label = sign["room"].replace("_", " ")
        self.say.say(f"sign_{sign['id']}", f"I can see the {label} sign ({sign['side']} face), "
                     f"{sign['distance']:.1f} m away.", now, every=10.0)
        if not self.localized:
            lm = self.memory.landmarks.get(sign["id"])
            if lm is None:
                self.pending.append((sign, pos_o, n_o))
                return
            self.T = Frame2D.from_landmark(pos_o, n_o, lm["pos"], lm["normal"])
            self._finish_localization(label, now)
            return
        lm = self.memory.landmarks.get(sign["id"])
        if lm is not None and sign["distance"] < 3.0 and self.memory.loaded:
            # Gentle re-localization: corrects slow odometry drift against a known sign.
            self.T = self.T.blend(Frame2D.from_landmark(pos_o, n_o, lm["pos"], lm["normal"]), 0.2)
        pos_m = self.T.apply(pos_o[None])[0]
        n_m = self.T.rotate(n_o)
        if self.memory.update_landmark(sign["id"], sign["room"], sign["side"], pos_m, n_m):
            self.say.say(f"new_{sign['id']}", f"New landmark: the {label} door ({sign['side']} sign) is now in my memory.", now)
        self.memory.update_room(sign["room"])

    def _finish_localization(self, label, now):
        self.localized = True
        self.map = self.memory.grid
        for free, blocked, pose in self.buffer:
            self.map.integrate(free, blocked, self.T.apply_pose(pose))
        self.buffer.clear()
        for sign, pos_o, n_o in self.pending:
            self.memory.update_landmark(sign["id"], sign["room"], sign["side"],
                                        self.T.apply(pos_o[None])[0], self.T.rotate(n_o))
            self.memory.update_room(sign["room"])
        self.pending.clear()
        self.path, self.explore_target, self.last_plan = [], None, -math.inf
        self.say.say("localized", f"I recognise the {label} sign from my saved memory, so I now know where I am. "
                     f"Memory: {self.memory.summary()}.", now)

    # -- planning / control -------------------------------------------------
    def _set_state(self, state, now, goal=None, xy=None):
        if state == self.state:
            return
        self.state = state
        room = self.room.replace("_", " ") if self.room else None
        if state == "SCANNING":
            what = (f"a door sign that tells me where the {room} is" if room and goal is None
                    else "a sign I recognise" if not self.localized else "a way forward")
            text = f"Turning in place to look for {what}."
        elif state == "EXPLORING":
            tx, ty = self.explore_target if self.explore_target is not None else (float("nan"), float("nan"))
            text = f"No known route yet; exploring toward unseen space near ({tx:.1f}, {ty:.1f})."
        elif state == "NAVIGATING":
            dist = float(np.linalg.norm(goal - xy)) if goal is not None and xy is not None else float("nan")
            target = f"the {room} entrance" if room else "the goal"
            source = "from memory" if self.memory.loaded else "through the space I have mapped"
            text = f"Following a path {source} to {target}, {dist:.1f} m to go."
        elif state == "REPLANNING":
            text = "Something is blocking the path I was on. Replanning."
        elif state == "BLOCKED":
            text = "Looked all the way around and found no safe way forward. Stopping."
        elif state == "CAMERA_LOST":
            text = "Camera images are missing or stale. Stopping until they return."
        elif state == "ARRIVED":
            text = f"Arrived at the {room}." if room else "Arrived at the goal."
        else:
            text = state
        self.say.say("state", text, now)

    def command(self, pose, now):
        if self.done:
            return 0.0, 0.0
        if now - self.last_frame > 0.5 or now < self.last_frame:
            self._set_state("CAMERA_LOST", now)
            return 0.0, 0.0
        pose_m = self.to_map(pose)
        xy = pose_m[:2]
        if self.progress_pose is None or np.linalg.norm(xy - self.progress_pose) > 0.15:
            self.progress_pose, self.progress_time = xy.copy(), now
        if now - self.progress_time > 45.0:
            self._set_state("BLOCKED", now)
            return 0.0, 0.0
        goal = self.goal
        tolerance = self.config["room_stop_distance"] if self.room else 0.3
        if goal is not None and np.linalg.norm(goal - xy) < tolerance:
            self._set_state("ARRIVED", now)
            self.done = True
            return 0.0, 0.0
        if self.last_yaw is not None:
            self.scan_angle += abs(wrap(pose[2] - self.last_yaw))
        self.last_yaw = pose[2]
        v_max = float(self.config.get("max_speed", 0.18))
        w_max = float(self.config.get("max_turn_rate", 0.5))
        scan = float(self.config.get("scan_rate", 0.4))
        if self.initial_scan:
            if goal is None and self.scan_angle < 2 * math.pi:
                self._set_state("SCANNING", now, goal)
                return 0.0, scan
            self.initial_scan = False
            self.scan_angle = 0.0
        if now - self.last_plan >= 0.5:
            if self.explore_target is not None and np.linalg.norm(self.explore_target - xy) < 0.25:
                x, y = self.map.cells(self.explore_target)
                self.map.visits[max(0, y - 3):y + 4, max(0, x - 3):x + 4] += 10
                self.explore_target = None
            self.path = self.map.route(xy, goal if goal is not None else self.explore_target)
            if goal is None and self.path:
                self.explore_target = self.path[-1]
            self.last_plan = now
        if not self.path:
            # Observe a full turn before declaring that no safe continuation exists.
            if self.scan_angle < 2 * math.pi:
                self._set_state("SCANNING", now, goal)
                return 0.0, scan
            self._set_state("BLOCKED", now)
            return 0.0, 0.0
        self.scan_angle = 0.0
        # Use a short checked lookahead (longer when driving faster); never cut
        # corners through inflated obstacles.
        target = None
        for point in self.path[1:]:
            if np.linalg.norm(point - xy) > max(0.45, 2.5 * v_max):
                break
            if self.map.safe_segment(xy, point):
                target = point
        if target is None:
            self.path = []
            self._set_state("REPLANNING", now)
            return 0.0, 0.0
        delta = target - xy
        err = wrap(math.atan2(delta[1], delta[0]) - pose_m[2])
        self._set_state("NAVIGATING" if goal is not None else "EXPLORING", now, goal, xy)
        remaining = float(np.linalg.norm(goal - xy)) if goal is not None else float("nan")
        self.say.say("progress", f"Still {self.state.lower()}: {remaining:.1f} m to go, "
                     f"{len(self.path)} waypoints, {100 * self.map.known_fraction():.1f}% of the map seen.",
                     now, every=6.0)
        v = min(v_max, np.linalg.norm(delta) * 0.6) if abs(err) < 0.35 else 0.0
        return v, float(np.clip(1.5 * err, -w_max, w_max))

    def guard(self, pose, now, v, w):
        """Teleop also requires fresh observed clearance in the direction of travel."""
        if now - self.last_frame > 0.5:
            return 0.0, 0.0
        pose_m = self.to_map(pose)
        distance = math.copysign(0.35 + abs(v) * 0.5, v)
        end = pose_m[:2] + distance * np.array([math.cos(pose_m[2]), math.sin(pose_m[2])])
        return (v if v == 0 or self.map.safe_segment(pose_m[:2], end) else 0.0), w
