"""RGB floor mapping and online navigation, independent of MuJoCo and layout.

This baseline assumes a flat floor with a calibrated colour model. Distances
are ground-plane projections, NOT monocular depth estimates. Replace the
floor mask with a trained traversability model for non-demo environments.
"""
import heapq
import math

import cv2
import numpy as np

from room_signs import SignReader
from navigation_memory import NavigationMemory


def local_to_map(points, pose):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return np.asarray(points) @ np.array([[c, s], [-s, c]]) + pose[:2]


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


class FloorVision:
    def __init__(self, config):
        self.config = config
        self.sign_reader = SignReader(config)
        self.masks = {}

    def observe(self, frames):
        free, blocked, labels = [], [], []
        for frame in frames:
            rgb = frame.rgb
            # No simulator segmentation or depth buffers. This colour classifier
            # is intentionally explicit and configurable, not semantic room AI.
            mask = ((np.ptp(rgb.astype(np.int16), axis=2) <= np.maximum(2, rgb.mean(axis=2)*self.config["floor_max_chroma"]))
                    & (rgb.min(axis=2) >= self.config["floor_min_value"])).astype(np.uint8)
            labels.extend(self.sign_reader.read(frame))
            # Wall signs stay non-floor; their 3-D poses come from calibrated PnP.
            # Restrict to substantial connected patches; discard speckles/highlights.
            count, components, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            keep = np.zeros(count, np.uint8)
            keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= 70)
            mask = keep[components]
            # A high-mounted camera can see floor OVER a short wall. Keep only
            # the uninterrupted ground run from the image bottom in each column;
            # distant floor behind an occluding object must not clear that object.
            mask = np.flip(np.logical_and.accumulate(np.flip(mask, axis=0), axis=0), axis=0).astype(np.uint8)
            self.masks[frame.name] = mask
            h, w = mask.shape
            yy, xx = np.mgrid[0:h:2, 0:w:2]
            pixels = np.column_stack((xx.ravel(), yy.ravel()))
            points, valid = ground_points(frame, pixels)
            distance = np.linalg.norm(points, axis=1)
            valid &= (distance >= 0.3) & (distance <= self.config["max_range"])
            floor = mask[yy, xx].ravel().astype(bool)
            # Only obstacle silhouettes bordering visible floor are grounded.
            # Projecting arbitrary upper pixels as floor would invent obstacles.
            free.append(points[valid & floor])
            # Ground contact is the bottom silhouette, not every vertical edge
            # of an object (those edges would project false obstacles into doors).
            run = mask.sum(axis=0)
            columns = np.flatnonzero((run > 0) & (run < h))
            contact = np.column_stack((columns, h-run[columns]-1))
            hits, hit_valid = ground_points(frame, contact)
            hit_distance = np.linalg.norm(hits, axis=1)
            hit_valid &= (hit_distance >= 0.3) & (hit_distance <= self.config["max_range"])
            blocked.append(hits[hit_valid])
        return (np.concatenate(free) if free else np.empty((0, 2)),
                np.concatenate(blocked) if blocked else np.empty((0, 2)), labels)


class ObservedMap:
    """Fixed-size local odometry grid. Unknown space is never traversable."""
    def __init__(self, config):
        self.resolution = config["resolution"]
        self.size = int(math.ceil(config["map_size"] / self.resolution))
        self.offset = self.size // 2
        self.radius = config["robot_radius"]
        self.evidence = np.zeros((self.size, self.size), np.float32)
        self.seen = np.zeros_like(self.evidence, bool)
        self.visits = np.zeros_like(self.evidence)
        self.traveled = np.zeros_like(self.evidence, bool)
        # Wall-clock expiry per cell for `block_temporarily` - never written into
        # `evidence`/saved to disk, so a transient block can't corrupt the
        # persisted map or permanently rule out a real shortcut.
        self.blocked_until = np.zeros_like(self.evidence)
        self._previous_position = None
        self.targets = {}
        self._traversable = None
        # Known occupied footprint at startup; no assumed room, corridor, or route.
        yy, xx = np.mgrid[:self.size, :self.size]
        footprint = np.hypot(xx-self.offset, yy-self.offset) * self.resolution <= self.radius + 0.1
        self.evidence[footprint] = -3
        self.seen[footprint] = True

    def cells(self, points):
        return np.floor(np.asarray(points) / self.resolution).astype(int) + self.offset

    def xy(self, cell):
        return (np.asarray(cell) - self.offset + 0.5) * self.resolution

    def inside(self, cell):
        return 0 <= cell[0] < self.size and 0 <= cell[1] < self.size

    def integrate(self, free, blocked, labels, pose):
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
        for label, position in labels.items():
            self.targets[label] = local_to_map(np.asarray(position)[None], pose)[0]
        x, y = self.cells(pose[:2])
        if self.inside((x, y)):
            self.visits[y, x] += 1
            if self._previous_position is None:
                self.traveled[y, x] = True
            else:
                count = max(2, int(np.linalg.norm(pose[:2]-self._previous_position)/self.resolution*3)+1)
                for cell in self.cells(np.linspace(self._previous_position, pose[:2], count)):
                    if self.inside(cell):
                        self.traveled[cell[1], cell[0]] = True
            self._previous_position = pose[:2].copy()

    def traveled_route(self, start_xy, goal_xy):
        """Topology learned from motion, used only as a guide for live local planning."""
        ys, xs = np.nonzero(self.traveled)
        if not len(xs):
            return []
        cells = np.column_stack((xs, ys))
        points = self.xy(cells)
        start_index = np.argmin(np.linalg.norm(points-start_xy, axis=1))
        goal_index = np.argmin(np.linalg.norm(points-goal_xy, axis=1))
        if np.linalg.norm(points[start_index]-start_xy) > .6 or np.linalg.norm(points[goal_index]-goal_xy) > .45:
            return []
        start, goal = tuple(cells[start_index]), tuple(cells[goal_index])
        queue, costs, previous = [(0., start)], {start: 0.}, {}
        while queue:
            distance, u = heapq.heappop(queue)
            if distance != costs[u]:
                continue
            if u == goal:
                path = [u]
                while path[-1] != start:
                    path.append(previous[path[-1]])
                return [self.xy(cell) for cell in reversed(path)]
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                v = (u[0]+dx, u[1]+dy)
                if not self.inside(v) or not self.traveled[v[1], v[0]]:
                    continue
                candidate = distance+math.hypot(dx, dy)
                if candidate < costs.get(v, math.inf):
                    costs[v], previous[v] = candidate, u
                    heapq.heappush(queue, (candidate, v))
        return []

    def traversable(self, now=None):
        if self._traversable is None:
            occupied = ((self.evidence >= 0) | ~self.seen).astype(np.uint8)
            radius = int(math.ceil(self.radius / self.resolution))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
            inflated = cv2.dilate(occupied, kernel).astype(bool)
            self._traversable = self.seen & (self.evidence < 0) & ~inflated
        if now is None:
            return self._traversable
        return self._traversable & ~(self.blocked_until > now)

    def block_temporarily(self, xy, now, radius=1.5, duration=45.0):
        """Treat a neighborhood as impassable until `now + duration`.

        For when a route through here keeps failing its live safety check on
        every replan - e.g. floor-vision noise at a grazing viewing angle down
        a long straight corridor can flip an already-verified-free cell to
        blocked on a single bad frame, so the planned route flickers between
        accepted and rejected forever instead of settling either way. Rather
        than retry the exact same spot indefinitely, force routing to treat it
        as blocked for a while so it commits to a different, already-known
        route (e.g. the main hallway) instead. Not evidence, so it can't
        corrupt the persisted map, and it expires so the shortcut stays
        available to retry later.
        """
        cx, cy = self.cells(xy)
        r = int(math.ceil(radius / self.resolution))
        y0, y1 = max(0, cy-r), min(self.size, cy+r+1)
        x0, x1 = max(0, cx-r), min(self.size, cx+r+1)
        region = self.blocked_until[y0:y1, x0:x1]
        np.maximum(region, now + duration, out=region)

    def route(self, start_xy, goal_xy=None, now=None):
        """Dijkstra over observed free cells; otherwise choose a reachable frontier.

        The frontier is tried twice: first excluding any free cell close to an
        observed wall (avoids picking pointless wall-hugging viewpoints in open
        rooms), then - only if that finds nothing - without that exclusion. In
        a narrow corridor (a doorway width or so), every traversable cell can
        be "close to a wall" simultaneously on both sides, which would
        otherwise collapse the frontier to nothing map-wide and leave the
        robot with no next move even though most of the map is still unseen.
        """
        start = tuple(self.cells(start_xy))
        free = self.traversable(now)
        if not self.inside(start):
            return []
        if not free[start[1], start[0]]:
            # The robot's own cell can read as briefly non-traversable close to
            # an inflated wall boundary (sensor noise, or simply standing near
            # the edge of a doorway-width corridor) without it actually being
            # blocked. Snap to the nearest traversable cell instead of failing
            # outright, so a transient blip here doesn't strand the robot with
            # no route at all.
            ys, xs = np.nonzero(free)
            if len(xs) == 0:
                return []
            nearest = np.argmin((xs-start[0])**2 + (ys-start[1])**2)
            if (xs[nearest]-start[0])**2 + (ys[nearest]-start[1])**2 > (3*int(math.ceil(self.radius/self.resolution)))**2:
                return []  # nothing traversable nearby; a real dead end
            start = (int(xs[nearest]), int(ys[nearest]))
        goal = tuple(self.cells(goal_xy)) if goal_xy is not None else None
        frontier_width = 2 * (int(math.ceil(self.radius / self.resolution)) + 2) + 1
        # A wall bordering unknown space is not an exploration frontier. Start
        # from actual free/unknown boundaries, excluding occupied silhouettes,
        # then choose a reachable viewpoint set back by the base clearance.
        unknown_adjacent = cv2.dilate((~self.seen).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        boundary_base = self.seen & (self.evidence < 0) & unknown_adjacent
        wall_adjacent = cv2.dilate((self.evidence > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0

        def frontier_for(avoid_walls):
            boundary = boundary_base & ~wall_adjacent if avoid_walls else boundary_base
            return free & (cv2.dilate(boundary.astype(np.uint8),
                                      np.ones((frontier_width, frontier_width), np.uint8)) > 0)

        for avoid_walls in (True, False):
            frontier = frontier_for(avoid_walls)
            cost, previous = {start: 0.0}, {}
            queue = [(0.0, start)]
            best, best_score = None, -math.inf
            approach, approach_score = None, math.inf
            start_goal_distance = np.linalg.norm(np.asarray(start_xy)-goal_xy) if goal_xy is not None else 0
            while queue:
                distance, u = heapq.heappop(queue)
                if distance != cost[u]:
                    continue
                if goal is not None and u == goal:
                    best = u
                    break
                x, y = u
                if goal_xy is not None:
                    remaining = np.linalg.norm(self.xy(u)-goal_xy)
                    if remaining < start_goal_distance-.3 and distance > .2:
                        score = remaining + .05*distance
                        if score < approach_score:
                            approach, approach_score = u, score
                if frontier[y, x] and distance > 0.5:
                    # Prefer nearby unexplored views; avoid repeatedly selecting
                    # already visited viewpoints. A known goal biases exploration.
                    score = -distance - 0.5 * self.visits[y, x]
                    if goal_xy is not None:
                        score -= 2 * np.linalg.norm(self.xy(u) - goal_xy)
                    if score > best_score:
                        best, best_score = u, score
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                    v = (x+dx, y+dy)
                    if not self.inside(v) or not free[v[1], v[0]]:
                        continue
                    if dx and dy and (not free[y, x+dx] or not free[y+dy, x]):
                        continue
                    candidate = distance + self.resolution * math.hypot(dx, dy)
                    if candidate < cost.get(v, math.inf):
                        cost[v], previous[v] = candidate, u
                        heapq.heappush(queue, (candidate, v))
            # If a remembered goal is temporarily disconnected, first get close
            # enough to re-observe the gap from a safe reachable viewpoint. Stopping
            # several metres away cannot refresh a stale doorway in camera range.
            if goal is not None and best != goal and approach is not None:
                best = approach
            if best is not None:
                break
        if best is None:
            return []
        path = [best]
        while path[-1] != start:
            path.append(previous[path[-1]])
        return [self.xy(cell) for cell in reversed(path)]

    def safe_segment(self, start, end, now=None):
        free = self.traversable(now)
        n = max(2, int(np.linalg.norm(np.asarray(end)-start) / self.resolution * 2) + 1)
        cells = self.cells(np.linspace(start, end, n))
        if not ((cells >= 0) & (cells < self.size)).all():
            return False
        # Skip the very first sample - the robot's own current cell. In a tight
        # passage (doorway width or so) it can briefly read as non-traversable
        # from inflation/perception noise even though the robot is already
        # safely there; requiring it to also pass would veto every future
        # target and strand the robot in a REPLANNING loop with no way out.
        return bool(free[cells[1:, 1], cells[1:, 0]].all())


class VisionNavigator:
    def __init__(self, config, room=None, goal=None, memory_path=None, memory=None):
        """`memory`, if given, is an already-loaded NavigationMemory to keep
        using (its map and localization state carry over) instead of loading
        `memory_path` fresh. Switching to a new room/goal target mid-session
        (a voice command, say) should reuse it: loading fresh would discard
        that the robot already knows exactly where it is and force it to
        re-locate a known landmark by turning in place, which fails outright
        - and gets permanently stuck - if none happens to be in view right
        then. Only an actual physical reset (spawn pose, 'r' key) should
        start localization over from scratch."""
        self.config = config
        self.room, self.goal = room, None if goal is None else np.asarray(goal, float)
        self.vision = FloorVision(config)
        if memory is not None:
            self.map = memory.grid
            self.memory = memory
        else:
            self.map = ObservedMap(config)
            self.memory = NavigationMemory(self.map, memory_path)
        self.local_goal = None if goal is None else np.asarray(goal, float)
        self.current_pose = np.zeros(3)
        self.localization_turn = 0.0
        self.localization_yaw = None
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
        self.initial_scan = room is not None
        self.arrival_turn = None
        self.arrival_yaw = None
        self.memory_hint = None
        self.backup_attempts = 0
        self.stall_since = None

    def observe(self, frames, pose):
        # Reject missing/stale cameras rather than treating unavailable pixels as clear.
        if len(frames) != 3 or {f.name for f in frames} != {c["name"] for c in self.config["cameras"]}:
            return
        if not all(f.rgb.shape == (self.config["height"], self.config["width"], 3)
                   and np.isfinite(f.rgb).all() for f in frames):
            return
        free, blocked, labels = self.vision.observe(frames)
        if len(free) < 50:
            self.last_frame = -math.inf
            return
        self.last_frame = min(f.time for f in frames)
        map_pose = self.memory.update(labels, pose)
        if map_pose is None:
            return
        self.current_pose = map_pose
        self.map.integrate(free, blocked, {}, map_pose)
        self.memory.learn_entry_targets()
        if self.room in self.map.targets:
            self.goal = self.map.targets[self.room]
        elif self.local_goal is not None:
            self.goal = self.memory.pose(np.r_[self.local_goal, 0])[:2]

    def command(self, pose, now):
        if self.done:
            return 0.0, 0.0
        if now - self.last_frame > 0.5 or now < self.last_frame:
            self.state = "CAMERA_LOST"
            return 0.0, 0.0
        if not self.memory.localized:
            if self.localization_yaw is not None:
                self.localization_turn += abs((pose[2]-self.localization_yaw+math.pi) % (2*math.pi)-math.pi)
            self.localization_yaw = pose[2]
            self.state = "LOCALIZING" if self.localization_turn < 2*math.pi else "LOCALIZATION_REQUIRED"
            return (0.0, .4) if self.state == "LOCALIZING" else (0.0, 0.0)
        pose = self.memory.pose(pose)
        self.current_pose = pose
        if self.arrival_turn is not None:
            self.arrival_turn += abs((pose[2]-self.arrival_yaw+math.pi) % (2*math.pi)-math.pi)
            self.arrival_yaw = pose[2]
            inside_known = any(self.memory.landmarks[key]["face"] == "inside"
                               for key in self.memory.rooms[self.room]["signs"])
            if inside_known or self.arrival_turn >= 2*math.pi:
                self.state, self.done = "ARRIVED", True
                self.map.targets[self.room] = pose[:2].copy()
                return 0.0, 0.0
            self.state = "ARRIVAL_SCAN"
            return 0.0, .4
        if self.progress_pose is None or np.linalg.norm(pose[:2]-self.progress_pose) > 0.15:
            self.progress_pose, self.progress_time = pose[:2].copy(), now
        if now - self.progress_time > 45.0:
            self.state = "BLOCKED"
            return 0.0, 0.0
        if now - self.progress_time > 20.0:
            # No physical progress in 20s despite repeated (re)planning - a
            # narrow passage can leave the robot stuck rejecting every
            # candidate next step, or with no reachable frontier at all, from
            # this exact spot and heading. Back up and turn to change vantage
            # point before the 45s mark gives up for good; a few seconds of
            # different geometry in view is often enough to unstick it.
            self.state = "BACKING_UP"
            return -0.12, 0.3
        tolerance = self.config["room_stop_distance"] if self.room else 0.3
        if self.goal is not None and np.linalg.norm(self.goal - pose[:2]) < tolerance:
            if self.room in self.memory.rooms:
                self.memory.rooms[self.room]["visited"] = True
                self.map.targets[self.room] = pose[:2].copy()
                if not any(self.memory.landmarks[key]["face"] == "inside"
                           for key in self.memory.rooms[self.room]["signs"]):
                    self.arrival_turn, self.arrival_yaw = 0., pose[2]
                    self.state = "ARRIVAL_SCAN"
                    return 0.0, .4
            self.state, self.done = "ARRIVED", True
            return 0.0, 0.0
        if self.last_yaw is not None:
            self.scan_angle += abs((pose[2]-self.last_yaw + math.pi) % (2*math.pi) - math.pi)
        self.last_yaw = pose[2]
        if self.initial_scan:
            if self.goal is None and self.scan_angle < 2*math.pi:
                self.state = "SCANNING"
                return 0.0, 0.4
            self.initial_scan = False
            self.scan_angle = 0.0
        if now - self.last_plan >= 0.5:
            if self.explore_target is not None and np.linalg.norm(self.explore_target-pose[:2]) < 0.25:
                x, y = self.map.cells(self.explore_target)
                self.map.visits[max(0,y-3):y+4, max(0,x-3):x+4] += 10
                self.explore_target = None
            planning_goal = self.goal if self.goal is not None else self.explore_target
            self.memory_hint = None
            # route() already finds the shortest currently-known path through
            # everywhere observed as free - if the apartment has more than one
            # way to the goal (e.g. a direct corridor as well as the original
            # hallway route), this picks whichever is actually shorter once
            # both are known, rather than always retracing a specific
            # previously-driven path. traveled_route is only a fallback for a
            # remembered goal that's temporarily disconnected from what's
            # currently observed and needs to be re-approached to refresh it.
            self.path = self.map.route(pose[:2], planning_goal, now)
            if (self.goal is not None and not self.path
                    and self.memory.rooms.get(self.room, {}).get("visited")):
                history = self.map.traveled_route(pose[:2], self.goal)
                if history:
                    self.memory_hint = history[-1]
                    for point in history:
                        if np.linalg.norm(point-pose[:2]) >= .8:
                            self.memory_hint = point
                            break
                    self.path = self.map.route(pose[:2], self.memory_hint, now)
            if self.goal is None and self.path:
                self.explore_target = self.path[-1]
            self.last_plan = now
        if not self.path:
            # Observe a full turn before concluding nothing reachable is in
            # view. If that comes up empty, try a few rounds of backing up
            # and rescanning - a different vantage point a bit further back
            # can reveal a frontier that was occluded or too close to a wall
            # to register from here - before finally giving up.
            if self.scan_angle < 2*math.pi:
                self.state = "SCANNING"
                return 0.0, 0.4
            if self.backup_attempts < 3:
                self.backup_attempts += 1
                self.scan_angle = 0.0
                self.state = "BACKING_UP"
                return -0.12, 0.3
            self.state = "BLOCKED"
            return 0.0, 0.0
        self.scan_angle = 0.0
        self.backup_attempts = 0
        # Use a short checked lookahead; never cut corners through inflated obstacles.
        target = None
        for point in self.path[1:]:
            if np.linalg.norm(point-pose[:2]) > 0.45:
                break
            if self.map.safe_segment(pose[:2], point, now):
                target = point
        if target is None:
            self.path = []
            # A route that keeps getting planned but never clears its own live
            # safety check (e.g. floor-vision noise at a grazing angle down a
            # long straight corridor flipping an already-verified-free cell to
            # blocked on a single bad frame) will otherwise flicker between
            # REPLANNING and SCANNING forever, retrying the exact same
            # bottleneck every 0.5s replan without ever trying a different,
            # already-known way round (e.g. the main hallway). If we haven't
            # taken a real step in a while, block the spot we're stuck at for
            # a while so the next plan is forced to go a different way.
            if self.stall_since is None:
                self.stall_since = now
            elif now - self.stall_since > 6.0:
                self.map.block_temporarily(pose[:2], now)
                self.explore_target = None
                self.stall_since = None
            self.state = "REPLANNING"
            return 0.0, 0.0
        self.stall_since = None
        delta = target - pose[:2]
        err = (math.atan2(delta[1], delta[0]) - pose[2] + math.pi) % (2*math.pi) - math.pi
        if self.goal is None:
            self.state = "EXPLORING"
        elif self.memory_hint is not None:
            self.state = "FOLLOWING_MEMORY"
        else:
            self.state = "NAVIGATING" if np.linalg.norm(self.path[-1]-self.goal) < .2 else "INSPECTING_ROUTE"
        v = min(0.18, np.linalg.norm(delta) * 0.6) if abs(err) < 0.35 else 0.0
        return v, float(np.clip(1.5 * err, -0.5, 0.5))

    def guard(self, pose, now, v, w):
        """Teleop also requires fresh observed clearance in the direction of travel."""
        if now-self.last_frame > 0.5:
            return 0.0, 0.0
        if not self.memory.localized:
            return 0.0, w
        pose = self.memory.pose(pose)
        distance = math.copysign(0.35 + abs(v)*0.5, v)
        end = pose[:2] + distance * np.array([math.cos(pose[2]), math.sin(pose[2])])
        return (v if v == 0 or self.map.safe_segment(pose[:2], end, now) else 0.0), w

    def explain(self, v=0.0, w=0.0):
        """Concise observable intent for terminal narration, including wait reasons."""
        room = self.room.replace("_", " ") if self.room else "the requested goal"
        explanations = {
            "OBSERVING": "I am observing the floor and learning free space from my cameras.",
            "LOCALIZING": "I loaded my map and am turning to find a remembered code before translating.",
            "LOCALIZATION_REQUIRED": "I could not match a saved code after a full turn. I am stopped; my saved map is unchanged.",
            "CAMERA_LOST": "I am stopped because the camera observations are missing, stale, or unusable.",
            "SCANNING": "I am turning to inspect unobserved space and look for readable room signs.",
            "BLOCKED": "I am stopped: I cannot establish a safe route or have made no progress.",
            "BACKING_UP": "I have made no progress for a while. I am backing up and turning to try a different vantage point.",
            "REPLANNING": "New observations changed the safe path. I am stopped while I replan.",
            "ARRIVED": f"I reached {room} using the learned map.",
            "ARRIVAL_SCAN": f"I reached the room-side destination. I am scanning inside {room} to remember landmarks for a future restart.",
        }
        if self.state in explanations:
            return explanations[self.state]
        if self.state == "EXPLORING":
            intent = f"I do not know a safe route to {room} yet; I am exploring observed free space." if self.room else "I am exploring and saving newly observed layout."
        elif self.state == "INSPECTING_ROUTE":
            intent = f"The remembered route to {room} has a gap. I am approaching a safe viewpoint to check it with my cameras."
        elif self.state == "FOLLOWING_MEMORY":
            intent = f"I am following a route I previously traveled toward {room}, checking each local segment with my cameras."
        else:
            distance = float(np.linalg.norm(self.goal-self.current_pose[:2])) if self.goal is not None else 0
            intent = f"I am following my learned map toward {room}, about {distance:.1f} m away."
        action = ("Turning left" if w > 0 else "Turning right") if v == 0 and abs(w) > .05 else "Moving forward" if v > 0 else "Holding position"
        return f"{intent} {action}; {len(self.path)} cells in the current plan."
