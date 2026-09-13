"""Learned map persistence and visual landmark localization; no world imports."""
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


def wrap(angle):
    return (angle + math.pi) % (2*math.pi) - math.pi


def compose(a, b):
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([a[0]+c*b[0]-s*b[1], a[1]+s*b[0]+c*b[1], wrap(a[2]+b[2])])


def inverse(pose):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return np.array([-c*pose[0]-s*pose[1], s*pose[0]-c*pose[1], -pose[2]])


class NavigationMemory:
    VERSION = 1

    def __init__(self, grid, path=None):
        self.grid = grid
        self.path = Path(path) if path else None
        self.landmarks = {}
        self.rooms = {}
        self.transform = np.zeros(3)  # map <- this run's odometry
        self.localized = True
        self.loaded = False
        self.observations = 0
        self.events = []
        self._candidate = None
        self._matches = 0
        self._text_votes = {}
        if self.path and self.path.exists():
            self.load()

    def pose(self, odometry):
        return compose(self.transform, odometry)

    def seed_pose(self, known_pose, odometry):
        """Align this memory's transform so pose(odometry) reproduces
        known_pose right now, and mark it localized - for switching to a
        different persisted memory (e.g. towing vs plain) without a physical
        move. The robot hasn't gone anywhere, only which map/landmark
        database it's consulting changed, so it shouldn't have to blindly
        re-localize (turn in place hunting for a sign) just to reuse a pose
        it already knows - especially risky while towing near a wall."""
        self.transform = compose(known_pose, inverse(odometry))
        self.localized = True

    def load(self):
        # No pickled Python objects. Validate before replacing any live map data.
        try:
            with np.load(self.path, allow_pickle=False) as saved:
                meta = json.loads(str(saved["metadata"].item()))
                if (meta["version"] != self.VERSION or meta["resolution"] != self.grid.resolution
                        or meta["size"] != self.grid.size):
                    raise ValueError("unsupported memory version or grid calibration")
                arrays = {key: saved[key].copy() for key in ("evidence", "seen", "visits")}
                traveled = saved["traveled"].copy() if "traveled" in saved else np.zeros_like(arrays["seen"])
                if traveled.shape != self.grid.evidence.shape:
                    raise ValueError("invalid motion history")
                if any(a.shape != self.grid.evidence.shape or not np.isfinite(a).all() for a in arrays.values()):
                    raise ValueError("invalid map arrays")
                for landmark in meta["landmarks"].values():
                    if np.asarray(landmark["pose"]).shape != (3,) or not np.isfinite(landmark["pose"]).all():
                        raise ValueError("invalid landmark pose")
                for target in meta["targets"].values():
                    if np.asarray(target).shape != (2,) or not np.isfinite(target).all():
                        raise ValueError("invalid room target")
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f"Cannot load memory {self.path}: {error}. Use a different --memory path or --no-memory.") from error
        self.grid.evidence = arrays["evidence"].astype(np.float32)
        self.grid.seen = arrays["seen"].astype(bool)
        self.grid.visits = arrays["visits"].astype(np.float32)
        self.grid.traveled = traveled.astype(bool)
        self.grid.targets = {key: np.asarray(value, float) for key, value in meta["targets"].items()}
        self.grid._traversable = None
        self.landmarks, self.rooms = meta["landmarks"], meta["rooms"]
        self.observations = meta["observations"]
        self.loaded, self.localized = True, False
        self.events.append(f"Loaded {self.grid.seen.sum()} observed cells and {len(self.landmarks)} landmarks; looking for a known code to locate myself.")

    def save(self):
        if self.path is None or not self.localized:
            return False  # never overwrite a loaded map using an unaligned run
        self.path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"version": self.VERSION, "resolution": self.grid.resolution, "size": self.grid.size,
                "landmarks": self.landmarks, "rooms": self.rooms, "observations": self.observations,
                "targets": {key: value.tolist() for key, value in self.grid.targets.items()}}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.path.parent, prefix=self.path.name+".", suffix=".tmp", delete=False) as output:
                temporary = Path(output.name)
                np.savez_compressed(output, evidence=self.grid.evidence, seen=self.grid.seen,
                                    visits=self.grid.visits, traveled=self.grid.traveled, metadata=json.dumps(meta))
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary and temporary.exists():
                temporary.unlink()
        return True

    def update(self, observations, odometry):
        candidates = []
        for observation in observations:
            key = str(observation.marker_id)
            relative = np.r_[observation.position[:2], observation.yaw]
            if key in self.landmarks:
                world_marker = np.asarray(self.landmarks[key]["pose"])
                candidates.append(compose(world_marker, inverse(compose(odometry, relative))))
        if candidates:
            candidate = np.median(candidates, axis=0)
            candidate[2] = math.atan2(np.mean(np.sin(np.array(candidates)[:, 2])),
                                     np.mean(np.cos(np.array(candidates)[:, 2])))
            if not self.localized:
                agrees = (self._candidate is not None and np.linalg.norm(candidate[:2]-self._candidate[:2]) < .15
                          and abs(wrap(candidate[2]-self._candidate[2])) < .12)
                self._matches = self._matches+1 if agrees else 1
                self._candidate = candidate
                if self._matches >= 3:
                    self.transform = candidate
                    self.localized = True
                    self.events.append("Recognized a saved landmark. My position is aligned with the learned map; I can navigate from memory.")
            else:
                delta = candidate-self.transform
                delta[2] = wrap(delta[2])
                if np.linalg.norm(delta[:2]) < .4 and abs(delta[2]) < .15:
                    self.transform += .08*delta  # modest drift correction, reject pose outliers
                    self.transform[2] = wrap(self.transform[2])
        if not self.localized:
            return None
        pose = self.pose(odometry)
        for observation in observations:
            key = str(observation.marker_id)
            relative = np.r_[observation.position[:2], observation.yaw]
            landmark_pose = compose(pose, relative)
            if key not in self.landmarks:
                self.landmarks[key] = {"pose": landmark_pose.tolist(), "room": None, "face": None}
                self.events.append(f"Learned localization landmark {key} from a camera image.")
            landmark = self.landmarks[key]
            if observation.room and landmark["room"] is None:
                label = (observation.room, observation.face)
                previous, count = self._text_votes.get(key, (None, 0))
                count = count+1 if previous == label else 1
                self._text_votes[key] = (label, count)
                if count >= 3:
                    landmark.update(room=observation.room, face=observation.face)
                    self.rooms.setdefault(observation.room, {"signs": [], "visited": False})["signs"].append(key)
                    self.events.append(f"Read '{observation.room.replace('_', ' ')}' on the {observation.face} sign repeatedly; remembering this room.")
            if landmark["room"] and landmark["face"] == "inside":
                normal = np.array([math.cos(landmark_pose[2]), math.sin(landmark_pose[2])])
                if np.dot(pose[:2]-landmark_pose[:2], normal) > .45 and np.linalg.norm(relative[:2]) < 2.5:
                    room_name = landmark["room"]
                    room = self.rooms[room_name]
                    if not room["visited"]:
                        room["visited"] = True
                        self.grid.targets[room_name] = pose[:2].copy()
                        self.events.append(f"I am inside {room_name.replace('_', ' ')}; saved a visited destination in the map.")
        self.observations += 1
        return pose

    def learn_entry_targets(self):
        """Find room-side floor from observed free space, never from building geometry."""
        ys, xs = np.nonzero(self.grid.traversable())
        points = self.grid.xy(np.column_stack((xs, ys)))
        for name, room in self.rooms.items():
            if room["visited"] or not len(points):
                continue
            candidates = []
            for key in room["signs"]:
                landmark = self.landmarks[key]
                x, y, yaw = landmark["pose"]
                normal = np.array([math.cos(yaw), math.sin(yaw)])
                if landmark["face"] == "entrance":
                    normal = -normal
                delta = points - [x, y]
                inward = delta @ normal
                lateral = np.abs(delta @ np.array([-normal[1], normal[0]]))
                valid = (inward > .65) & (inward < 1.6) & (lateral < 1.7)
                if valid.any():
                    available = points[valid]
                    best = available[np.argmin(np.linalg.norm(available-[x, y], axis=1))]
                    candidates.append(best)
            if candidates:
                self.grid.targets[name] = candidates[0]

    def drain_events(self):
        events, self.events = self.events, []
        return events
