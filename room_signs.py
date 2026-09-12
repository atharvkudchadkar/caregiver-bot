"""Printed doorway signs and a small, explicit English-sign recognizer.

Marker IDs carry no room semantics. Text is recognized from rectified pixels
using the supplied room vocabulary/font, not a marker-ID-to-room lookup.
"""
from dataclasses import dataclass
import math

import cv2
import numpy as np

WIDTH, HEIGHT = 800, 600
MARKER_CORNERS = np.float32([[250, 40], [550, 40], [550, 340], [250, 340]])


def sign_image(marker_id, room, face):
    image = np.full((HEIGHT, WIDTH), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    image[40:340, 250:550] = cv2.aruco.generateImageMarker(dictionary, marker_id, 300)
    for text, y, size in ((room.replace("_", " ").upper(), 450, 2.0), (face.upper(), 545, 1.5)):
        width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, 3)[0][0]
        cv2.putText(image, text, ((WIDTH-width)//2, y), cv2.FONT_HERSHEY_SIMPLEX, size, 0, 3, cv2.LINE_AA)
    return image


@dataclass
class LandmarkObservation:
    marker_id: int
    position: np.ndarray  # marker centre in robot's gravity-aligned frame
    yaw: float           # outward-facing wall normal
    room: str | None
    face: str | None
    error: float


class SignReader:
    def __init__(self, config):
        self.config = config
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), parameters)
        self.templates = {(room, face): sign_image(0, room, face)[365:580]
                          for room in config["room_names"] for face in ("inside", "entrance")}
        half = config["marker_size"] / 2
        self.object_points = np.float32([[-half, half, 0], [half, half, 0],
                                        [half, -half, 0], [-half, -half, 0]])

    def read(self, frame):
        gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        observations = []
        if ids is None:
            return observations
        cv_to_base = frame.rotation @ np.diag([1, -1, -1])
        for corner, marker_id in zip(corners, ids.ravel()):
            pixels = corner[0]
            ok, rvec, tvec = cv2.solvePnP(self.object_points, pixels, frame.K, None,
                                         flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok or tvec[2, 0] <= 0:
                continue
            projected, _ = cv2.projectPoints(self.object_points, rvec, tvec, frame.K, None)
            error = float(np.linalg.norm(projected.reshape(4, 2)-pixels, axis=1).mean())
            normal = cv_to_base @ cv2.Rodrigues(rvec)[0][:, 2]
            position = frame.origin + cv_to_base @ tvec.ravel()
            if error > 1.5 or abs(normal[2]) > .4 or np.linalg.norm(position[:2]) > 5:
                continue
            room = face = None
            transform = cv2.getPerspectiveTransform(pixels.astype(np.float32), MARKER_CORNERS)
            # Reject clipped text rather than guessing from the code or a partial word.
            source_quad = cv2.perspectiveTransform(
                np.float32([[[50, 365], [749, 365], [749, 579], [50, 579]]]), np.linalg.inv(transform))[0]
            h, w = gray.shape
            if ((source_quad[:, 0] >= 0) & (source_quad[:, 0] < w)
                    & (source_quad[:, 1] >= 0) & (source_quad[:, 1] < h)).all():
                rectified = cv2.warpPerspective(gray, transform, (WIDTH, HEIGHT))[365:580]
                # Score the room and INSIDE/ENTRANCE separately. A readable room
                # name must never compensate for a hidden or unreadable side line.
                results = []
                for start, end, labels in ((0, 110, [(name, "inside") for name in self.config["room_names"]]),
                                            (115, 215, [(self.config["room_names"][0], side) for side in ("inside", "entrance")])):
                    sample = cv2.resize(rectified[start:end], (320, end-start)).astype(np.float32)
                    scores = []
                    for label in labels:
                        reference = cv2.resize(self.templates[label][start:end], (320, end-start)).astype(np.float32)
                        score = float(cv2.matchTemplate(sample, reference, cv2.TM_CCOEFF_NORMED)[0, 0])
                        scores.append((score, label))
                    scores.sort(reverse=True)
                    results.append(scores[0][1] if scores[0][0] > .55 and scores[0][0]-scores[1][0] > .06 else None)
                if all(results):
                    room, face = results[0][0], results[1][1]
            observations.append(LandmarkObservation(int(marker_id), position,
                                math.atan2(normal[1], normal[0]), room, face, error))
        return observations
