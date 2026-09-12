"""Regenerate the wheelchair's standard ArUco DICT_4X4_50 marker, ID 0."""
from pathlib import Path

import cv2


HERE = Path(__file__).resolve().parent
OUT = HERE / "wheelchair_aruco_4x4_50_id0.png"
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
code = cv2.aruco.generateImageMarker(dictionary, 0, 896, borderBits=1)
print_image = cv2.copyMakeBorder(code, 64, 64, 64, 64, cv2.BORDER_CONSTANT, value=255)
# MuJoCo maps a box texture onto its rear face with a horizontal reflection.
# Pre-flip the image so the camera sees the canonical ArUco code.
print_image = cv2.flip(print_image, 1)
if not cv2.imwrite(str(OUT), print_image):
    raise RuntimeError(f"could not write {OUT}")
print(OUT)
