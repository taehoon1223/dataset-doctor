"""yolo-hbb: `cls cx cy w h`, normalized to 0-1."""
import numpy as np

EXPECTED = "4 values (cx cy w h)"


def valid_count(n: int) -> bool:
    return n == 4


def to_points(values: np.ndarray) -> np.ndarray:
    cx, cy, w, h = values
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
