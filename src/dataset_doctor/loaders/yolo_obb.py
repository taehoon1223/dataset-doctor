"""yolo-obb: `cls x1 y1 x2 y2 x3 y3 x4 y4`, normalized to 0-1."""
import numpy as np

EXPECTED = "8 values (x1 y1 ... x4 y4)"


def valid_count(n: int) -> bool:
    return n == 8


def to_points(values: np.ndarray) -> np.ndarray:
    return values.reshape(4, 2)
