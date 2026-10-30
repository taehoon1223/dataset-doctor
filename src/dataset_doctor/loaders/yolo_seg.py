"""yolo-seg: `cls x1 y1 x2 y2 ... xn yn`, polygon normalized to 0-1."""
import numpy as np

EXPECTED = "an even number of values (x y pairs)"


def valid_count(n: int) -> bool:
    return n > 0 and n % 2 == 0


def to_points(values: np.ndarray) -> np.ndarray:
    return values.reshape(-1, 2)
