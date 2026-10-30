"""③ Geometry. Angles and areas are measured in pixels: in normalized coordinates a real
rectangle on a 1920x1080 image looks like a parallelogram."""
from __future__ import annotations

from collections import Counter

import numpy as np
import shapely
from shapely.geometry import Polygon

from dataset_doctor.core.models import Annotation, Dataset, Issue, Sample, Severity
from dataset_doctor.core.registry import register_check

COORD_TOLERANCE = 0.05         # overshoot past [0, 1] up to this is a WARNING, beyond it an ERROR
TINY_BOX_PX = 2.0
DEGENERATE_AREA_PX = 1.0
OBB_ANGLE_TOLERANCE_DEG = 10.0


def _issue(code: str, severity: Severity, sample: Sample, ann: Annotation, **detail) -> Issue:
    return Issue(code, severity, sample.image, sample.label_file, ann.line, detail)


def signed_area(pts: np.ndarray) -> float:
    """Shoelace. In image coordinates (y down) a positive area means clockwise on screen."""
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def segments_intersect(p1, p2, q1, q2) -> bool:
    """Proper crossing only; touching or collinear segments don't count."""
    return (_cross(q1, q2, p1) * _cross(q1, q2, p2) < 0) and (_cross(p1, p2, q1) * _cross(p1, p2, q2) < 0)


def obb_self_intersects(pts: np.ndarray) -> bool:
    # Opposite edges (0-1 vs 2-3, 1-2 vs 3-0) cross when the vertex order is twisted.
    return segments_intersect(pts[0], pts[1], pts[2], pts[3]) or segments_intersect(pts[1], pts[2], pts[3], pts[0])


def obb_max_angle_deviation(pts: np.ndarray) -> float:
    """Largest |corner angle - 90°| in degrees."""
    worst = 0.0
    for i in range(4):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % 4]
        v1, v2 = a - b, c - b
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        worst = max(worst, abs(90.0 - np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))))
    return float(worst)


def obb_windings(dataset: Dataset) -> list[tuple[Sample, Annotation, str]]:
    """Winding ("cw"/"ccw" on screen) of every well-formed OBB in the dataset."""
    found = []
    for sample in dataset.samples:
        if sample.size is None:
            continue
        for ann in sample.annotations:
            pts = ann.points * np.array(sample.size)
            area = signed_area(pts)
            if abs(area) >= DEGENERATE_AREA_PX and not obb_self_intersects(pts):
                found.append((sample, ann, "cw" if area > 0 else "ccw"))
    return found


def majority_winding(windings: list[tuple[Sample, Annotation, str]]) -> str | None:
    counts = Counter(w for _, _, w in windings)
    return max(("cw", "ccw"), key=lambda w: counts[w]) if counts else None


@register_check(stage="geometry")
def coord_range(sample: Sample):
    for ann in sample.annotations:
        lo, hi = float(ann.points.min()), float(ann.points.max())
        overshoot = max(-lo, hi - 1.0)
        if overshoot > 0:
            severity = Severity.WARNING if overshoot <= COORD_TOLERANCE else Severity.ERROR
            yield _issue("coord_out_of_range", severity, sample, ann, min=lo, max=hi)


@register_check(formats={"yolo-hbb"}, stage="geometry")
def hbb_geometry(sample: Sample):
    for ann in sample.annotations:
        w, h = (float(v) for v in ann.values[2:])
        if w <= 0 or h <= 0:
            yield _issue("non_positive_size", Severity.ERROR, sample, ann, w=w, h=h)
        elif sample.size is not None:
            w_px, h_px = w * sample.size[0], h * sample.size[1]
            if min(w_px, h_px) < TINY_BOX_PX:
                yield _issue("tiny_box", Severity.WARNING, sample, ann, w_px=w_px, h_px=h_px)


@register_check(formats={"yolo-obb"}, stage="geometry")
def obb_geometry(sample: Sample):
    if sample.size is None:
        return
    for ann in sample.annotations:
        pts = ann.points * np.array(sample.size)
        area = signed_area(pts)
        degenerate = abs(area) < DEGENERATE_AREA_PX
        if degenerate:
            yield _issue("degenerate_obb", Severity.ERROR, sample, ann, area_px=abs(area))
        if obb_self_intersects(pts):
            yield _issue("self_intersecting_obb", Severity.ERROR, sample, ann)
        deviation = obb_max_angle_deviation(pts)
        if deviation > OBB_ANGLE_TOLERANCE_DEG:
            yield _issue("non_rectangular_obb", Severity.WARNING, sample, ann, max_angle_deviation=deviation)
        shortest = float(np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1).min())
        if not degenerate and shortest < TINY_BOX_PX:
            yield _issue("tiny_box", Severity.WARNING, sample, ann, shortest_side_px=shortest)


@register_check(formats={"yolo-obb"}, stage="geometry", scope="dataset")
def obb_winding(dataset: Dataset):
    windings = obb_windings(dataset)
    majority = majority_winding(windings)
    for sample, ann, winding in windings:
        if winding != majority:
            yield _issue("inconsistent_winding", Severity.WARNING, sample, ann, winding=winding, majority=majority)


@register_check(formats={"yolo-seg"}, stage="geometry")
def seg_geometry(sample: Sample):
    if sample.size is None:
        return
    for ann in sample.annotations:
        pts = ann.points * np.array(sample.size)
        if len(pts) < 3:
            yield _issue("invalid_polygon", Severity.ERROR, sample, ann, reason=f"{len(pts)} point(s), need at least 3")
            continue
        poly = Polygon(pts)
        if poly.area < DEGENERATE_AREA_PX:
            yield _issue("invalid_polygon", Severity.ERROR, sample, ann, reason="zero area")
        elif not poly.is_valid:
            yield _issue("invalid_polygon", Severity.ERROR, sample, ann, reason=shapely.is_valid_reason(poly))
        else:
            w_px, h_px = pts.max(axis=0) - pts.min(axis=0)
            if min(w_px, h_px) < TINY_BOX_PX:
                yield _issue("tiny_box", Severity.WARNING, sample, ann, w_px=float(w_px), h_px=float(h_px))
