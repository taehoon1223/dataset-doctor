import numpy as np
from fixtures import make_dataset

from dataset_doctor import scan
from dataset_doctor.checks.geometry import obb_max_angle_deviation
from dataset_doctor.core.models import Severity


def codes(issues):
    return sorted(i.code for i in issues)


def obb_line(cls, pts):
    return f"{cls} " + " ".join(f"{v:.6f}" for v in np.asarray(pts).ravel())


def test_coord_slightly_out_of_range_is_warning(tmp_path):
    ds = make_dataset(tmp_path, "yolo-obb")
    ds.add_label("a", "0 0.4 0.4 1.03 0.4 1.03 0.6 0.4 0.6")
    [issue] = scan(ds.root, ds.fmt).issues
    assert (issue.code, issue.severity, issue.detail["max"]) == ("coord_out_of_range", Severity.WARNING, 1.03)


def test_coord_far_out_of_range_is_error(tmp_path):
    ds = make_dataset(tmp_path, "yolo-seg")
    ds.add_label("a", "0 -0.2 0.4 0.6 0.4 0.6 0.6 -0.2 0.6")
    [issue] = scan(ds.root, ds.fmt).issues
    assert (issue.code, issue.severity, issue.detail["min"]) == ("coord_out_of_range", Severity.ERROR, -0.2)


def test_hbb_coords_are_checked_at_the_box_edges(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", "0 0.95 0.5 0.12 0.2")  # right edge at 1.01
    [issue] = scan(ds.root, ds.fmt).issues
    assert issue.code == "coord_out_of_range" and abs(issue.detail["max"] - 1.01) < 1e-9


def test_non_positive_size(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", "0 0.5 0.5 0 0.2\n0 0.5 0.5 0.2 -0.1")
    issues = scan(ds.root, ds.fmt).issues
    assert [(i.code, i.severity, i.line) for i in issues] == [
        ("non_positive_size", Severity.ERROR, 1), ("non_positive_size", Severity.ERROR, 2)]


def test_tiny_hbb_uses_pixels(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", "0 0.5 0.5 0.002 0.2", size=(640, 640))   # 1.28 px wide
    ds.add_label("b", "0 0.5 0.5 0.002 0.2", size=(4000, 640))  # 8 px wide
    [issue] = scan(ds.root, ds.fmt).issues
    assert (issue.code, issue.image.stem) == ("tiny_box", "a")


def test_detects_butterfly_obb(tmp_path):
    ds = make_dataset(tmp_path, fmt="yolo-obb")
    ds.add_label("img_001", "0 0.1 0.1 0.5 0.5 0.5 0.1 0.1 0.5")  # vertex order twisted
    issues = scan(ds.root, ds.fmt).issues
    assert any(i.code == "self_intersecting_obb" and i.line == 1 for i in issues)


def test_degenerate_obb(tmp_path):
    ds = make_dataset(tmp_path, "yolo-obb")
    ds.add_label("a", "0 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5")
    assert codes(scan(ds.root, ds.fmt).issues) == ["degenerate_obb"]


def test_non_rectangular_obb(tmp_path):
    ds = make_dataset(tmp_path, "yolo-obb")
    ds.add_label("a", "0 0.3 0.3 0.6 0.3 0.7 0.6 0.4 0.6")  # parallelogram
    [issue] = scan(ds.root, ds.fmt).issues
    assert issue.code == "non_rectangular_obb" and issue.detail["max_angle_deviation"] > 10


def test_rotated_rectangle_on_wide_image_is_rectangular(tmp_path):
    # A 30° rotated rectangle on 1920x1080 is skewed in normalized coordinates but square-cornered in pixels.
    w, h = 1920, 1080
    t = np.deg2rad(30)
    corners = np.array([[-200, -100], [200, -100], [200, 100], [-200, 100]])
    rotated = corners @ np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]]) + [960, 540]
    normalized = rotated / [w, h]
    assert obb_max_angle_deviation(normalized) > 10

    ds = make_dataset(tmp_path, "yolo-obb")
    ds.add_label("a", obb_line(0, normalized), size=(w, h))
    assert scan(ds.root, ds.fmt).issues == []


def test_tiny_obb(tmp_path):
    ds = make_dataset(tmp_path, "yolo-obb")
    ds.add_label("a", "0 0.4 0.4 0.6 0.4 0.6 0.402 0.4 0.402")  # 1.28 px tall
    assert codes(scan(ds.root, ds.fmt).issues) == ["tiny_box"]


def test_inconsistent_winding(tmp_path):
    clockwise = "0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6"
    counter_clockwise = "0 0.1 0.1 0.1 0.3 0.3 0.3 0.3 0.1"
    ds = make_dataset(tmp_path, "yolo-obb")
    ds.add_label("a", clockwise)
    ds.add_label("b", f"{clockwise}\n{counter_clockwise}")
    [issue] = scan(ds.root, ds.fmt).issues
    assert (issue.code, issue.severity, issue.image.stem, issue.line) == (
        "inconsistent_winding", Severity.WARNING, "b", 2)
    assert issue.detail == {"winding": "ccw", "majority": "cw"}


def test_invalid_polygons(tmp_path):
    ds = make_dataset(tmp_path, "yolo-seg")
    ds.add_label("a", "\n".join([
        "0 0.1 0.1 0.5 0.5",                       # 2 points
        "0 0.1 0.1 0.3 0.3 0.5 0.5",               # collinear: zero area
        "0 0.1 0.1 0.6 0.5 0.6 0.1 0.1 0.3",       # bow-tie: self-intersecting
    ]))
    issues = scan(ds.root, ds.fmt).issues
    assert [(i.code, i.severity, i.line) for i in issues] == [
        ("invalid_polygon", Severity.ERROR, 1), ("invalid_polygon", Severity.ERROR, 2),
        ("invalid_polygon", Severity.ERROR, 3)]
    assert "point" in issues[0].detail["reason"]
    assert issues[1].detail["reason"] == "zero area"
    assert "Self-intersection" in issues[2].detail["reason"]


def test_tiny_polygon(tmp_path):
    ds = make_dataset(tmp_path, "yolo-seg")
    ds.add_label("a", "0 0.4 0.4 0.6 0.4 0.6 0.402 0.4 0.402")
    assert codes(scan(ds.root, ds.fmt).issues) == ["tiny_box"]


def test_corrupt_image_skips_pixel_checks(tmp_path):
    ds = make_dataset(tmp_path, "yolo-obb")
    ds.add_label("a", "0 0.1 0.1 0.5 0.5 0.5 0.1 0.1 0.5")
    ds.image_path("a").write_bytes(b"broken")
    assert codes(scan(ds.root, ds.fmt).issues) == ["corrupt_image"]
