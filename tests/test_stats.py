from fixtures import make_dataset

from dataset_doctor import scan
from dataset_doctor.core.models import Severity


def test_duplicate_label(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", "0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n0 0.501 0.5 0.2 0.2")
    issues = scan(ds.root, ds.fmt).issues
    assert [(i.code, i.severity, i.line, i.detail["duplicate_of_line"]) for i in issues] == [
        ("duplicate_label", Severity.WARNING, 2, 1), ("duplicate_label", Severity.WARNING, 3, 1)]


def test_label_conflict(tmp_path):
    ds = make_dataset(tmp_path, "yolo-obb")
    ds.add_label("a", "0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6\n2 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6")
    [issue] = scan(ds.root, ds.fmt).issues
    assert (issue.code, issue.line, issue.detail["classes"]) == ("label_conflict", 2, [0, 2])


def test_overlapping_but_distinct_boxes_are_fine(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", "0 0.5 0.5 0.2 0.2\n0 0.52 0.5 0.2 0.2")  # IoU ~0.82
    assert scan(ds.root, ds.fmt).issues == []


def test_compute_stats(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", "0 0.5 0.5 0.02 0.02\n0 0.5 0.5 0.5 0.5\n0 0.2 0.2 0.1 0.1", size=(1000, 500))
    ds.add_label("b", "1 0.5 0.5 0.1 0.1", size=(1000, 500))
    ds.write_data_yaml(["cat", "dog", "bird"])
    stats = scan(ds.root, ds.fmt).stats

    assert stats["images"] == 2 and stats["labels"] == 4
    assert stats["classes"] == [{"id": 0, "name": "cat", "count": 3}, {"id": 1, "name": "dog", "count": 1},
                                {"id": 2, "name": "bird", "count": 0}]
    assert stats["imbalance_ratio"] == 3.0
    # areas in px: 20x10=200 (small), 500x250 (large), 100x50=5000 (medium), 100x50 (medium)
    assert {k: stats["box_size"][k] for k in ("small", "medium", "large")} == {"small": 1, "medium": 2, "large": 1}
    assert stats["objects_per_image"]["mean"] == 2.0 and stats["objects_per_image"]["max"] == 3.0
    assert stats["resolutions"] == [{"width": 1000, "height": 500, "count": 2}]
    assert stats["aspect_ratio"]["median"] == 2.0
