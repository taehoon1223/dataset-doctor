import json
import shutil

import pytest
from fixtures import make_dataset

from dataset_doctor import scan
from dataset_doctor.fix import fix_dataset

SQUARE = "0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6"


def test_clip_coords(tmp_path):
    ds = make_dataset(tmp_path, "yolo-obb")
    label = ds.add_label("a", f"{SQUARE}\n1 0.1 0.1 1.2 0.1 1.2 0.3 0.1 0.3")
    original = label.read_text()
    out = tmp_path / "fixed"

    changes = fix_dataset(ds.root, out, ds.fmt, clip_coords=True)

    assert label.read_text() == original  # source untouched
    assert (out / "labels/train/a.txt").read_text() == f"{SQUARE}\n1 0.1 0.1 1 0.1 1 0.3 0.1 0.3\n"
    assert changes == [{"action": "edit_line", "label_file": "labels/train/a.txt", "line": 2,
                        "before": "1 0.1 0.1 1.2 0.1 1.2 0.3 0.1 0.3", "after": "1 0.1 0.1 1 0.1 1 0.3 0.1 0.3",
                        "reasons": ["coord_out_of_range"]}]
    assert json.loads((out / "fix_log.json").read_text(encoding="utf-8"))["changes"] == changes
    assert scan(out, ds.fmt).issues == []


def test_clip_hbb_keeps_box_inside_image(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", "0 0.9 0.5 0.4 0.2")  # spans x 0.7..1.1
    out = tmp_path / "fixed"
    fix_dataset(ds.root, out, ds.fmt, clip_coords=True)
    assert (out / "labels/train/a.txt").read_text() == "0 0.85 0.5 0.3 0.2\n"


def test_reorder_obb(tmp_path):
    ds = make_dataset(tmp_path, "yolo-obb")
    ds.add_label("a", SQUARE)                                   # clockwise
    ds.add_label("b", "0 0.1 0.1 0.5 0.5 0.5 0.1 0.1 0.5\n"    # butterfly
                      "0 0.7 0.7 0.7 0.9 0.9 0.9 0.9 0.7")      # counter-clockwise
    out = tmp_path / "fixed"
    changes = fix_dataset(ds.root, out, ds.fmt, reorder_obb=True)

    assert [(c["line"], c["reasons"]) for c in changes] == [
        (1, ["self_intersecting_obb"]), (2, ["inconsistent_winding"])]
    lines = (out / "labels/train/b.txt").read_text().splitlines()
    assert lines[0] == "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5"  # first corner kept, now clockwise
    assert lines[1] == "0 0.7 0.7 0.9 0.7 0.9 0.9 0.7 0.9"
    assert scan(out, ds.fmt).issues == []


def test_dedup(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", "0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1")
    ds.add_label("b", "0 0.3 0.3 0.1 0.1")
    shutil.copy(ds.image_path("a"), ds.image_path("b"))           # b is a duplicate image of a
    ds.add_label("c", "0 0.3 0.3 0.1 0.1", split="val")
    shutil.copy(ds.image_path("a"), ds.image_path("c", "val"))    # leak: report only
    out = tmp_path / "fixed"

    changes = fix_dataset(ds.root, out, ds.fmt, dedup=True)

    assert sorted(c["action"] for c in changes) == ["remove_image", "remove_line"]
    assert not (out / "images/train/b.png").exists() and not (out / "labels/train/b.txt").exists()
    assert ds.image_path("b").exists()
    assert (out / "labels/train/a.txt").read_text() == "0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1\n"
    assert [i.code for i in scan(out, ds.fmt).issues] == ["split_leak"]


def test_out_must_not_exist_or_be_inside_dataset(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", "0 0.5 0.5 0.2 0.2")
    with pytest.raises(FileExistsError):
        fix_dataset(ds.root, tmp_path, ds.fmt, dedup=True)
    with pytest.raises(ValueError, match="inside the dataset"):
        fix_dataset(ds.root, ds.root / "fixed", ds.fmt, dedup=True)
