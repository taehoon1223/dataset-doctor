import json
import shutil

import pytest
import yaml
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
    ds.add_label("b", ds.label_path("a").read_text().strip())
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


@pytest.mark.parametrize("labeled_name", ["a", "b"])
def test_dedup_keeps_labeled_image(tmp_path, labeled_name):
    ds = make_dataset(tmp_path, "yolo-hbb")
    for name in ("a", "b", "c"):
        ds.add_image(name)
    ds.add_label(labeled_name, "0 0.5 0.5 0.2 0.2")
    for name in ("b", "c"):
        shutil.copyfile(ds.image_path("a"), ds.image_path(name))
    out = tmp_path / "fixed"

    changes = fix_dataset(ds.root, out, ds.fmt, dedup=True)

    result = scan(out, ds.fmt)
    assert result.stats["images"] == result.stats["labels"] == 1
    assert result.dataset.samples[0].image.stem == labeled_name
    assert result.issues == []
    assert len(changes) == 2
    assert all(c["duplicate_of"] == f"images/train/{labeled_name}.png" for c in changes)
    assert ds.label_path(labeled_name).exists()


@pytest.mark.parametrize("other_label", ["", "1 0.5 0.5 0.2 0.2", "broken"])
def test_dedup_preserves_different_labels(tmp_path, other_label):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", "0 0.5 0.5 0.2 0.2")
    ds.add_label("b", other_label)
    shutil.copyfile(ds.image_path("a"), ds.image_path("b"))
    out = tmp_path / "fixed"

    assert fix_dataset(ds.root, out, ds.fmt, dedup=True) == []

    for name in ("a", "b"):
        assert (out / f"images/train/{name}.png").exists()
        assert (out / f"labels/train/{name}.txt").read_bytes() == ds.label_path(name).read_bytes()


def test_dedup_preserves_shared_label_file(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", "0 0.5 0.5 0.2 0.2")
    shutil.copyfile(ds.image_path("a"), ds.image_path("a", suffix=".bmp"))
    out = tmp_path / "fixed"

    fix_dataset(ds.root, out, ds.fmt, dedup=True)

    result = scan(out, ds.fmt)
    assert result.stats["images"] == result.stats["labels"] == 1
    assert result.issues == []


def test_dedup_without_labels(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_image("a")
    shutil.copyfile(ds.image_path("a"), ds.image_path("b"))
    out = tmp_path / "fixed"

    changes = fix_dataset(ds.root, out, ds.fmt, dedup=True)

    assert len(changes) == 1 and changes[0]["label_file"] is None
    assert (out / "images/train/a.png").exists()
    assert not (out / "images/train/b.png").exists()


def test_dedup_does_not_remove_files_referenced_by_other_splits(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    for name in ("a", "b"):
        ds.add_label(name, "0 0.5 0.5 0.2 0.2")
    shutil.copyfile(ds.image_path("a"), ds.image_path("b"))
    (ds.root / "data.yaml").write_text("train: images/train\nval: images/train\n")
    out = tmp_path / "fixed"

    assert fix_dataset(ds.root, out, ds.fmt, dedup=True) == []

    assert (out / "images/train/a.png").exists()
    assert (out / "images/train/b.png").exists()


def test_fix_preserves_relative_yaml_and_comments(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", "0 0.9 0.5 0.4 0.2")
    text = "# Dataset classes\nnames: [object]\ntrain: images/train\n"
    (ds.root / "data.yaml").write_text(text, encoding="utf-8")
    out = tmp_path / "fixed"

    changes = fix_dataset(ds.root, out, ds.fmt, clip_coords=True)

    assert (out / "data.yaml").read_text(encoding="utf-8") == text
    assert not any(c["action"] == "write_config" for c in changes)
    assert scan(out, ds.fmt).issues == []


@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("config_location", ["default", "custom", "external"])
def test_fix_rebases_yaml_paths(tmp_path, as_list, config_location):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", "0 0.9 0.5 0.4 0.2")
    train = str(ds.root / "images/train")
    cfg = {"train": [train] if as_list else train, "names": ["object"]}
    config = {"default": ds.root / "data.yaml", "custom": ds.root / "custom.yaml",
              "external": tmp_path / "external.yaml"}[config_location]
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    original = config.read_bytes()
    out = tmp_path / "fixed"

    changes = fix_dataset(ds.root, out, ds.fmt,
                          None if config_location == "default" else config, clip_coords=True)

    result = scan(out, ds.fmt)
    assert result.issues == []
    assert result.dataset.samples[0].image.is_relative_to(out)
    assert yaml.safe_load((out / "data.yaml").read_text())["train"] == (
        ["images/train"] if as_list else "images/train")
    assert any(c["action"] == "write_config" for c in changes)
    assert config.read_bytes() == original
    assert "coord_out_of_range" in [i.code for i in scan(ds.root, ds.fmt, config).issues]


def test_fix_rejects_external_split_before_copying(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", "0 0.9 0.5 0.4 0.2")
    root = tmp_path / "config-only"
    root.mkdir()
    (root / "data.yaml").write_text(yaml.safe_dump({"train": str(ds.root / "images/train")}))
    out = tmp_path / "fixed"

    with pytest.raises(ValueError, match="outside the dataset"):
        fix_dataset(root, out, ds.fmt, clip_coords=True)

    assert not out.exists()
