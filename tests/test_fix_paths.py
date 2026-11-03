import os
import shutil
import subprocess

import pytest
import yaml
from fixtures import make_dataset

from dataset_doctor import scan
from dataset_doctor.fix import fix_dataset

LABEL = "0 0.9 0.5 0.4 0.2"


def directory_link(link, target):
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        # Windows CI may lack symlink privileges; junctions exercise the same path mapping.
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                       check=True, capture_output=True)


def test_dedup_internal_image_link_matches_log(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_image("a/x")
    directory_link(ds.root / "images/train/b", ds.root / "images/train/a")
    (ds.root / "data.yaml").write_text("train: [images/train/a, images/train/b]\n")
    out = tmp_path / "fixed"

    changes = fix_dataset(ds.root, out, ds.fmt, dedup=True)

    [change] = changes
    assert change["image"] == "images/train/b/x.png"
    assert change["duplicate_of"] == "images/train/a/x.png"
    assert (out / change["duplicate_of"]).is_file()
    assert not (out / change["image"]).exists()
    assert ds.image_path("a/x").is_file()
    assert ds.image_path("b/x").is_file()


@pytest.mark.parametrize("external", [False, True])
def test_clip_linked_labels_updates_copy_only(tmp_path, external):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("x", LABEL)
    target = (tmp_path if external else ds.root) / "actual_labels"
    ds.root.joinpath("labels").rename(target)
    directory_link(ds.root / "labels", target)
    original = (target / "train/x.txt").read_bytes()
    out = tmp_path / "fixed"

    changes = fix_dataset(ds.root, out, ds.fmt, clip_coords=True)

    assert scan(out, ds.fmt).issues == []
    assert changes[0]["label_file"] == "labels/train/x.txt"
    assert (out / "labels/train/x.txt").read_text().strip() == "0 0.85 0.5 0.3 0.2"
    assert (target / "train/x.txt").read_bytes() == original


@pytest.mark.parametrize("with_yaml", [False, True])
def test_dedup_external_image_link(tmp_path, with_yaml):
    ds = make_dataset(tmp_path, "yolo-hbb")
    for name in ("a", "b"):
        ds.add_label(name, "0 0.5 0.5 0.2 0.2")
    shutil.copyfile(ds.image_path("a"), ds.image_path("b"))
    external = tmp_path / "external_images"
    ds.root.joinpath("images/train").rename(external)
    directory_link(ds.root / "images/train", external)
    if with_yaml:
        ds.write_data_yaml(["object"])
    out = tmp_path / "fixed"

    fix_dataset(ds.root, out, ds.fmt, dedup=True)

    assert scan(out, ds.fmt).issues == []
    assert (out / "images/train/a.png").is_file()
    assert not (out / "images/train/b.png").exists()
    assert (external / "b.png").is_file()


@pytest.mark.parametrize("folder", ["images/train/", "./images/train", "images/../images/train"])
def test_relative_yaml_spelling_and_comments_are_preserved(tmp_path, folder):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", LABEL)
    text = f"# Keep my comment\ntrain: {folder}\nnames: [object]\n"
    (ds.root / "data.yaml").write_text(text, encoding="utf-8")
    out = tmp_path / "fixed"

    changes = fix_dataset(ds.root, out, ds.fmt, clip_coords=True)

    assert (out / "data.yaml").read_text(encoding="utf-8") == text
    assert all(c["action"] != "write_config" for c in changes)
    assert scan(out, ds.fmt).issues == []


def test_copy_yaml_does_not_point_training_at_original(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", LABEL)
    config = ds.root / "data.yaml"
    config.write_text(yaml.safe_dump({"path": str(ds.root), "train": "images/train"}))
    original = config.read_bytes()
    out = tmp_path / "fixed"

    fix_dataset(ds.root, out, ds.fmt, clip_coords=True)

    copied = yaml.safe_load((out / "data.yaml").read_text())
    assert "path" not in copied  # Ultralytics defaults to the YAML's directory.
    assert copied["train"] == "images/train"
    assert config.read_bytes() == original
    assert scan(out, ds.fmt).issues == []


def test_dedup_same_files_in_differently_spelled_splits(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    for name in ("a", "b"):
        ds.add_label(name, "0 0.5 0.5 0.2 0.2")
    shutil.copyfile(ds.image_path("a"), ds.image_path("b"))
    (ds.root / "data.yaml").write_text("train: images/train\nval: images/../images/train\n")
    out = tmp_path / "fixed"

    assert fix_dataset(ds.root, out, ds.fmt, dedup=True) == []

    assert (out / "images/train/a.png").is_file()
    assert (out / "images/train/b.png").is_file()


def test_external_label_is_rejected_before_copy(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", LABEL)
    # The last /images/ is the dataset root itself, so labels live outside it.
    root = ds.root / "images"
    config = root / "data.yaml"
    config.write_text("train: train\n")
    out = tmp_path / "fixed"

    with pytest.raises(ValueError, match="outside the dataset"):
        fix_dataset(root, out, ds.fmt, clip_coords=True)

    assert not out.exists()


def test_external_alias_back_into_dataset_is_rebased(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", LABEL)
    alias = tmp_path / "alias"
    directory_link(alias, ds.root)
    (ds.root / "data.yaml").write_text(yaml.safe_dump({"train": str(alias / "images/train")}))
    out = tmp_path / "fixed"

    changes = fix_dataset(ds.root, out, ds.fmt, clip_coords=True)

    assert scan(out, ds.fmt).issues == []
    assert changes[-1]["label_file"] == "labels/train/a.txt"
    assert yaml.safe_load((out / "data.yaml").read_text())["train"] == "images/train"
    assert ds.label_path("a").read_text().strip() == LABEL


def test_shared_label_with_different_path_spellings_is_logged_once(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", LABEL)
    (ds.root / "data.yaml").write_text("train: images/train\nval: images/../images/train\n")
    out = tmp_path / "fixed"

    changes = fix_dataset(ds.root, out, ds.fmt, clip_coords=True)

    assert len(changes) == 1
    assert changes[0]["label_file"] == "labels/train/a.txt"
    assert [i.code for i in scan(out, ds.fmt).issues] == ["split_leak"]


def test_link_followed_by_parent_is_rejected_before_copy(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", LABEL)
    external = tmp_path / "external"
    (external / "child").mkdir(parents=True)
    directory_link(ds.root / "alias", external / "child")
    (ds.root / "data.yaml").write_text("train: alias/../images/train\n")
    out = tmp_path / "fixed"

    with pytest.raises(ValueError, match=r"link.*\.\."):
        fix_dataset(ds.root, out, ds.fmt, clip_coords=True)

    assert not out.exists()
