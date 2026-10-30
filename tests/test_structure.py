import io
import shutil

import pytest
from fixtures import make_dataset
from PIL import Image

from dataset_doctor import scan
from dataset_doctor.core.models import Severity

CLEAN_LINES = {
    "yolo-hbb": "0 0.5 0.5 0.2 0.2",
    "yolo-obb": "0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6",
    "yolo-seg": "0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6",
}


@pytest.mark.parametrize("fmt", CLEAN_LINES)
def test_clean_dataset_has_no_issues(tmp_path, fmt):
    ds = make_dataset(tmp_path, fmt)
    ds.add_label("a", CLEAN_LINES[fmt])
    ds.add_label("b", CLEAN_LINES[fmt], split="val")
    assert scan(ds.root, fmt).issues == []


def test_image_without_label(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    image = ds.add_image("a")
    [issue] = scan(ds.root, ds.fmt).issues
    assert (issue.code, issue.severity, issue.image) == ("image_without_label", Severity.INFO, image)


def test_label_without_image(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", CLEAN_LINES["yolo-hbb"])
    orphan = ds.label_path("ghost")
    orphan.write_text(CLEAN_LINES["yolo-hbb"])
    [issue] = scan(ds.root, ds.fmt).issues
    assert (issue.code, issue.severity, issue.image, issue.label_file) == (
        "label_without_image", Severity.ERROR, None, orphan)


def test_empty_label_file_is_info(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", "")
    [issue] = scan(ds.root, ds.fmt).issues
    assert (issue.code, issue.severity, issue.detail) == ("empty_file", Severity.INFO, {"file": "label"})


def test_empty_image_file_is_error(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", CLEAN_LINES["yolo-hbb"])
    ds.image_path("a").write_bytes(b"")
    [issue] = scan(ds.root, ds.fmt).issues
    assert (issue.code, issue.severity, issue.detail) == ("empty_file", Severity.ERROR, {"file": "image"})


def test_corrupt_image(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", CLEAN_LINES["yolo-hbb"])
    ds.image_path("a").write_bytes(b"definitely not a png")
    [issue] = scan(ds.root, ds.fmt).issues
    assert (issue.code, issue.severity) == ("corrupt_image", Severity.ERROR)


def test_truncated_jpeg_is_corrupt(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", CLEAN_LINES["yolo-hbb"])
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 10, 10)).save(buf, "JPEG")
    jpeg = ds.image_path("a", suffix=".jpg")
    jpeg.write_bytes(buf.getvalue()[:-2])  # only the end-of-image marker is missing; Pillow's verify() misses this
    ds.image_path("a").unlink()
    [issue] = scan(ds.root, ds.fmt).issues
    assert issue.code == "corrupt_image" and "end-of-image" in issue.detail["error"]


def test_duplicate_image_in_same_split(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    first = ds.add_image("a")
    copy = ds.image_path("b")
    shutil.copy(first, copy)
    issues = [i for i in scan(ds.root, ds.fmt).issues if i.code != "image_without_label"]
    assert [(i.code, i.image, i.detail) for i in issues] == [("duplicate_image", copy, {"duplicate_of": first})]


def test_split_leak(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", CLEAN_LINES["yolo-hbb"])
    ds.add_label("b", CLEAN_LINES["yolo-hbb"], split="val")
    shutil.copy(ds.image_path("a"), ds.image_path("b", split="val"))
    [issue] = scan(ds.root, ds.fmt).issues
    assert issue.code == "split_leak" and issue.severity == Severity.WARNING
    assert issue.image == ds.image_path("b", split="val")
    assert issue.detail == {"splits": ["train", "val"], "same_as": ds.image_path("a")}


def test_missing_images_dir_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="Ultralytics layout"):
        scan(tmp_path, "yolo-hbb")
