import io
import shutil

import pytest
from fixtures import make_dataset
from PIL import Image, JpegImagePlugin

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


@pytest.mark.parametrize("truncated", [False, True])
def test_bmp_pixel_data_is_decoded(tmp_path, truncated):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", CLEAN_LINES["yolo-hbb"])
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 10, 10)).save(buf, "BMP")
    bmp = ds.image_path("a", suffix=".bmp")
    bmp.write_bytes(buf.getvalue()[:100] if truncated else buf.getvalue())
    ds.image_path("a").unlink()

    result = scan(ds.root, ds.fmt)

    if truncated:
        [issue] = result.issues
        assert (issue.code, issue.severity) == ("corrupt_image", Severity.ERROR)
        assert result.dataset.samples[0].size is None
    else:
        assert result.issues == []
        assert result.dataset.samples[0].size == (64, 64)


@pytest.mark.parametrize("progressive", [False, True])
@pytest.mark.parametrize("orientation", [1, 6])
def test_jpeg_decodes_small_but_preserves_original_size(tmp_path, monkeypatch, progressive, orientation):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", "0 0.5 0.5 0.004 0.004")  # Not tiny at original size, tiny at draft size.
    jpeg = ds.image_path("a", suffix=".jpg")
    exif = Image.Exif()
    exif[0x0112] = orientation
    with Image.new("RGB", (1600, 800), (100, 20, 40)) as im:
        im.save(jpeg, "JPEG", progressive=progressive, exif=exif)
    ds.image_path("a").unlink()
    decoded_sizes = []
    original_load = JpegImagePlugin.JpegImageFile.load

    def record_load(im, *args, **kwargs):
        decoded_sizes.append(im.size)
        return original_load(im, *args, **kwargs)

    monkeypatch.setattr(JpegImagePlugin.JpegImageFile, "load", record_load)

    result = scan(ds.root, ds.fmt)

    assert result.issues == []
    assert result.dataset.samples[0].size == ((800, 1600) if orientation == 6 else (1600, 800))
    assert decoded_sizes and all(w <= 200 and h <= 100 for w, h in decoded_sizes)


@pytest.mark.parametrize("progressive", [False, True])
@pytest.mark.parametrize("damage", ["missing_eoi", "cut_body", "bad_header"])
def test_jpeg_draft_still_detects_corruption(tmp_path, progressive, damage):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", CLEAN_LINES["yolo-hbb"])
    buf = io.BytesIO()
    with Image.new("RGB", (512, 512), (100, 20, 40)) as im:
        im.save(buf, "JPEG", progressive=progressive)
    data = buf.getvalue()
    damaged = {"missing_eoi": data[:-2], "cut_body": data[:len(data) // 2],
               "bad_header": b"bad" + data[3:]}[damage]
    ds.image_path("a", suffix=".jpg").write_bytes(damaged)
    ds.image_path("a").unlink()

    [issue] = scan(ds.root, ds.fmt).issues

    assert (issue.code, issue.severity) == ("corrupt_image", Severity.ERROR)


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
