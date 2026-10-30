import pytest
from fixtures import make_dataset

from dataset_doctor import scan
from dataset_doctor.core.models import Severity


def test_obb_with_seven_values(tmp_path):
    ds = make_dataset(tmp_path, "yolo-obb")
    ds.add_label("a", "0 0.4 0.4 0.6 0.4 0.6 0.6 0.4")
    [issue] = scan(ds.root, ds.fmt).issues
    assert (issue.code, issue.severity, issue.line, issue.detail["found"]) == (
        "wrong_column_count", Severity.ERROR, 1, 7)


def test_seg_with_odd_coordinate_count(tmp_path):
    ds = make_dataset(tmp_path, "yolo-seg")
    ds.add_label("a", "0 0.4 0.4 0.6 0.4 0.6 0.6 0.4")
    [issue] = scan(ds.root, ds.fmt).issues
    assert issue.code == "wrong_column_count"


@pytest.mark.parametrize("bad", ["0 abc 0.5 0.2 0.2", "0 nan 0.5 0.2 0.2", "0 0.5 inf 0.2 0.2", "0 1e999 0.5 0.2 0.2"])
def test_parse_error(tmp_path, bad):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", bad)
    [issue] = scan(ds.root, ds.fmt).issues
    assert (issue.code, issue.severity, issue.line) == ("parse_error", Severity.ERROR, 1)


def test_line_numbers_count_blank_lines(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", "0 0.5 0.5 0.2 0.2\n\nbroken")
    [issue] = scan(ds.root, ds.fmt).issues
    assert (issue.code, issue.line) == ("parse_error", 3)


def test_class_id_beyond_data_yaml(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", "1 0.5 0.5 0.2 0.2\n7 0.3 0.3 0.1 0.1")
    ds.write_data_yaml(["cat", "dog"])
    [issue] = scan(ds.root, ds.fmt).issues
    assert (issue.code, issue.line, issue.detail) == ("class_id_out_of_range", 2, {"class": 7, "max": 1})


def test_explicit_data_yaml_path(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", "3 0.5 0.5 0.2 0.2")
    data = tmp_path / "elsewhere.yaml"
    data.write_text("train: images/train\nnames: {0: cat, 1: dog}\n")
    assert [i.code for i in scan(ds.root, ds.fmt).issues] == []  # no data.yaml in root: range unknown
    assert [i.code for i in scan(ds.root, ds.fmt, data).issues] == ["class_id_out_of_range"]


def test_non_integer_class(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", "1.5 0.5 0.5 0.2 0.2\n1.0 0.5 0.5 0.2 0.2")
    [issue] = scan(ds.root, ds.fmt).issues
    assert (issue.code, issue.line, issue.detail["class"]) == ("class_id_out_of_range", 1, "1.5")


def test_negative_class_without_data_yaml(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", "-1 0.5 0.5 0.2 0.2")
    [issue] = scan(ds.root, ds.fmt).issues
    assert issue.code == "class_id_out_of_range"


def test_out_of_range_class_still_gets_geometry_checks(tmp_path):
    ds = make_dataset(tmp_path, "yolo-hbb")
    ds.add_label("a", "9 0.5 0.5 0 0.2")
    ds.write_data_yaml(["cat"])
    assert sorted(i.code for i in scan(ds.root, ds.fmt).issues) == ["class_id_out_of_range", "non_positive_size"]
