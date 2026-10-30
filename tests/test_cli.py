import json

from click.testing import CliRunner
from fixtures import make_dataset

from dataset_doctor.cli import main

BUTTERFLY = "0 0.1 0.1 0.5 0.5 0.5 0.1 0.1 0.5"
SQUARE = "0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6"


def run(*args):
    return CliRunner().invoke(main, [str(a) for a in args])


def test_scan_prints_summary(tmp_path):
    ds = make_dataset(tmp_path, "yolo-obb")
    ds.add_label("a", f"{SQUARE}\n7 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n{BUTTERFLY}")
    ds.add_label("b", "0 0.4 0.4 1.032 0.4 1.032 0.6 0.4 0.6", split="val")
    ds.write_data_yaml(["plane", "ship", "car", "truck", "boat"], splits=("train", "val"))
    result = run("scan", ds.root, "--format", "yolo-obb")
    assert result.exit_code == 0, result.output
    assert "yolo-obb — 2 images / 4 labels" in result.output
    assert "class_id_out_of_range" in result.output and "(max=4, found 7)" in result.output
    assert "self_intersecting_obb" in result.output
    assert "(max 1.032)" in result.output


def test_scan_requires_format(tmp_path):
    ds = make_dataset(tmp_path, "yolo-obb")
    ds.add_label("a", SQUARE)
    result = run("scan", ds.root)
    assert result.exit_code == 2 and "--format" in result.output


def test_fail_on(tmp_path):
    ds = make_dataset(tmp_path, "yolo-obb")
    ds.add_label("a", "0 0.4 0.4 1.03 0.4 1.03 0.6 0.4 0.6")  # a WARNING only
    assert run("scan", ds.root, "--format", "yolo-obb", "--fail-on", "error").exit_code == 0
    assert run("scan", ds.root, "--format", "yolo-obb", "--fail-on", "warning").exit_code == 1
    ds.add_label("b", BUTTERFLY)
    assert run("scan", ds.root, "--format", "yolo-obb", "--fail-on", "error").exit_code == 1


def test_bad_layout_is_a_clean_error(tmp_path):
    result = run("scan", tmp_path, "--format", "yolo-hbb")
    assert result.exit_code == 1 and "Ultralytics layout" in result.output


def test_json_report(tmp_path):
    ds = make_dataset(tmp_path, "yolo-obb")
    ds.add_label("a", BUTTERFLY)
    out = tmp_path / "report"
    result = run("scan", ds.root, "--format", "yolo-obb", "--report", "json", "--out", out)
    assert result.exit_code == 0, result.output
    data = json.loads((out / "report.json").read_text(encoding="utf-8"))
    issue = next(i for i in data["issues"] if i["code"] == "self_intersecting_obb")
    assert issue == {"code": "self_intersecting_obb", "severity": "ERROR", "image": "images/train/a.png",
                     "label_file": "labels/train/a.txt", "line": 1, "detail": {}}
    assert data["stats"]["labels"] == 1


def test_html_report(tmp_path):
    ds = make_dataset(tmp_path, "yolo-obb")
    ds.add_label("a", f"{SQUARE}\n{BUTTERFLY}")
    ds.add_label("b", "0 0.4 0.4 1.03 0.4 1.03 0.6 0.4 0.6")  # coord_out_of_range as WARNING ...
    ds.add_label("c", "0 0.4 0.4 1.5 0.4 1.5 0.6 0.4 0.6")    # ... and as ERROR
    ds.label_path("ghost").write_text(SQUARE)
    out = tmp_path / "report"
    result = run("scan", ds.root, "--format", "yolo-obb", "--report", "html", "--out", out)
    assert result.exit_code == 0, result.output
    assert f"→ {out / 'index.html'}" in result.output
    html = (out / "index.html").read_text(encoding="utf-8")
    assert html.count('data-code="self_intersecting_obb"') == 1
    assert html.count('data-code="coord_out_of_range"') == 1  # one filter chip per code
    assert 'src="../dataset/images/train/a.png"' in html
    assert html.count("<polygon") == 4 and html.count('class="bad"') == 3
    assert "labels/train/ghost.txt" in html


def test_fix_requires_an_action(tmp_path):
    ds = make_dataset(tmp_path, "yolo-obb")
    ds.add_label("a", SQUARE)
    result = run("fix", ds.root, "--out", tmp_path / "fixed", "--format", "yolo-obb")
    assert result.exit_code == 2 and "nothing to fix" in result.output


def test_fix_command(tmp_path):
    ds = make_dataset(tmp_path, "yolo-obb")
    ds.add_label("a", BUTTERFLY)
    result = run("fix", ds.root, "--out", tmp_path / "fixed", "--format", "yolo-obb", "--reorder-obb")
    assert result.exit_code == 0, result.output
    assert "edit_line 1" in result.output
    assert (tmp_path / "fixed" / "fix_log.json").is_file()
