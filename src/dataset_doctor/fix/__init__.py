"""Write a repaired copy of a dataset. The source is never modified and every change is logged to
fix_log.json. Anything that needs a judgement call (class conflicts, split leaks, ...) is left alone."""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from dataset_doctor.checks.geometry import majority_winding, obb_windings, signed_area
from dataset_doctor.report import relpath
from dataset_doctor.scanner import scan

REORDER_CODES = {"self_intersecting_obb", "inconsistent_winding"}


def fix_dataset(root: Path, out: Path, fmt: str, data_yaml: Path | None = None, *,
                clip_coords: bool = False, reorder_obb: bool = False, dedup: bool = False) -> list[dict]:
    root, out = Path(root).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"{out} already exists")
    if out.is_relative_to(root):
        raise ValueError(f"output folder {out} must not be inside the dataset")

    result = scan(root, fmt, data_yaml)
    shutil.copytree(root, out)

    def target(path: Path) -> Path:
        return out / path.relative_to(root)

    changes: list[dict] = []
    removed: set[Path] = set()
    if dedup:
        for issue in result.issues:
            if issue.code != "duplicate_image":
                continue
            target(issue.image).unlink()
            if issue.label_file is not None:
                target(issue.label_file).unlink()
            removed.add(issue.image)
            changes.append({"action": "remove_image", "image": relpath(issue.image, root),
                            "label_file": relpath(issue.label_file, root) if issue.label_file else None,
                            "reason": "duplicate_image", "duplicate_of": relpath(issue.detail["duplicate_of"], root)})

    codes_by_line: dict[tuple[Path, int], set[str]] = defaultdict(set)
    for issue in result.issues:
        if issue.line is not None:
            codes_by_line[(issue.label_file, issue.line)].add(issue.code)
    majority = majority_winding(obb_windings(result.dataset)) if reorder_obb else None

    for sample in result.dataset.samples:
        if sample.label_file is None or sample.image in removed:
            continue
        lines: list[str | None] = sample.label_file.read_text(encoding="utf-8", errors="replace").splitlines()
        changed = False
        for ann in sample.annotations:
            codes = codes_by_line.get((sample.label_file, ann.line), set())
            entry = {"label_file": relpath(sample.label_file, root), "line": ann.line, "before": lines[ann.line - 1]}
            if dedup and "duplicate_label" in codes:
                lines[ann.line - 1] = None
                changes.append({"action": "remove_line", **entry, "reason": "duplicate_label"})
                changed = True
                continue
            values, reasons = ann.values, []
            if clip_coords and "coord_out_of_range" in codes:
                values = _clip(values, fmt)
                reasons.append("coord_out_of_range")
            if reorder_obb and codes & REORDER_CODES:
                values = _reorder_obb(values, majority)
                reasons += sorted(codes & REORDER_CODES)
            if reasons:
                lines[ann.line - 1] = f"{ann.cls} " + " ".join(f"{v:.6g}" for v in values)
                changes.append({"action": "edit_line", **entry, "after": lines[ann.line - 1], "reasons": reasons})
                changed = True
        if changed:
            target(sample.label_file).write_text("".join(f"{line}\n" for line in lines if line is not None),
                                                 encoding="utf-8")

    log = {"source": str(root), "format": fmt, "note": "line numbers refer to the source label files",
           "changes": changes}
    (out / "fix_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    return changes


def _clip(values: np.ndarray, fmt: str) -> np.ndarray:
    if fmt == "yolo-hbb":  # clip the box edges, then convert back to center/size
        cx, cy, w, h = values
        x1, y1, x2, y2 = np.clip([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 0.0, 1.0)
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
    return np.clip(values, 0.0, 1.0)


def _reorder_obb(values: np.ndarray, majority: str | None) -> np.ndarray:
    """Put the 4 corners in convex order with the dataset's majority winding, keeping the first corner first."""
    pts = values.reshape(4, 2)
    center = pts.mean(axis=0)
    idx = np.argsort(np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0]))
    if majority is not None and (signed_area(pts[idx]) > 0) != (majority == "cw"):
        idx = idx[::-1]
    idx = np.roll(idx, -int(np.flatnonzero(idx == 0)[0]))
    return pts[idx].ravel()
