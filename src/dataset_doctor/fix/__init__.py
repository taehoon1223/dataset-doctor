"""Write a repaired copy of a dataset. The source is never modified and every change is logged to
fix_log.json. Anything that needs a judgement call (class conflicts, split leaks, ...) is left alone."""
from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from dataset_doctor.checks.geometry import majority_winding, obb_windings, signed_area
from dataset_doctor.loaders.yolo import SPLIT_ORDER
from dataset_doctor.scanner import scan

REORDER_CODES = {"self_intersecting_obb", "inconsistent_winding"}


def fix_dataset(root: Path, out: Path, fmt: str, data_yaml: Path | None = None, *,
                clip_coords: bool = False, reorder_obb: bool = False, dedup: bool = False) -> list[dict]:
    root, out = Path(root).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"{out} already exists")
    if out.is_relative_to(root):
        raise ValueError(f"output folder {out} must not be inside the dataset")

    config_text = _copy_config(root, data_yaml)
    result = scan(root, fmt, data_yaml)
    paths = set(result.dataset.orphan_labels)
    for sample in result.dataset.samples:
        paths.add(sample.image)
        if sample.label_file is not None:
            paths.add(sample.label_file)
    # Validate every image and label before creating the output, using copytree's layout.
    relative_paths = {path: _copy_relative_path(root, path) for path in paths}
    shutil.copytree(root, out)

    def target(path: Path) -> Path:
        return out / relative_paths[path]

    changes: list[dict] = []
    if config_text is not None:
        config = out / "data.yaml"
        before = config.read_text(encoding="utf-8") if config.is_file() else None
        if before != config_text:
            config.write_text(config_text, encoding="utf-8")
            changes.append({"action": "write_config", "file": "data.yaml", "before": before,
                            "after": config_text, "reason": "copy_dataset_paths"})

    removed: set[Path] = set()
    if dedup:
        groups = defaultdict(dict)
        image_splits = defaultdict(set)
        for sample in result.dataset.samples:
            image_splits[target(sample.image)].add(sample.split)
            if sample.file_hash:
                groups[(sample.split, sample.file_hash)][target(sample.image)] = sample
        removals = []
        for group in groups.values():
            if len(group) < 2:
                continue
            samples = sorted(group.values(), key=lambda s: (s.label_file is None, s.image))
            labels = {s.label_file.read_bytes() for s in samples if s.label_file is not None}
            if len(labels) > 1:  # Conflicting annotations, including explicit background labels: keep all.
                continue
            removals.extend((sample, samples[0]) for sample in samples[1:]
                            if len(image_splits[target(sample.image)]) == 1)
        removed = {target(s.image) for s, _ in removals}
        kept_labels = {target(s.label_file) for s in result.dataset.samples
                       if target(s.image) not in removed and s.label_file is not None}
        deleted_labels = set()
        for sample, first in removals:
            target(sample.image).unlink()
            label = sample.label_file
            if label is not None and target(label) not in kept_labels and target(label) not in deleted_labels:
                target(label).unlink()
                deleted_labels.add(target(label))
            else:
                label = None
            changes.append({"action": "remove_image", "image": relative_paths[sample.image].as_posix(),
                            "label_file": relative_paths[label].as_posix() if label else None,
                            "reason": "duplicate_image", "duplicate_of": relative_paths[first.image].as_posix()})

    codes_by_line: dict[tuple[Path, int], set[str]] = defaultdict(set)
    for issue in result.issues:
        if issue.line is not None:
            codes_by_line[(target(issue.label_file), issue.line)].add(issue.code)
    majority = majority_winding(obb_windings(result.dataset)) if reorder_obb else None

    visited_labels: set[Path] = set()
    for sample in result.dataset.samples:
        if sample.label_file is None or target(sample.image) in removed:
            continue
        label_target = target(sample.label_file)
        if label_target in visited_labels:
            continue
        visited_labels.add(label_target)
        lines: list[str | None] = sample.label_file.read_text(encoding="utf-8", errors="replace").splitlines()
        changed = False
        for ann in sample.annotations:
            codes = codes_by_line.get((label_target, ann.line), set())
            entry = {"label_file": relative_paths[sample.label_file].as_posix(),
                     "line": ann.line, "before": lines[ann.line - 1]}
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
            label_target.write_text("".join(f"{line}\n" for line in lines if line is not None), encoding="utf-8")

    log = {"source": str(root), "format": fmt, "note": "line numbers refer to the source label files",
           "changes": changes}
    (out / "fix_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    return changes


def _copy_relative_path(root: Path, path: Path) -> Path:
    """Map a source path to its copied location without following links inside the dataset."""
    if ".." in path.parts:
        prefix = Path(path.anchor)
        for part in path.parts[1:]:
            if part == ".." and prefix.resolve() != Path(os.path.normpath(prefix)):
                raise ValueError(f"{path}: a link followed by '..' is ambiguous; use a direct dataset path")
            prefix /= part
    normalized = Path(os.path.normpath(path))
    if not normalized.is_relative_to(root):
        normalized = path.resolve()  # Allow an external alias that points back into the dataset.
    if not normalized.is_relative_to(root):
        raise ValueError(f"{path} is outside the dataset; fix requires images and labels inside the dataset")
    return normalized.relative_to(root)


def _copy_config(root: Path, data_yaml: Path | None) -> str | None:
    """Prepare the effective config for the copy, rejecting split folders that won't be copied."""
    path = Path(data_yaml) if data_yaml is not None else root / "data.yaml"
    if data_yaml is None and not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(text) or {}
    # The scanner uses the supplied dataset root. Let training use the copied YAML's directory too.
    changed = "path" in cfg
    cfg.pop("path", None)
    for split in SPLIT_ORDER:
        value = cfg.get(split)
        if not value:
            continue
        folders = []
        for folder in ([value] if isinstance(value, str) else value):
            relative = _copy_relative_path(root, root / folder)
            # Preserve harmless spelling differences, and therefore comments, in relative configs.
            folders.append(folder if Path(os.path.normpath(folder)) == relative else relative.as_posix())
        replacement = folders[0] if isinstance(value, str) else folders
        changed |= replacement != value
        cfg[split] = replacement
    return yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True) if changed else text


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
