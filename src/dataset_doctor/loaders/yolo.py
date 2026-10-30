"""Ultralytics dataset layout: images/<split>/... with labels/<split>/... next to it."""
from __future__ import annotations

import hashlib
import io
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from PIL import Image

from dataset_doctor.checks.syntax import parse_label
from dataset_doctor.core.models import EMPTY_FILE, Dataset, Sample
from dataset_doctor.loaders import FORMATS

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLIT_ORDER = ("train", "val", "test")
EXIF_ORIENTATION = 0x0112

# Aerial datasets (e.g. DOTA) have legitimately huge images; don't treat them as decompression bombs.
Image.MAX_IMAGE_PIXELS = None


def label_path(image: Path) -> Path:
    """Ultralytics rule: replace the last /images/ in the path with /labels/ and use .txt."""
    head, sep, tail = str(image).rpartition(f"{os.sep}images{os.sep}")
    path = f"{head}{os.sep}labels{os.sep}{tail}" if sep else str(image)
    return Path(path).with_suffix(".txt")


def load_dataset(root: Path, fmt: str, data_yaml: Path | None = None) -> Dataset:
    root = Path(root)
    if data_yaml is None and (root / "data.yaml").is_file():
        data_yaml = root / "data.yaml"
    if data_yaml is not None:
        names, split_dirs = _read_data_yaml(Path(data_yaml), root)
    else:
        names, split_dirs = None, _discover_splits(root)

    jobs: list[tuple[str, Path]] = []
    orphans: list[Path] = []
    for split, dirs in split_dirs.items():
        for image_dir in dirs:
            images = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())
            jobs += [(split, p) for p in images]
            label_dir = label_path(image_dir / "_.jpg").parent
            if label_dir.is_dir():
                expected = {label_path(p) for p in images}
                orphans += sorted(p for p in label_dir.rglob("*.txt") if p not in expected)

    spec = FORMATS[fmt]
    # Reading, hashing and decoding images is I/O-bound and releases the GIL, so threads are enough.
    with ThreadPoolExecutor() as pool:
        samples = list(pool.map(lambda job: _load_sample(job[0], job[1], spec, names), jobs))
    return Dataset(root=root, fmt=fmt, names=names, splits=list(split_dirs), samples=samples, orphan_labels=orphans)


def _load_sample(split: str, image: Path, spec, names: dict[int, str] | None) -> Sample:
    sample = Sample(image=image, split=split, label_file=None)
    data = image.read_bytes()
    if not data:
        sample.image_error = EMPTY_FILE
    else:
        sample.file_hash = hashlib.md5(data).hexdigest()
        try:
            with Image.open(io.BytesIO(data)) as im:
                im.verify()
            with Image.open(io.BytesIO(data)) as im:  # verify() leaves the image unusable, so reopen
                w, h = im.size
                if im.getexif().get(EXIF_ORIENTATION) in (5, 6, 7, 8):  # rotated 90°: training sees w/h swapped
                    w, h = h, w
                fmt = im.format
            if fmt == "JPEG" and not data.rstrip(b"\x00").endswith(b"\xff\xd9"):
                raise OSError("truncated JPEG (missing end-of-image marker)")
            sample.size = (w, h)
        except Exception as e:  # Pillow raises many unrelated exception types for broken files
            sample.image_error = f"{type(e).__name__}: {e}"

    label = label_path(image)
    if label.is_file():
        sample.label_file = label
        text = label.read_text(encoding="utf-8", errors="replace")
        sample.annotations, sample.syntax_errors = parse_label(text, spec, names)
    return sample


def _read_data_yaml(path: Path, root: Path) -> tuple[dict[int, str] | None, dict[str, list[Path]]]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = cfg.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    elif isinstance(names, dict):
        names = {int(k): str(v) for k, v in names.items()}
    elif "nc" in cfg:
        names = {i: str(i) for i in range(int(cfg["nc"]))}

    splits: dict[str, list[Path]] = {}
    for split in SPLIT_ORDER:
        value = cfg.get(split)
        if not value:
            continue
        dirs = [root / d for d in ([value] if isinstance(value, str) else value)]  # absolute paths stay absolute
        for d in dirs:
            if not d.is_dir():
                raise ValueError(f"data.yaml '{split}': {d} is not a directory (image list files are not supported)")
        splits[split] = dirs
    if not splits:
        raise ValueError(f"{path}: no train/val/test entries")
    return names, splits


def _discover_splits(root: Path) -> dict[str, list[Path]]:
    images_dir = root / "images"
    subdirs = [d for d in images_dir.iterdir() if d.is_dir()] if images_dir.is_dir() else []
    if not subdirs:
        raise ValueError(f"expected Ultralytics layout {images_dir}{os.sep}<split>{os.sep} (or pass --data)")
    order = {name: i for i, name in enumerate(SPLIT_ORDER)}
    subdirs.sort(key=lambda d: (order.get(d.name, len(order)), d.name))
    return {d.name: [d] for d in subdirs}
