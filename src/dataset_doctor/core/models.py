from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import numpy as np

EMPTY_FILE = "empty file"  # Sample.image_error for a 0-byte image


class Severity(IntEnum):
    INFO = 0
    WARNING = 1
    ERROR = 2


@dataclass
class Issue:
    code: str              # "self_intersecting_obb"
    severity: Severity
    image: Path | None     # None for a label file that has no image
    label_file: Path | None
    line: int | None       # 1-based line number in the label file
    detail: dict = field(default_factory=dict)


@dataclass
class Annotation:
    line: int              # 1-based line number in the label file
    cls: int
    values: np.ndarray     # normalized values after the class id, as written
    points: np.ndarray     # (N, 2) normalized polygon (HBB: its 4 corners)


@dataclass
class Sample:
    image: Path
    split: str
    label_file: Path | None            # None when the image has no label file
    size: tuple[int, int] | None = None  # (w, h) in pixels; None if unreadable
    image_error: str | None = None
    file_hash: str | None = None
    annotations: list[Annotation] = field(default_factory=list)
    syntax_errors: list[tuple[int, str, dict]] = field(default_factory=list)  # (line, code, detail)


@dataclass
class Dataset:
    root: Path
    fmt: str
    names: dict[int, str] | None   # from data.yaml; None if no data.yaml
    splits: list[str]
    samples: list[Sample]
    orphan_labels: list[Path]      # label files with no matching image
