from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import dataset_doctor.checks  # noqa: F401  (registers the checks)
from dataset_doctor.checks.stats import compute_stats
from dataset_doctor.core.models import Dataset, Issue
from dataset_doctor.core.registry import checks_for
from dataset_doctor.loaders import FORMATS
from dataset_doctor.loaders.yolo import load_dataset


@dataclass
class ScanResult:
    dataset: Dataset
    issues: list[Issue]
    stats: dict


def scan(root: str | Path, fmt: str, data_yaml: str | Path | None = None) -> ScanResult:
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; expected one of {', '.join(FORMATS)}")
    dataset = load_dataset(Path(root), fmt, Path(data_yaml) if data_yaml else None)

    issues: list[Issue] = []
    sample_checks = checks_for(fmt, "sample")
    for sample in dataset.samples:
        for check in sample_checks:
            issues.extend(check.func(sample))
    for check in checks_for(fmt, "dataset"):
        issues.extend(check.func(dataset))
    return ScanResult(dataset, issues, compute_stats(dataset))
