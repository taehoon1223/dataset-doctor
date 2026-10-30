"""② Line-level syntax: column count, number parsing, class ids."""
from __future__ import annotations

import math

import numpy as np

from dataset_doctor.core.models import Annotation, Issue, Sample, Severity
from dataset_doctor.core.registry import register_check


def parse_label(text: str, spec, names: dict[int, str] | None) -> tuple[list[Annotation], list[tuple[int, str, dict]]]:
    """Parse a label file into annotations plus (line, code, detail) for every broken line.

    A line with an out-of-range class id is still returned as an annotation so its geometry gets checked.
    """
    annotations: list[Annotation] = []
    errors: list[tuple[int, str, dict]] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        tokens = raw.split()
        if not tokens:
            continue
        try:
            numbers = [float(t) for t in tokens]
        except ValueError:
            numbers = []
        if not numbers or not all(map(math.isfinite, numbers)):
            errors.append((line_no, "parse_error", {"text": raw.strip()}))
            continue
        values = np.array(numbers[1:])
        if not spec.valid_count(len(values)):
            errors.append((line_no, "wrong_column_count", {"expected": spec.EXPECTED, "found": len(values)}))
            continue
        if not numbers[0].is_integer():
            errors.append((line_no, "class_id_out_of_range", {"class": tokens[0], "reason": "not an integer"}))
            continue
        cls = int(numbers[0])
        if cls < 0 or (names is not None and cls not in names):
            detail = {"class": cls} if names is None else {"class": cls, "max": max(names)}
            errors.append((line_no, "class_id_out_of_range", detail))
        annotations.append(Annotation(line_no, cls, values, spec.to_points(values)))
    return annotations, errors


@register_check(stage="syntax")
def label_syntax(sample: Sample):
    for line, code, detail in sample.syntax_errors:
        yield Issue(code, Severity.ERROR, sample.image, sample.label_file, line, detail)
