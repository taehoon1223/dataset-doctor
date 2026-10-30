"""④ Consistency between labels, and dataset statistics."""
from __future__ import annotations

from collections import Counter

import numpy as np
from shapely import STRtree
from shapely.geometry import Polygon

from dataset_doctor.checks.geometry import signed_area
from dataset_doctor.core.models import Dataset, Issue, Sample, Severity
from dataset_doctor.core.registry import register_check

DUPLICATE_IOU = 0.95
COCO_SMALL_PX2, COCO_MEDIUM_PX2 = 32**2, 96**2


@register_check(stage="stats")
def overlapping_labels(sample: Sample):
    # IoU doesn't change when x and y are scaled to pixels, so normalized coordinates are fine here.
    items = [(a, Polygon(a.points)) for a in sample.annotations if len(a.points) >= 3]
    items = [(a, p) for a, p in items if p.is_valid and p.area > 0]
    if len(items) < 2:
        return
    polys = [p for _, p in items]
    flagged: set[int] = set()
    for i, j in sorted(zip(*STRtree(polys).query(polys, predicate="intersects"))):
        if i >= j:
            continue
        (a, pa), (b, pb) = items[i], items[j]
        if b.line in flagged:
            continue
        inter = pa.intersection(pb).area
        iou = inter / (pa.area + pb.area - inter)
        if iou < DUPLICATE_IOU:
            continue
        flagged.add(b.line)
        if a.cls == b.cls:
            yield Issue("duplicate_label", Severity.WARNING, sample.image, sample.label_file, b.line,
                        {"duplicate_of_line": a.line, "iou": round(iou, 4)})
        else:
            yield Issue("label_conflict", Severity.WARNING, sample.image, sample.label_file, b.line,
                        {"other_line": a.line, "classes": [a.cls, b.cls], "iou": round(iou, 4)})


def _quantiles(values) -> dict | None:
    if len(values) == 0:
        return None
    q = np.percentile(values, [0, 25, 50, 75, 100])
    return dict(zip(("min", "p25", "median", "p75", "max"), (float(v) for v in q)))


def compute_stats(dataset: Dataset) -> dict:
    names = dataset.names or {}
    class_counts = Counter({cls: 0 for cls in names})
    class_counts.update(a.cls for s in dataset.samples for a in s.annotations)
    present = [n for n in class_counts.values() if n > 0]

    areas = np.array([abs(signed_area(a.points * np.array(s.size)))
                      for s in dataset.samples if s.size is not None for a in s.annotations])
    objects = [len(s.annotations) for s in dataset.samples]
    resolutions = Counter(s.size for s in dataset.samples if s.size is not None)
    aspect = [w / h for (w, h), n in resolutions.items() for _ in range(n)]

    return {
        "images": len(dataset.samples),
        "labels": sum(objects),
        "classes": [{"id": c, "name": names.get(c), "count": n} for c, n in sorted(class_counts.items())],
        "imbalance_ratio": max(present) / min(present) if present else None,
        "box_size": {
            "small": int((areas < COCO_SMALL_PX2).sum()),
            "medium": int(((areas >= COCO_SMALL_PX2) & (areas < COCO_MEDIUM_PX2)).sum()),
            "large": int((areas >= COCO_MEDIUM_PX2).sum()),
            "sqrt_area_px": _quantiles(np.sqrt(areas)),
        },
        "objects_per_image": {"mean": float(np.mean(objects)) if objects else None, **(_quantiles(objects) or {})},
        "resolutions": [{"width": w, "height": h, "count": n} for (w, h), n in resolutions.most_common(10)],
        "aspect_ratio": _quantiles(aspect),
    }
