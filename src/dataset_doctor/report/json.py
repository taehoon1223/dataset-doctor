from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from dataset_doctor import __version__
from dataset_doctor.report import relpath
from dataset_doctor.scanner import ScanResult


def to_dict(result: ScanResult) -> dict:
    root = result.dataset.root

    def path(p: Path | None) -> str | None:
        return None if p is None else relpath(p, root)

    counts = Counter((i.severity, i.code) for i in result.issues)
    return {
        "version": __version__,
        "format": result.dataset.fmt,
        "root": str(root),
        "summary": [{"severity": sev.name, "code": code, "count": n}
                    for (sev, code), n in sorted(counts.items(), key=lambda kv: (-kv[0][0], -kv[1], kv[0][1]))],
        "stats": result.stats,
        "issues": [{"code": i.code, "severity": i.severity.name, "image": path(i.image),
                    "label_file": path(i.label_file), "line": i.line, "detail": i.detail}
                   for i in result.issues],
    }


def write_json(result: ScanResult, out_dir: Path) -> Path:
    root = result.dataset.root

    def default(obj):
        if isinstance(obj, Path):
            return relpath(obj, root)
        raise TypeError(f"{type(obj).__name__} is not JSON serializable")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "report.json"
    path.write_text(json.dumps(to_dict(result), indent=2, ensure_ascii=False, default=default), encoding="utf-8")
    return path
