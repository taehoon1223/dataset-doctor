from __future__ import annotations

import os
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

from jinja2 import Environment, FileSystemLoader

from dataset_doctor import __version__
from dataset_doctor.report import relpath
from dataset_doctor.scanner import ScanResult


def write_html(result: ScanResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = Environment(loader=FileSystemLoader(Path(__file__).parent), autoescape=True)
    html = env.get_template("report.html.j2").render(**_context(result, out_dir.resolve()))
    path = out_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def _image_src(image: Path, out_dir: Path) -> str:
    try:
        return quote(Path(os.path.relpath(image.resolve(), out_dir)).as_posix())
    except ValueError:  # Windows: image and report on different drives
        return image.resolve().as_uri()


def _format_detail(detail: dict, root: Path) -> str:
    def fmt(v):
        if isinstance(v, Path):
            return relpath(v, root)
        if isinstance(v, float):
            return f"{v:.4g}"
        if isinstance(v, list):
            return " ↔ ".join(map(fmt, v))
        return str(v)

    return ", ".join(f"{k}={fmt(v)}" for k, v in detail.items())


def _context(result: ScanResult, out_dir: Path) -> dict:
    root = result.dataset.root
    samples = {s.image: s for s in result.dataset.samples}

    grouped = defaultdict(list)  # image (or orphan label file) -> its issues
    for issue in result.issues:
        grouped[issue.image or issue.label_file].append(issue)

    cards = []
    for key, issues in grouped.items():
        sample = samples.get(key)
        line_codes = defaultdict(set)
        for issue in issues:
            if issue.line is not None:
                line_codes[issue.line].add(issue.code)
        polygons = [] if sample is None else [
            {"points": " ".join(f"{x:.5f},{y:.5f}" for x, y in ann.points),
             "codes": " ".join(sorted(line_codes[ann.line])), "line": ann.line, "cls": ann.cls}
            for ann in sample.annotations]
        cards.append({
            "title": relpath(key, root),
            "src": _image_src(key, out_dir) if sample is not None else None,
            "codes": " ".join(sorted({i.code for i in issues})),
            "severity": max(i.severity for i in issues),
            "polygons": polygons,
            "issues": [{"code": i.code, "severity": i.severity.name, "line": i.line,
                        "detail": _format_detail(i.detail, root)} for i in issues],
        })
    cards.sort(key=lambda c: (-c["severity"], c["title"]))

    counts = Counter((i.severity, i.code) for i in result.issues)
    summary = [{"severity": sev.name, "code": code, "count": n}
               for (sev, code), n in sorted(counts.items(), key=lambda kv: (-kv[0][0], -kv[1], kv[0][1]))]
    # One filter per code (a code can appear at two severities); the dot shows the worst one.
    filters: dict[str, dict] = {}
    for row in summary:
        f = filters.setdefault(row["code"], {"code": row["code"], "severity": row["severity"], "count": 0})
        f["count"] += row["count"]

    stats = result.stats
    names = {c["id"]: c["name"] for c in stats["classes"]}
    classes = sorted(stats["classes"], key=lambda c: -c["count"])
    class_max = max((c["count"] for c in classes), default=0) or 1
    sizes = [(k, stats["box_size"][k]) for k in ("small", "medium", "large")]
    size_max = max(n for _, n in sizes) or 1

    return {
        "version": __version__,
        "fmt": result.dataset.fmt,
        "root": str(root),
        "stats": stats,
        "summary": summary,
        "filters": list(filters.values()),
        "cards": cards,
        "class_names": names,
        "classes": [{**c, "pct": 100 * c["count"] / class_max} for c in classes],
        "sizes": [{"name": k, "count": n, "pct": 100 * n / size_max} for k, n in sizes],
    }
