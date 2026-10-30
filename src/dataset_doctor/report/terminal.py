from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import click

from dataset_doctor import __version__
from dataset_doctor.core.models import Issue, Severity
from dataset_doctor.scanner import ScanResult

COLORS = {Severity.ERROR: "red", Severity.WARNING: "yellow", Severity.INFO: "cyan"}


def print_summary(result: ScanResult, report_path: Path | None = None) -> None:
    stats = result.stats
    click.echo(f"Dataset Doctor v{__version__} — {result.dataset.fmt} — "
               f"{stats['images']:,} images / {stats['labels']:,} labels\n")

    groups: dict[tuple[Severity, str], list[Issue]] = defaultdict(list)
    for issue in result.issues:
        groups[(issue.severity, issue.code)].append(issue)
    if not groups:
        click.echo(" No issues found.")
    for (severity, code), issues in sorted(groups.items(), key=lambda kv: (-kv[0][0], -len(kv[1]), kv[0][1])):
        line = f" {click.style(f'{severity.name:<8}', fg=COLORS[severity])} {code:<26} {len(issues):>5}"
        note = _note(code, issues)
        click.echo(f"{line}   ({note})" if note else line)

    if report_path is not None:
        click.echo(f"\n→ {report_path}")


def _note(code: str, issues: list[Issue]) -> str | None:
    if code == "class_id_out_of_range":
        maxes = [i.detail["max"] for i in issues if "max" in i.detail]
        found = sorted({str(i.detail["class"]) for i in issues}, key=lambda c: (len(c), c))
        shown = ", ".join(found[:5]) + (", …" if len(found) > 5 else "")
        return f"max={max(maxes)}, found {shown}" if maxes else f"found {shown}"
    if code == "coord_out_of_range":
        hi, lo = max(i.detail["max"] for i in issues), min(i.detail["min"] for i in issues)
        parts = ([f"max {hi:.4g}"] if hi > 1 else []) + ([f"min {lo:.4g}"] if lo < 0 else [])
        return ", ".join(parts)
    if code == "split_leak":
        return ", ".join(sorted({" ↔ ".join(i.detail["splits"]) for i in issues}))
    return None
