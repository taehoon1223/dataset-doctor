from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import click

from dataset_doctor import __version__
from dataset_doctor.core.models import Severity
from dataset_doctor.fix import fix_dataset
from dataset_doctor.loaders import FORMATS
from dataset_doctor.report import terminal
from dataset_doctor.report.html import write_html
from dataset_doctor.report.json import write_json
from dataset_doctor.scanner import scan as run_scan

dataset_argument = click.argument("dataset", type=click.Path(exists=True, file_okay=False, path_type=Path))
format_option = click.option("--format", "fmt", required=True, type=click.Choice(list(FORMATS)), help="Label format.")
data_option = click.option("--data", "data_yaml", type=click.Path(exists=True, dir_okay=False, path_type=Path),
                           help="data.yaml with splits and class names (default: DATASET/data.yaml if present).")


@click.group()
@click.version_option(__version__, prog_name="dataset-doctor")
def main():
    """Catch broken YOLO labels before you train on them."""
    # Redirected output on Windows defaults to the ANSI code page (e.g. cp949), which can't encode "—" or "↔".
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


@main.command()
@dataset_argument
@format_option
@data_option
@click.option("--report", "report_kind", type=click.Choice(["terminal", "json", "html"]), default="terminal",
              show_default=True, help="Also write a report file to --out.")
@click.option("--out", type=click.Path(file_okay=False, path_type=Path), default=Path("report"), show_default=True,
              help="Folder for the json/html report.")
@click.option("--fail-on", type=click.Choice(["error", "warning"]),
              help="Exit with status 1 if any issue at or above this severity (for CI).")
def scan(dataset, fmt, data_yaml, report_kind, out, fail_on):
    """Check a dataset and summarize the problems found."""
    try:
        result = run_scan(dataset, fmt, data_yaml)
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    report_path = None
    if report_kind == "json":
        report_path = write_json(result, out)
    elif report_kind == "html":
        report_path = write_html(result, out)
    terminal.print_summary(result, report_path)

    if fail_on:
        threshold = Severity.ERROR if fail_on == "error" else Severity.WARNING
        if any(issue.severity >= threshold for issue in result.issues):
            raise SystemExit(1)


@main.command()
@dataset_argument
@click.option("--out", required=True, type=click.Path(file_okay=False, path_type=Path),
              help="New folder for the repaired copy (must not exist).")
@format_option
@data_option
@click.option("--clip-coords", is_flag=True, help="Clip coordinates into [0, 1].")
@click.option("--reorder-obb", is_flag=True, help="Fix twisted or opposite-winding OBB corner order.")
@click.option("--dedup", is_flag=True, help="Drop duplicate labels, and duplicate images within a split.")
def fix(dataset, out, fmt, data_yaml, clip_coords, reorder_obb, dedup):
    """Write a repaired copy of DATASET to --out. The original is not touched."""
    if not (clip_coords or reorder_obb or dedup):
        raise click.UsageError("nothing to fix: pass --clip-coords, --reorder-obb and/or --dedup")
    try:
        changes = fix_dataset(dataset, out, fmt, data_yaml,
                              clip_coords=clip_coords, reorder_obb=reorder_obb, dedup=dedup)
    except (ValueError, FileExistsError) as e:
        raise click.ClickException(str(e)) from e

    counts = Counter(c["action"] for c in changes)
    summary = ", ".join(f"{action} {n}" for action, n in sorted(counts.items())) or "no changes"
    click.echo(f"{summary}\n→ {out / 'fix_log.json'}")
