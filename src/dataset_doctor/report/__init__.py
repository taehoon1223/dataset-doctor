from __future__ import annotations

from pathlib import Path


def relpath(path: Path, root: Path) -> str:
    """Path relative to the dataset root (posix style), or as-is when it lives outside the root."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)
