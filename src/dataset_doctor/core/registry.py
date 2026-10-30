from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from dataset_doctor.core.models import Issue

STAGES = ("structure", "syntax", "geometry", "stats")


@dataclass(frozen=True)
class Check:
    func: Callable[..., Iterable[Issue]]
    formats: frozenset[str] | None  # None = every format
    stage: str
    scope: str                      # "sample": func(Sample), "dataset": func(Dataset)


_CHECKS: list[Check] = []


def register_check(*, stage: str, formats: set[str] | None = None, scope: str = "sample"):
    assert stage in STAGES and scope in ("sample", "dataset")

    def decorator(func):
        _CHECKS.append(Check(func, frozenset(formats) if formats else None, stage, scope))
        return func

    return decorator


def checks_for(fmt: str, scope: str) -> list[Check]:
    found = [c for c in _CHECKS if c.scope == scope and (c.formats is None or fmt in c.formats)]
    return sorted(found, key=lambda c: STAGES.index(c.stage))
