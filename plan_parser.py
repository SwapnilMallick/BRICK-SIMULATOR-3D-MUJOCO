from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union


@dataclass(frozen=True)
class PickAction:
    brick_id: str


@dataclass(frozen=True)
class OrientAction:
    a: float
    b: float
    c: float


@dataclass(frozen=True)
class PlaceAction:
    x: float
    y: float
    z: float


PlanAction = Union[PickAction, OrientAction, PlaceAction]

_PICK_RE = re.compile(r'^pick\(\s*["\']([^"\']+)["\']\s*\)$', re.IGNORECASE)
_ORIENT_RE = re.compile(r'^orient\(\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*\)$', re.IGNORECASE)
_PLACE_RE = re.compile(r'^place\(\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*\)$', re.IGNORECASE)
_FLOAT = r'[+-]?\d*\.?\d+'
_MOVE_BRICK_RE = re.compile(
    rf'^moveBrick\(\s*["\']([^"\']+)["\']\s*'
    rf',\s*({_FLOAT})\s*,\s*({_FLOAT})\s*,\s*({_FLOAT})\s*'
    rf',\s*({_FLOAT})\s*,\s*({_FLOAT})\s*,\s*({_FLOAT})\s*\)$',
    re.IGNORECASE,
)


def parse_plan(path: Path) -> List[PlanAction]:
    actions: List[PlanAction] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        m = _MOVE_BRICK_RE.match(line)
        if m:
            actions.append(PickAction(brick_id=m.group(1)))
            actions.append(OrientAction(a=float(m.group(2)), b=float(m.group(3)), c=float(m.group(4))))
            actions.append(PlaceAction(x=float(m.group(5)), y=float(m.group(6)), z=float(m.group(7))))
            continue

        m = _PICK_RE.match(line)
        if m:
            actions.append(PickAction(brick_id=m.group(1)))
            continue

        m = _ORIENT_RE.match(line)
        if m:
            actions.append(OrientAction(a=float(m.group(1)), b=float(m.group(2)), c=float(m.group(3))))
            continue

        m = _PLACE_RE.match(line)
        if m:
            actions.append(PlaceAction(x=float(m.group(1)), y=float(m.group(2)), z=float(m.group(3))))
            continue

        raise ValueError(f"Unrecognised instruction on line {line_no}: {raw!r}")

    return actions
