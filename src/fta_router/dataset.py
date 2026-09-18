"""Load and summarise behavioural routing JSONL datasets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from fta_router.schema import RoutingExample, validate_example


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def load_examples(path: str | Path) -> list[RoutingExample]:
    return [RoutingExample.from_dict(r) for r in load_jsonl(path)]


def validate_file(path: str | Path) -> list[str]:
    """Validate all rows in a JSONL file; return error messages."""
    path = Path(path)
    errors: list[str] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path.name}:{line_no}: invalid JSON: {exc}")
                continue
            errors.extend(validate_example(row, line_no=line_no))
            rid = row.get("id")
            if isinstance(rid, str):
                if rid in seen_ids:
                    errors.append(f"{path.name}:{line_no}: duplicate id '{rid}'")
                seen_ids.add(rid)
    return errors


def label_distribution(rows: Iterable[dict[str, Any]], field: str = "primary_action") -> dict[str, int]:
    return dict(Counter(r[field] for r in rows))


def summarise(path: str | Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    return {
        "path": str(path),
        "n": len(rows),
        "primary_action": label_distribution(rows, "primary_action"),
        "domain": label_distribution(rows, "domain"),
        "reasoning_level": label_distribution(rows, "reasoning_level"),
    }
