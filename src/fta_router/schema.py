"""Schema for behavioural routing examples (schema version 1.0)."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping

PRIMARY_ACTIONS = ("answer_small", "rag", "tools", "escalate_large")
REASONING_LEVELS = ("low", "medium", "high")
DOMAINS = (
    "general_knowledge",
    "company_current",
    "tool_action",
    "multi_hop_planning",
    "ambiguous",
    "safety_mild",
    "open_domain_qa",
)

REQUIRED_FIELDS = (
    "id",
    "query",
    "primary_action",
    "needs_rag",
    "needs_tools",
    "reasoning_level",
    "rationale",
    "domain",
)


@dataclass(frozen=True)
class RoutingExample:
    """One labelled behavioural routing decision."""

    id: str
    query: str
    primary_action: str
    needs_rag: bool
    needs_tools: bool
    reasoning_level: str
    rationale: str
    domain: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "RoutingExample":
        return cls(
            id=str(row["id"]),
            query=str(row["query"]),
            primary_action=str(row["primary_action"]),
            needs_rag=bool(row["needs_rag"]),
            needs_tools=bool(row["needs_tools"]),
            reasoning_level=str(row["reasoning_level"]),
            rationale=str(row["rationale"]),
            domain=str(row["domain"]),
        )


def validate_example(row: Mapping[str, Any], *, line_no: int | None = None) -> list[str]:
    """Return a list of validation error strings (empty if valid)."""
    prefix = f"line {line_no}: " if line_no is not None else ""
    errors: list[str] = []

    for field in REQUIRED_FIELDS:
        if field not in row:
            errors.append(f"{prefix}missing required field '{field}'")
    if errors:
        return errors

    if not isinstance(row["id"], str) or not row["id"].strip():
        errors.append(f"{prefix}id must be a non-empty string")
    if not isinstance(row["query"], str) or not row["query"].strip():
        errors.append(f"{prefix}query must be a non-empty string")
    if not isinstance(row["rationale"], str) or not row["rationale"].strip():
        errors.append(f"{prefix}rationale must be a non-empty string")

    action = row["primary_action"]
    if action not in PRIMARY_ACTIONS:
        errors.append(
            f"{prefix}primary_action '{action}' not in {PRIMARY_ACTIONS}"
        )

    if not isinstance(row["needs_rag"], bool):
        errors.append(f"{prefix}needs_rag must be bool, got {type(row['needs_rag']).__name__}")
    if not isinstance(row["needs_tools"], bool):
        errors.append(
            f"{prefix}needs_tools must be bool, got {type(row['needs_tools']).__name__}"
        )

    level = row["reasoning_level"]
    if level not in REASONING_LEVELS:
        errors.append(f"{prefix}reasoning_level '{level}' not in {REASONING_LEVELS}")

    domain = row["domain"]
    if domain not in DOMAINS:
        errors.append(f"{prefix}domain '{domain}' not in {DOMAINS}")

    # Invariants from schema.md
    if action == "rag" and row.get("needs_rag") is not True:
        errors.append(f"{prefix}primary_action=rag requires needs_rag=true")
    if action == "tools" and row.get("needs_tools") is not True:
        errors.append(f"{prefix}primary_action=tools requires needs_tools=true")

    return errors
