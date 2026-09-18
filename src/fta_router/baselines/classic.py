"""Non-neural routing baselines for Phase-1 comparisons."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Sequence

from fta_router.schema import PRIMARY_ACTIONS


class MajorityBaseline:
    """Always predict the most frequent training label."""

    def __init__(self) -> None:
        self.label: str = "answer_small"

    def fit(self, labels: Sequence[str]) -> "MajorityBaseline":
        counts = Counter(labels)
        self.label = counts.most_common(1)[0][0]
        return self

    def predict(self, queries: Sequence[str]) -> list[str]:
        return [self.label] * len(queries)


class KeywordHeuristicBaseline:
    """Lightweight keyword rules approximating the 4-way space."""

    TOOL_RE = re.compile(
        r"\b(create|open|file|schedule|comment|restart|deploy|check calendar|"
        r"jira|pull request|\bpr\b|github issue|merge|page the|mute|scale)\b",
        re.I,
    )
    RAG_RE = re.compile(
        r"\b(our|yesterday|last week|runbook|sla|incident|status|release notes|"
        r"company|internal|what failed|confluence|wiki|postmortem)\b",
        re.I,
    )
    ESCALATE_RE = re.compile(
        r"\b(tradeoff|trade-off|architect|multi-hop|root cause|plan a|"
        r"compare options|design a|why did|analyse|analyze|recommend|propose a)\b",
        re.I,
    )

    def fit(self, labels: Sequence[str] | None = None) -> "KeywordHeuristicBaseline":
        return self

    def predict_one(self, query: str) -> str:
        if self.TOOL_RE.search(query):
            return "tools"
        if self.ESCALATE_RE.search(query):
            return "escalate_large"
        if self.RAG_RE.search(query):
            return "rag"
        return "answer_small"

    def predict(self, queries: Iterable[str]) -> list[str]:
        return [self.predict_one(q) for q in queries]


class PromptRubricSimulatedBaseline:
    """Deterministic simulation of a *prompted* 4-way router rubric.

    Honesty statement
    -----------------
    This is **not** an LLM API call and **not** a neural zero-shot model.
    It encodes the same decision order documented in ``data/routing/LABELING.md``
    as scored regex/cue weights (tools > escalate > rag > answer_small ties).
    It is a strong *prompt-rubric / rule* baseline for Phase-1 when no LLM
    API key is available. Do not cite it as a large-model prompted router.
    """

    # Weighted cue groups (substring / regex)
    TOOL_CUES = [
        (re.compile(r"\b(create|file|open|schedule|book|invite|assign|merge|close|"
                    r"restart|deploy|scale|mute|page|acknowledge|trigger|set the feature flag|"
                    r"post a message|comment|upload|enable maintenance)\b", re.I), 3.0),
        (re.compile(r"\b(jira|pagerduty|github|calendar|zendesk|linear|datadog|notion|"
                    r"servicenow|slack|zoom)\b", re.I), 2.0),
        (re.compile(r"\b(via the .*api|ops api|deploy api|ci api)\b", re.I), 2.5),
    ]
    ESCALATE_CUES = [
        (re.compile(r"\b(tradeoff|trade-offs|trade-off|compare|recommend|propose|"
                    r"design a|threat model|root-?cause|multi-week|multi-region|"
                    r"migration plan|decision (matrix|tree)|synthesi[sz]e|critique|"
                    r"reconcile|prioritis[ee]|falsif)\b", re.I), 3.0),
        (re.compile(r"\b(plan |analyse|analyze|outline a|draft a|argue |evaluate risks)\b", re.I), 2.0),
        (re.compile(r"\b(without downtime|failure modes|with options|with risks)\b", re.I), 1.5),
    ]
    RAG_CUES = [
        (re.compile(r"\b(our |we |company|internal)\b", re.I), 2.5),
        (re.compile(r"\b(yesterday|last (week|night|monday|tuesday|quarter|month)|last night)\b", re.I), 2.5),
        (re.compile(r"\b(runbook|playbook|sla|slo|postmortem|confluence|wiki|rfc|"
                    r"release notes|on-call|codeowner|retention policy|what failed)\b", re.I), 2.5),
        (re.compile(r"\b(find the|where is the|who owns|who is the|document the)\b", re.I), 1.5),
    ]
    ANSWER_CUES = [
        (re.compile(r"^(what is|what does|what do|define|explain|list |write a|how do (i|you)|"
                    r"convert |hello|hi[,!]|thanks|good morning)", re.I), 1.5),
        (re.compile(r"\b(in general|typically|as a practice|briefly)\b", re.I), 1.0),
    ]
    SAFETY_CUES = re.compile(
        r"\b(phishing|bypass the company|without permission|without authorisation|"
        r"fake invoice|disable security logging|hide fraudulent)\b",
        re.I,
    )

    def fit(self, labels: Sequence[str] | None = None) -> "PromptRubricSimulatedBaseline":
        return self

    def _score(self, query: str, cues: list) -> float:
        return sum(w for rx, w in cues if rx.search(query))

    def predict_one(self, query: str) -> str:
        # Mild safety → answer_small (refusal path), never tools
        if self.SAFETY_CUES.search(query):
            return "answer_small"
        scores = {
            "tools": self._score(query, self.TOOL_CUES),
            "escalate_large": self._score(query, self.ESCALATE_CUES),
            "rag": self._score(query, self.RAG_CUES),
            "answer_small": self._score(query, self.ANSWER_CUES) + 0.1,  # slight prior
        }
        # Decision order on ties: tools > escalate > rag > answer_small (LABELING.md)
        order = ["tools", "escalate_large", "rag", "answer_small"]
        best = max(scores.values())
        for a in order:
            if scores[a] == best and best > 0.5:
                return a
        return "answer_small"

    def predict(self, queries: Iterable[str]) -> list[str]:
        return [self.predict_one(q) for q in queries]


def available_baselines() -> dict[str, type]:
    return {
        "majority": MajorityBaseline,
        "keyword": KeywordHeuristicBaseline,
        "prompt_rubric": PromptRubricSimulatedBaseline,
    }


assert set(PRIMARY_ACTIONS) == {"answer_small", "rag", "tools", "escalate_large"}
