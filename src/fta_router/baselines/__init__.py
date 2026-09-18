"""Non-neural routing baselines for Phase-1 comparisons."""

from fta_router.baselines.classic import (
    KeywordHeuristicBaseline,
    MajorityBaseline,
    PromptRubricSimulatedBaseline,
    available_baselines,
)

__all__ = [
    "MajorityBaseline",
    "KeywordHeuristicBaseline",
    "PromptRubricSimulatedBaseline",
    "available_baselines",
]
