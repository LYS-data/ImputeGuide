"""ImputeGuide: budget-bounded whole-table imputer selection for clustering."""

from .candidates import CandidateSet, merge_candidate_rankings
from .confirmation import ConfirmationDecision, confirm_challenger
from .execution import MethodAttempt, execute_whole_table, validate_completion
from .history import HistoricalRun, StableStrategy, build_stable_strategy
from .opportunity import (
    HistoricalOpportunity,
    HistoricalRanking,
    ProbeRanking,
    ScenarioMatch,
    block_balanced_similarity,
    rank_historical_opportunities,
    rank_probe_expansion,
)
from .protocol import ProtocolValidation, validate_protocol
from .registry import (
    METHODS,
    build_imputer,
    describe_imputer,
    methods,
)
from .selector import SelectionResult, select_from_structural_evidence
from .sampling import missing_fraction, stratified_validation_rows
from .runner import TargetSelectionRun, run_csv, run_target_selection
from .structural import (
    ComponentMap,
    PerturbationPlan,
    StructuralPerturbation,
    StructuralScore,
    build_perturbation_plan,
    score_perturbation,
    score_plan,
)

__all__ = [
    "CandidateSet",
    "ComponentMap",
    "ConfirmationDecision",
    "HistoricalRun",
    "HistoricalOpportunity",
    "HistoricalRanking",
    "METHODS",
    "PerturbationPlan",
    "ProtocolValidation",
    "ProbeRanking",
    "ScenarioMatch",
    "SelectionResult",
    "MethodAttempt",
    "StableStrategy",
    "StructuralPerturbation",
    "StructuralScore",
    "TargetSelectionRun",
    "build_perturbation_plan",
    "build_imputer",
    "build_stable_strategy",
    "block_balanced_similarity",
    "confirm_challenger",
    "execute_whole_table",
    "describe_imputer",
    "merge_candidate_rankings",
    "missing_fraction",
    "methods",
    "rank_historical_opportunities",
    "rank_probe_expansion",
    "run_csv",
    "run_target_selection",
    "score_perturbation",
    "score_plan",
    "select_from_structural_evidence",
    "stratified_validation_rows",
    "validate_protocol",
    "validate_completion",
]
