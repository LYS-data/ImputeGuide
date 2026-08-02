"""End-to-end orchestration for target-specific ImputeGuide selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .candidates import CandidateSet, merge_candidate_rankings
from .execution import MethodAttempt, execute_whole_table
from .registry import METHODS, build_imputer
from .selector import SelectionResult, select_from_structural_evidence
from .structural import ComponentMap, build_perturbation_plan, score_plan


@dataclass(frozen=True, slots=True)
class TargetSelectionRun:
    """Results and cached completions from one target-selection run."""

    candidates: CandidateSet
    selection: SelectionResult
    attempts: tuple[MethodAttempt, ...]

    @property
    def selected_completion(self) -> np.ndarray | None:
        for attempt in self.attempts:
            if attempt.method == self.selection.selected_method:
                return attempt.completion
        return None


def _load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return value


def _validate_methods(anchor: str, rankings: Sequence[str]) -> None:
    unknown = [method for method in (anchor, *rankings) if method not in METHODS]
    if unknown:
        raise ValueError(f"unsupported imputation methods: {sorted(set(unknown))}")


def _score_values(scores) -> np.ndarray | None:
    if not scores or any(not item.valid or item.score is None for item in scores):
        return None
    return np.asarray([float(item.score) for item in scores], dtype=float)


def run_target_selection(
    matrix: np.ndarray,
    *,
    n_clusters: int,
    anchor: str,
    historical_ranking: Sequence[str],
    probe_ranking: Sequence[str],
    opportunity_score: float,
    opportunity_threshold: float,
    config_root: Path,
    seed: int = 42,
) -> TargetSelectionRun:
    """Run candidate execution, paired structural scoring, and final selection.

    The rankings and opportunity score are outputs of ImputeGuide's historical
    retrieval and target-probe stages. This function performs the target-side
    whole-table execution and confirmation stage without using target labels.
    """

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("input must be a two-dimensional numeric matrix")
    if values.shape[0] < n_clusters + 2 or n_clusters < 2:
        raise ValueError("n_clusters must be at least 2 and smaller than the row count")
    if not np.isnan(values).any():
        raise ValueError("input matrix does not contain missing values")

    config_root = Path(config_root).resolve()
    target = _load_yaml(config_root / "configs" / "target.yaml")
    _validate_methods(anchor, (*historical_ranking, *probe_ranking))

    generation = target["candidate_generation"]
    candidates = merge_candidate_rankings(
        historical_ranking,
        probe_ranking,
        anchor=anchor,
        historical_quota=int(generation["history_quota"]),
        probe_quota=int(generation["probe_quota"]),
    )
    gate_open = opportunity_score > opportunity_threshold and bool(candidates.merged)
    methods_to_run = (anchor, *candidates.merged) if gate_open else (anchor,)

    attempts = tuple(
        execute_whole_table(
            method,
            values,
            builder=lambda method=method: build_imputer(method, random_state=seed),
        )
        for method in methods_to_run
    )
    completion_success = {
        attempt.method: attempt.status == "success" for attempt in attempts
    }

    discovery_scores: dict[str, np.ndarray] = {}
    confirmation_scores: dict[str, np.ndarray] = {}
    if gate_open and completion_success.get(anchor, False):
        structural = target["structural_validation"]
        seeds = target["seeds"]
        validation_limit = int(structural["validation_sample_rows_max"])
        if len(values) > validation_limit:
            rng = np.random.default_rng(seed)
            validation_rows = np.sort(
                rng.choice(len(values), size=validation_limit, replace=False)
            )
        else:
            validation_rows = np.arange(len(values))
        validation_count = len(validation_rows)
        if n_clusters >= max(3, int(np.ceil(0.8 * validation_count))):
            raise ValueError("validation sample is too small for n_clusters")

        plan = build_perturbation_plan(
            n_rows=validation_count,
            n_features=values.shape[1],
            discovery_perturbations=int(structural["discovery_perturbations"]),
            confirmation_perturbations=int(structural["confirmation_perturbations"]),
            row_resamples=3,
            feature_subsets=2,
            row_fraction=0.8,
            feature_fraction=0.7,
            minimum_row_overlap=2,
            minimum_features=1,
            discovery_seed=seed + int(seeds["discovery_offset"]),
            confirmation_seed=seed + int(seeds["confirmation_offset"]),
        )
        weights_config = structural["component_weights"]
        weights = (
            float(weights_config["silhouette"]),
            float(weights_config["row_resampling_consistency"]),
            float(weights_config["feature_subspace_agreement"]),
        )
        component_maps = (
            ComponentMap(-1.0, 1.0),
            ComponentMap(-1.0, 1.0),
            ComponentMap(-1.0, 1.0),
        )

        def clusterer(array: np.ndarray, random_state: int) -> np.ndarray:
            return KMeans(
                n_clusters=n_clusters,
                n_init=20,
                random_state=random_state,
            ).fit_predict(array)

        def preprocess(array: np.ndarray) -> np.ndarray:
            return StandardScaler().fit_transform(array)

        for attempt in attempts:
            if attempt.completion is None:
                continue
            completion = attempt.completion[validation_rows]
            scoring = {
                "clusterer": clusterer,
                "component_maps": component_maps,
                "weights": weights,
                "preprocessor": preprocess,
                "minimum_row_overlap": 2,
            }
            discovery = _score_values(score_plan(completion, plan.discovery, **scoring))
            confirmation = _score_values(score_plan(
                completion, plan.confirmation, **scoring
            ))
            if discovery is None or confirmation is None:
                completion_success[attempt.method] = False
                continue
            discovery_scores[attempt.method] = discovery
            confirmation_scores[attempt.method] = confirmation

        if anchor not in discovery_scores:
            raise RuntimeError("stable strategy did not produce valid structural evidence")

    confirmation = target["confirmation"]
    selection = select_from_structural_evidence(
        anchor=anchor,
        candidates=candidates.merged,
        completion_success=completion_success,
        discovery_scores=discovery_scores,
        confirmation_scores=confirmation_scores,
        opportunity_score=float(opportunity_score),
        opportunity_threshold=float(opportunity_threshold),
        confirmation_alpha=float(confirmation["lower_tail_probability"]),
        bootstrap_repeats=int(confirmation["bootstrap_repeats"]),
        switch_margin=float(confirmation["switch_margin"]),
        bootstrap_seed=seed + int(target["seeds"]["confirmation_offset"]),
        full_table_budget=int(generation["full_table_attempt_budget"]),
    )
    return TargetSelectionRun(candidates, selection, attempts)


def run_csv(
    input_path: Path,
    output_dir: Path,
    *,
    no_header: bool = False,
    **selection_kwargs,
) -> TargetSelectionRun:
    """Load a CSV table, run selection, and write the completion and trace."""

    input_path = Path(input_path)
    frame = pd.read_csv(input_path, header=None if no_header else "infer")
    numeric = frame.apply(pd.to_numeric, errors="raise")
    run = run_target_selection(numeric.to_numpy(dtype=float), **selection_kwargs)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = {
        "selected_method": run.selection.selected_method,
        "challenger": run.selection.challenger,
        "stop_reason": run.selection.stop_reason,
        "candidate_sources": asdict(run.candidates),
        "attempted_methods": list(run.selection.attempted_methods),
        "successful_methods": list(run.selection.successful_methods),
        "runtime_seconds": {
            attempt.method: attempt.runtime_seconds for attempt in run.attempts
        },
        "discovery_gains": run.selection.discovery_gains,
        "confirmation": (
            asdict(run.selection.confirmation) if run.selection.confirmation else None
        ),
    }
    (output_dir / "selection.json").write_text(
        json.dumps(trace, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    completion = run.selected_completion
    if completion is not None:
        pd.DataFrame(completion, columns=frame.columns).to_csv(
            output_dir / "completed.csv",
            index=False,
            header=not no_header,
        )
    return run
