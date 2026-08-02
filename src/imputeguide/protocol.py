"""Cross-file validation for the ImputeGuide method configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class ProtocolValidation:
    errors: tuple[str, ...]

    @property
    def valid_schema(self) -> bool:
        return not self.errors

    @property
    def release_ready(self) -> bool:
        return not self.errors


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing configuration file: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return value


def validate_protocol(root: Path) -> ProtocolValidation:
    """Validate method identities, budgets, and selection parameters."""

    errors: list[str] = []
    try:
        method_config = _load(root / "configs" / "method.yaml")
        imputers = _load(root / "configs" / "imputers.yaml")
        target = _load(root / "configs" / "target.yaml")
    except (OSError, ValueError, yaml.YAMLError) as error:
        return ProtocolValidation((str(error),))

    protocol_version = method_config.get("method", {}).get("configuration_version")
    if protocol_version != "imputeguide-v1":
        errors.append("method.configuration_version must be imputeguide-v1")

    methods = [item.get("id") for item in imputers.get("methods", [])]
    expected_count = int(imputers.get("expected_count", -1))
    if expected_count != 19 or len(methods) != 19 or len(set(methods)) != 19:
        errors.append("imputer registry must contain 19 unique methods")

    probes = tuple(imputers.get("fixed_probe_methods", []))
    target_probes = tuple(target.get("profile", {}).get("fixed_probe_methods", []))
    expected_probes = ("mean", "knni", "mice", "missforest", "gain")
    if probes != expected_probes or target_probes != expected_probes:
        errors.append("fixed probes must match the configured five-method probe set")
    if not set(probes).issubset(methods):
        errors.append("every fixed probe must be present in the method pool")

    candidate = target.get("candidate_generation", {})
    history_quota = int(candidate.get("history_quota", -1))
    probe_quota = int(candidate.get("probe_quota", -1))
    challenger_budget = int(candidate.get("total_challenger_budget", -1))
    attempt_budget = int(candidate.get("full_table_attempt_budget", -1))
    if (history_quota, probe_quota, challenger_budget, attempt_budget) != (2, 2, 4, 5):
        errors.append("candidate quotas and full-table budget must be 2/2/4/5")

    structural = target.get("structural_validation", {})
    weights = structural.get("component_weights", {})
    expected_weights = {
        "silhouette": 0.40,
        "row_resampling_consistency": 0.35,
        "feature_subspace_agreement": 0.25,
    }
    if weights != expected_weights:
        errors.append("structural component weights must be 0.40/0.35/0.25")
    if (
        structural.get("discovery_perturbations") != 3
        or structural.get("confirmation_perturbations") != 5
    ):
        errors.append("discovery/confirmation perturbation counts must be 3/5")

    confirmation = target.get("confirmation", {})
    expected_confirmation = {
        "bootstrap_repeats": 2000,
        "lower_tail_probability": 0.10,
        "switch_margin": 0.005,
        "equality_action": "retain_stable_strategy",
    }
    if confirmation != expected_confirmation:
        errors.append("confirmation parameters do not match the method configuration")

    evaluation = method_config.get("evaluation", {})
    expected_counts = {
        "historical_dataset_count": 40,
        "target_dataset_count": 9,
        "imputer_count": 19,
        "scenarios_per_target": 27,
    }
    for key, expected in expected_counts.items():
        if evaluation.get(key) != expected:
            errors.append(f"evaluation.{key} must equal {expected}")

    return ProtocolValidation(tuple(errors))
