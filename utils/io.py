"""JSON serialization helpers for standardized local result persistence."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def make_json_serializable(obj: Any) -> Any:
    """Recursively convert scientific Python objects into JSON-safe values."""
    if isinstance(obj, dict):
        return {str(key): make_json_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    if isinstance(obj, tuple):
        return [make_json_serializable(item) for item in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def standardize_name_token(value: str | None, *, default: str) -> str:
    """Convert arbitrary labels into stable lowercase filename tokens."""
    raw = (value or default).strip().lower()
    token = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    return token or default


def build_standard_json_filename(
    *,
    result_type: str,
    dataset_name: str | None = None,
    scenario_name: str | None = None,
    methods: list[str] | None = None,
    run_tag: str | None = None,
    timestamp: str | None = None,
) -> str:
    """Build a standardized JSON filename for experiment outputs."""
    type_token = standardize_name_token(result_type, default="result")
    dataset_token = standardize_name_token(dataset_name, default="dataset")
    scenario_token = standardize_name_token(scenario_name, default="default")
    methods_token = standardize_name_token("-".join(methods or ["mixed"]), default="mixed")
    tag_token = standardize_name_token(run_tag, default="run")
    time_token = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{type_token}__{dataset_token}__{scenario_token}__{methods_token}__{tag_token}__{time_token}.json"


def resolve_json_output_path(
    output_path: str | Path,
    *,
    result_type: str,
    dataset_name: str | None = None,
    scenario_name: str | None = None,
    methods: list[str] | None = None,
    run_tag: str | None = None,
    timestamp: str | None = None,
) -> Path:
    """Resolve a final JSON path from either a file path or a target directory."""
    path = Path(output_path)
    if path.suffix.lower() == ".json":
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    path.mkdir(parents=True, exist_ok=True)
    filename = build_standard_json_filename(
        result_type=result_type,
        dataset_name=dataset_name,
        scenario_name=scenario_name,
        methods=methods,
        run_tag=run_tag,
        timestamp=timestamp,
    )
    return path / filename


def save_json(data: dict[str, Any], output_path: str | Path) -> Path:
    """Save a standardized JSON file locally using UTF-8 and stable formatting."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = make_json_serializable(data)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(serializable, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
    return path
