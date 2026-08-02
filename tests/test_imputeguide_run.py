"""End-to-end tests for the installed ImputeGuide command path."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from imputeguide.runner import run_csv


def _incomplete_cluster_table() -> np.ndarray:
    rng = np.random.default_rng(7)
    blocks = [
        rng.normal(center, 0.35, size=(12, 4))
        for center in (-4.0, 0.0, 4.0)
    ]
    values = np.vstack(blocks)
    missing = rng.random(values.shape) < 0.12
    for row in range(len(values)):
        if missing[row].all():
            missing[row, 0] = False
    for column in range(values.shape[1]):
        if missing[:, column].all():
            missing[0, column] = False
    values[missing] = np.nan
    return values


def test_run_csv_selects_method_and_writes_outputs(tmp_path: Path) -> None:
    source = _incomplete_cluster_table()
    input_path = tmp_path / "input.csv"
    output_dir = tmp_path / "output"
    pd.DataFrame(source, columns=["a", "b", "c", "d"]).to_csv(
        input_path, index=False
    )

    run = run_csv(
        input_path,
        output_dir,
        n_clusters=3,
        anchor="mode",
        historical_ranking=("knni",),
        probe_ranking=(),
        opportunity_score=1.0,
        opportunity_threshold=0.0,
        config_root=Path(__file__).resolve().parents[1],
        seed=19,
    )

    assert run.selection.selected_method in {"mode", "knni"}
    completed = pd.read_csv(output_dir / "completed.csv").to_numpy(dtype=float)
    assert completed.shape == source.shape
    assert np.isfinite(completed).all()
    observed = np.isfinite(source)
    assert np.allclose(completed[observed], source[observed])
    trace = json.loads((output_dir / "selection.json").read_text(encoding="utf-8"))
    assert trace["selected_method"] == run.selection.selected_method
    assert trace["attempted_methods"] == ["mode", "knni"]
