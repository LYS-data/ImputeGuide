# ImputeGuide

ImputeGuide is a budget-constrained imputation-method selection framework for
clustering incomplete tabular data. It combines historical evidence with the
structure of a target dataset to select an imputer and produce one reusable
completed table.

## Method

<p align="center">
  <img src="https://raw.githubusercontent.com/LYS-data/ImputeGuide/main/assets/imputeguide_framework.png" alt="ImputeGuide framework" width="100%">
</p>

ImputeGuide contains two stages:

1. **Historical evidence construction.** Evaluations collected from historical
   datasets are aggregated into a stable default strategy and a searchable case
   library.
2. **Target-specific selection.** Static descriptors and lightweight probes
   describe a new incomplete table. Historical retrieval and probe rankings
   generate candidates, which are compared using paired structural validation
   before the final strategy is selected.

The selected imputer returns a complete table with the same shape as the input
while preserving all observed values. Target labels are not used during method
selection.

## Supported imputers

The method library contains 19 imputers:

Mean, Median, Mode, KNNI, MICE, MissForest, Iterative RF, SoftImpute, GAIN,
MIWAE, HI-VAE, GRAPE, DiffPuter, MCFlow, MissDiff, NOMI, ReMasker, MIRI, and
HyperImpute.

## Datasets

The evaluation collection contains nine tabular datasets: Five Cluster,
MFeat Factors, Digits, Texture, Flame, Iris, Pendigits, Adult, and Dry Bean.
The experiments cover MCAR, MAR, and MNAR missingness at 10%, 20%, and 30%.

Place datasets under `data/raw/` and processed tables under `data/processed/`:

```text
data/
├── raw/<dataset-name>/
└── processed/<dataset-name>/
```

See [data/README.md](data/README.md) for the expected input format.

## Project structure

```text
assets/               method overview image
configs/              method and imputer configurations
data/                 datasets and processed tables
imputers/             adapters for the supported imputers
src/imputeguide/      candidate generation and selection framework
tests/                unit tests
utils/                data, metric, and clustering utilities
```

## Installation

```bash
conda env create -f environment.yml
conda activate imputeguide
```

Install optional deep-learning backends when needed:

```bash
python -m pip install -r requirements/full.txt
```

## Usage

Imputers share a whole-table execution interface. The following example runs
KNNI on an incomplete NumPy matrix:

```python
import numpy as np

from imputeguide import build_imputer, execute_whole_table

X_missing = np.array([
    [1.0, np.nan, 3.0],
    [2.0, 4.0, np.nan],
    [3.0, 5.0, 7.0],
])

result = execute_whole_table(
    "knni",
    X_missing,
    builder=lambda: build_imputer("knni"),
)
X_completed = result.completion
```

The main selection APIs are available from the `imputeguide` package:

- `build_stable_strategy`: constructs the stable strategy from historical runs;
- `merge_candidate_rankings`: combines history- and probe-based candidates;
- `build_perturbation_plan` and `score_plan`: compute structural evidence;
- `select_from_structural_evidence`: returns the selected imputation method.

## License

Project-authored code is released under the [MIT License](LICENSE). Third-party
components retain their original licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
