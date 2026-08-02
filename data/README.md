# Datasets

Store input datasets in `data/raw/` and generated tables in `data/processed/`.
Each input table should contain one row per sample and one column per feature.
Missing numerical values should be represented as `NaN`.

```text
data/
├── raw/
│   └── <dataset-name>/
│       └── data.csv
└── processed/
    └── <dataset-name>/
```

Keep clustering labels in a separate file or array. ImputeGuide receives only
the incomplete feature matrix during imputation-method selection.
