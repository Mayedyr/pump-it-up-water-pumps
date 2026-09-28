# Pump It Up: Data Mining the Water Table

Team repository for the [DrivenData Pump It Up competition](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/) and the AI7101 Machine Learning with Python project.

The task is a three-class classification problem: predict whether each Tanzanian waterpoint is `functional`, `functional needs repair`, or `non functional`. The competition metric is classification accuracy.

## Repository layout

```text
.
├── data/raw/              # Original competition CSVs (do not edit)
├── notebooks/             # Numbered exploratory and reporting notebooks
├── src/pump_it_up/        # Reusable loading and baseline-model code
├── submissions/           # Generated predictions (ignored by Git)
├── tests/                 # Fast data-contract tests
├── CONTRIBUTING.md        # Team workflow and conventions
└── pyproject.toml          # Environment, package, and tool configuration
```

## Quick start

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

# Verify the checked-in dataset and package
python -m pump_it_up.data
pytest

# Train the starter baseline and write a valid submission
python -m pump_it_up.train --cv-folds 5
```

The training command reports stratified cross-validation accuracy, fits the full training set, stores the fitted pipeline under `artifacts/`, and writes `submissions/baseline.csv` in the exact row order of the supplied submission template.

## Suggested team workflow

1. Create a short-lived branch from `main` for each experiment or feature.
2. Start EDA in `notebooks/01_eda.ipynb`; move reusable logic into `src/pump_it_up/`.
3. Record the validation design, features, parameters, score, and conclusion for every experiment.
4. Open a pull request; require at least one teammate review before merging.
5. Never edit files in `data/raw/`. Derived datasets belong in ignored `data/processed/` and must be reproducible from code.

See [CONTRIBUTING.md](CONTRIBUTING.md) for naming, review, and reproducibility conventions.

## Data provenance

The four CSVs in `data/raw/` are the copies supplied for the course project. Their row counts, schema expectations, and SHA-256 checksums are documented in [data/README.md](data/README.md). The data originates from Taarifa and the Tanzanian Ministry of Water and is distributed for the DrivenData competition. Dataset use remains subject to the competition's terms; no separate license is asserted here for the data.

## Project deliverables checklist

- Problem motivation and expected real-world value
- Reproducible load of `X` and `y`
- Motivated missing-value, categorical, target, and feature processing
- Seaborn visual exploration and findings
- Justified cross-validation procedure and metric
- Model comparison and hyperparameter search
- Error analysis, limitations, domain impact, and conclusion
- Clear final notebook/PDF presentation and prerecorded team video

