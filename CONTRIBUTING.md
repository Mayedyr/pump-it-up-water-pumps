# Contributing

## Branches and pull requests

- Protect `main`; make changes through pull requests.
- Use focused branches such as `eda/missingness`, `feature/waterpoint-age`, or `model/catboost`.
- Keep commits small and use imperative subjects, for example `Add construction-age feature`.
- Request at least one teammate review. Include the validation score and experimental conclusion when model behavior changes.

## Reproducible experiments

- Use the shared stratified folds (or document why a different split is required).
- Fix random seeds and fit every learned transformation inside a scikit-learn pipeline to prevent leakage.
- Do not tune against the competition test set or public leaderboard repeatedly.
- Record the code revision, feature set, parameters, mean score, fold variability, runtime, and conclusion.
- Move stable notebook code into `src/pump_it_up/`; notebooks should tell the analytical story rather than act as the only implementation.

## Data and generated files

- Treat `data/raw/` as read-only.
- Generate intermediate data under `data/processed/` and models under `artifacts/`; both are ignored.
- Generated submissions belong in `submissions/` and are ignored to avoid accidentally treating predictions as source.
- Never commit credentials, LMS exports, personal information, or environment files.

## Quality checks

Run these before opening a pull request:

```bash
ruff check .
pytest
```

