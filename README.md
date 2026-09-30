# Pump It Up: Data Mining the Water Table

AI7101 team project based on the [DrivenData Pump It Up challenge](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/). The goal is to predict each Tanzanian waterpoint's `status_group`: `functional`, `functional needs repair`, or `non functional`.

## Start working

The four original competition CSVs are in `data/raw/`. Their row counts and checksums are in [data/README.md](data/README.md). Keep these files unchanged.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook
```

On macOS, XGBoost may also require the OpenMP runtime: `brew install libomp`.

Open [notebooks/project.ipynb](notebooks/project.ipynb) for the team's analysis and presentation. It includes exploratory plots, feature engineering, eight model comparisons, holdout evaluation, an illustrative inspection-policy scenario, and competition submission generation, with explanations, interpretation, and conclusions.

The default settings use the full development set (`QUICK_RUN=False`) and enable final holdout evaluation (`RUN_FINAL_EVALUATION=True`). Set `QUICK_RUN=True` for a faster comparison on a smaller sample. Restart the kernel and run all cells to regenerate results. The full search may take considerable time. Generated result CSVs and the submission are saved under ignored `outputs/`.

Hyperparameters and the winning model are selected by cross-validation macro-F1, with accuracy used to break ties between models. Fixed inverse-frequency (balanced) class weighting is used for logistic regression, the tree ensembles, decision tree, and multiclass XGBoost; weights are not searched. KNN and the majority baseline remain unchanged. XGBoost weights are computed separately inside each training fit. The notebook has not yet been rerun with this selection metric or fixed weighting rule; saved outputs and static interpretations still reflect the previous accuracy-based run.

The results discussion and conclusion are static markdown cells based on the saved full-run results. Update these cells after rerunning experiments if the scores change. The notebook includes feature/model rationale, EDA interpretation, limitations, references, and an illustrative inspection-cost sensitivity analysis.

## Team workflow

- Work on a branch for each contribution and use pull requests to review changes before merging to `main`.
- Keep the original CSVs unchanged. Do not commit generated predictions, models, virtual environments, or notebook checkpoints.
- Before sharing a notebook change, restart the kernel, run all cells, and check that outputs are readable.
- Record the validation split, accuracy, and reasoning for each model experiment. Keep the competition test set out of validation.

The final course deliverables are a clear presentation (PDF or notebook slides), code, and a prerecorded team presentation video. The analysis should cover problem motivation, exploratory plots, preprocessing, cross-validation and model selection, results, limitations, and likely real-world impact.
