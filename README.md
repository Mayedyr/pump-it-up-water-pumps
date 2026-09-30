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

Open [notebooks/project.ipynb](notebooks/project.ipynb). It contains the full analysis and experiments from the team's original notebook: data checks, exploratory plots, feature engineering, eight model comparisons, holdout evaluation, a hypothetical inspection-policy scenario, and competition submission generation. Add written interpretation and conclusions alongside the results.

The default settings run every experiment on a smaller sample. For final scores and a submission, set `QUICK_RUN=False` and `RUN_FINAL_EVALUATION=True` in the setup cell, restart the kernel, and run all cells. The full search may take considerable time. Generated result CSVs and the submission are saved under ignored `outputs/`.

## Team workflow

- Work on a branch for each contribution and use pull requests to review changes before merging to `main`.
- Keep the original CSVs unchanged. Do not commit generated predictions, models, virtual environments, or notebook checkpoints.
- Before sharing a notebook change, restart the kernel, run all cells, and check that outputs are readable.
- Record the validation split, accuracy, and reasoning for each model experiment. Keep the competition test set out of validation.

The final course deliverables are a clear presentation (PDF or notebook slides), code, and a prerecorded team presentation video. The analysis should cover problem motivation, exploratory plots, preprocessing, cross-validation and model selection, results, limitations, and likely real-world impact.
