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

Open [notebooks/project.ipynb](notebooks/project.ipynb). It loads the data, checks that feature rows match their labels, and separates the input features (`X`) from the target (`y`). The remaining sections follow the course project rubric. Add the team's analysis and explain why each transformation and model choice makes sense.

## Team workflow

- Work on a branch for each contribution and use pull requests to review changes before merging to `main`.
- Keep the original CSVs unchanged. Do not commit generated predictions, models, virtual environments, or notebook checkpoints.
- Before sharing a notebook change, restart the kernel, run all cells, and check that outputs are readable.
- Record the validation split, accuracy, and reasoning for each model experiment. Keep the competition test set out of validation.

The final course deliverables are a clear presentation (PDF or notebook slides), code, and a prerecorded team presentation video. The analysis should cover problem motivation, exploratory plots, preprocessing, cross-validation and model selection, results, limitations, and likely real-world impact.
