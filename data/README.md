# Dataset

These are the original CSV files supplied for the AI7101 project. Do not modify them in place.

| File | Data rows | Purpose | SHA-256 |
|---|---:|---|---|
| `raw/train_values.csv` | 59,400 | Training features | `d8ebe40f49fe749a851c8ea28601115cd92442f7b20025fbc33881595ae75f5d` |
| `raw/train_labels.csv` | 59,400 | Training labels | `ae9b4f893e8e89df3a2187d38cade75c61dfb1d1e156ecd7615fee14fdbe0a24` |
| `raw/test_values.csv` | 14,850 | Test features | `a222110d5606910953607efa5112eafb1d6c30a483c4cbcd0b92c8306125c9b5` |
| `raw/submission_format.csv` | 14,850 | Required submission IDs/order | `387b15f692da6196b1f2610be1aa327a8cbe8dffa15454b25882f41a4077b9f6` |

Expected contracts:

- `id` is unique in each file.
- Training value and label IDs match one-to-one and in the supplied order.
- Test and submission IDs match one-to-one and in the supplied order.
- The target column is `status_group` and contains three classes.
- Train and test feature columns are identical.

The data-loading cell in `notebooks/project.ipynb` validates these contracts.

Source: Taarifa and the Tanzanian Ministry of Water, via the DrivenData Pump It Up competition. The repository does not grant a separate license for the dataset.
