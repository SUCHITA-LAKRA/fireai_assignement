# fireai_assignement — Senior Data Scientist Take-Home

Two projects in one repo:

- **Project A** — Inventory policy for Walmart California FOODS under censored demand (M5 data)
- **Project B** — Geo-level marketing mix model with experiment calibration (Google Meridian sample data)

## Quick start

Requires **Python 3.12 or 3.13** (needed by Project B's Meridian/JAX; Project A alone works on 3.10+).
Both projects share one environment — all packages were installed and tested together.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

make all                           # or: python run_all.py
```

Re-run at a different cutoff / hold-out window:

```bash
make project_a CUTOFF=1857         # or: python run_all.py --cutoff 1857
python run_all.py --holdout-start 2023-07-03
```

Target: full run in under 2 hours on a laptop.

## Data (not committed — download yourself)

| Project | Files | Put them in |
|---|---|---|
| A | `sales_train_evaluation.csv`, `sell_prices.csv`, `calendar.csv` from Kaggle "M5 Forecasting – Accuracy" | `data/raw/m5/` |
| B | `geo_all_channels.csv` from the google/meridian GitHub repo, folder `meridian/data/simulated_data/csv` | `data/raw/meridian/` |

## Repo layout

```
config/project_a.yaml     all Project A assumptions (costs, lead times, capacity) — change here only
project_a/                code: data_prep, demand recovery, forecast, simulate, policy, stress test
project_b/                code: MMM, calibration, optimisation
data/raw/                 downloaded data (git-ignored)
data/processed/           intermediate files (git-ignored)
outputs/                  charts and tables used in the memos
memos/                    memo_A.pdf, memo_B.pdf
tests/                    small checks (e.g. simulator cost accounting)
DECISIONS.md              major modelling choices, alternatives, why rejected
DATA_ISSUES.md            each data problem, how found, how handled
AI_USAGE.md               tools used, and one place the AI was wrong
```

## Project A pipeline

| Step | Script | Output | Status |
|---|---|---|---|
| 1. Load & filter CA FOODS | `project_a/data_prep.py` | `data/processed/ca_foods_long.parquet` | done |
| 2. Find data problems / censored demand | `project_a/demand_recovery.py` | recovered demand | to do |
| 3. Replay simulator | `project_a/simulate.py` | — | to do |
| 4. Baseline base-stock policy | `project_a/policy.py` | — | to do |
| 5. Probabilistic forecast + backtests | `project_a/forecast.py` | `project_a/forecasts.parquet` | to do |
| 6. Improved policy | `project_a/policy.py` | — | to do |
| 7. Stress test | `project_a/stress_test.py` | — | to do |
| 8. COO memo | — | `memos/memo_A.pdf` | to do |

## Project B pipeline

To be added.
