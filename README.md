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
config/project_b.yaml     all Project B assumptions (columns, hold-out, budget, experiment)
project_a/                code: data_prep, demand recovery, forecast, simulate, policy, stress test
project_b/                code: MMM, calibration, optimisation
data/raw/                 downloaded data (git-ignored)
data/processed/           intermediate files (git-ignored)
outputs/                  charts and tables used in the memos
memos/                    memo_A.pdf, memo_B.pdf
tests/                    small checks (e.g. simulator cost accounting)
DECISIONS.md              major modelling choices, alternatives, why rejected (max 3 pages)
docs/                     DECISIONS_PROJECT_B_DETAIL.md: the full Project B decision log behind the summary
DATA_ISSUES.md            each data problem, how found, how handled
AI_USAGE.md               tools used, and where the AI was wrong
```

## Project A pipeline

| Step | Script | Output | Status |
|---|---|---|---|
| 1. Load & filter CA FOODS | `project_a/data_prep.py` | `data/processed/ca_foods_long.parquet` | done |
| 2. Find data problems / censored demand | `project_a/demand_recovery.py` | corrected demand + charts in `outputs/project_a/` | done |
| 3. Replay simulator | `project_a/simulate.py` | `outputs/project_a/simulate_*.csv` | done |
| 4. Baseline base-stock policy | `project_a/policy.py` (`base_stock_policy`) | baseline cost and fill rate | done |
| 5. Probabilistic forecast + backtests | `project_a/forecast.py` | `project_a/forecasts.parquet` | to do |
| 6. Improved policy | `project_a/policy.py` | — | to do |
| 7. Stress test | `project_a/stress_test.py` | — | to do |
| 8. COO memo | — | `memos/memo_A.pdf` | to do |

## Project B pipeline

| Step | Script | Output | Status |
|---|---|---|---|
| 0. Load, check & prepare data; check the Channel3 experiment | `project_b/data_prep.py` | `data/processed/meridian_geo_weekly.parquet`, `outputs/project_b/step0_*` | done |
| 1. Causal graph (DAG) + tests of its assumptions | `project_b/dag.py` | `outputs/project_b/step1_*` (graph, bias chart, variable roles) | done |
| 2. Model inputs + hold-out split + naive benchmarks | `project_b/model_data.py` | `outputs/project_b/step2_*` | done |
| 3. Hierarchical Bayesian MMM (geo): priors, prior/posterior-predictive checks, convergence | `project_b/fit_model.py` | `project_b/posterior.nc`, `outputs/project_b/step3_*` | done |
| 4. Same model on national totals + comparison | `project_b/national.py` | `outputs/project_b/step4_*` | done |
| 5. Calibrate with the Channel3 experiment (weighted trust) + check it against the data | `project_b/calibrate.py` | `project_b/posterior_calibrated.nc`, `outputs/project_b/step5_*` | done |
| 6. ROI and marginal ROI + 13-week forecast | `project_b/report.py` | `project_b/roi_table.csv`, `project_b/forecast_13w.csv`, `outputs/project_b/step6_*` | done |
| 7. Budget split for 12M / 18M / 24M within ±30%, uncertainty, saturation stress test | `project_b/optimise.py` | `project_b/budget_allocation.csv`, `outputs/project_b/step7_*` | done |
| 8. Next experiment: value of information, test noise from history, power, recommended design | `project_b/next_experiment.py` | `outputs/project_b/step8_*` | done |
| 9. CMO/CFO memo, built from the saved results (no typed-in numbers) | `project_b/memo.py` | `memos/memo_B.pdf` | done |

Assumptions live in `config/project_b.yaml`, including every prior and the reason for it.

Step 3 (the model fit) is the slow step: about 10–30 minutes on a laptop. For a quick check that everything runs, use `make project_b QUICK=1` or `python run_all.py --quick` (a few minutes; numbers are not for results). Hold-out = the last 13 weeks unless `--holdout-start` is given.
