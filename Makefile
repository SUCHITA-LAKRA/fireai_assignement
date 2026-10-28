# One command reproduces everything:   make all
# Re-run Project A at another cutoff:   make project_a CUTOFF=1857
# Re-run Project B at another hold-out: make project_b HOLDOUT=2023-07-03
# Fast smoke test of Project B:         make project_b QUICK=1   (not for results)

PYTHON  ?= python
CUTOFF  ?= 1885
HOLDOUT ?=
QUICK   ?=

.PHONY: all setup project_a project_b clean

all: project_a project_b

setup:
	$(PYTHON) -m pip install -r requirements.txt

# --- Project A (steps are added here as we build them) ---
# Every step after data_prep re-trains on days <= CUTOFF only.
project_a: data/processed/ca_foods_long.parquet
	$(PYTHON) -m project_a.demand_recovery --cutoff $(CUTOFF)
	$(PYTHON) -m project_a.simulate --cutoff $(CUTOFF)

data/processed/ca_foods_long.parquet:
	$(PYTHON) -m project_a.data_prep

# --- Project B (steps are added here as we build them) ---
# Re-run with another hold-out window:  make project_b HOLDOUT=2023-07-03
HOLDOUT_ARG = $(if $(HOLDOUT),--holdout-start $(HOLDOUT),)
QUICK_ARG   = $(if $(QUICK),--quick,)

project_b:
	$(PYTHON) -m project_b.data_prep $(HOLDOUT_ARG)
	$(PYTHON) -m project_b.dag
	$(PYTHON) -m project_b.model_data
	$(PYTHON) -m project_b.fit_model $(QUICK_ARG)
	$(PYTHON) -m project_b.national $(QUICK_ARG)
	$(PYTHON) -m project_b.calibrate $(QUICK_ARG)
	$(PYTHON) -m project_b.report
	$(PYTHON) -m project_b.optimise
	$(PYTHON) -m project_b.next_experiment
	$(PYTHON) -m project_b.memo

clean:
	rm -rf data/processed/*.parquet data/processed/*.pkl outputs/project_a/* outputs/project_b/*
