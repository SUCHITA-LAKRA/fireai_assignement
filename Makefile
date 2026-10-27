# One command reproduces everything:   make all
# Re-run Project A at another cutoff:   make project_a CUTOFF=1857

PYTHON  ?= python
CUTOFF  ?= 1885
HOLDOUT ?=

.PHONY: all setup project_a project_b clean

all: project_a project_b

setup:
	$(PYTHON) -m pip install -r requirements.txt

# --- Project A (steps are added here as we build them) ---
project_a: data/processed/ca_foods_long.parquet

data/processed/ca_foods_long.parquet:
	$(PYTHON) -m project_a.data_prep

# --- Project B (filled in later) ---
project_b:
	@echo "Project B not built yet"

clean:
	rm -rf data/processed/*.parquet outputs/project_a/* outputs/project_b/*
