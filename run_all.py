"""
One-command runner (same as `make all`, for machines without make, e.g. Windows).

    python run_all.py                  # default cutoff 1885
    python run_all.py --cutoff 1857    # re-run Project A at another cutoff
    python run_all.py --holdout-start 2023-07-03   # Project B hold-out window
    python run_all.py --quick          # fast smoke test of Project B (not for results)
"""

import argparse
import subprocess
import sys


def run(module: str, *args: str) -> None:
    cmd = [sys.executable, "-m", module, *args]
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=int, default=1885)
    parser.add_argument("--holdout-start", default=None)
    parser.add_argument("--quick", action="store_true",
                        help="few MCMC draws: tests the pipeline, not for results")
    args = parser.parse_args()

    # Project A — steps are added here as we build them.
    # data_prep keeps all days; every later step uses days <= cutoff only.
    run("project_a.data_prep")
    cutoff = ["--cutoff", str(args.cutoff)]
    run("project_a.demand_recovery", *cutoff)
    run("project_a.simulate", *cutoff)

    # Project B — steps are added here as we build them
    # holdout = ["--holdout-start", args.holdout_start] if args.holdout_start else []
    # run("project_b.data_prep", *holdout)
    # run("project_b.dag")
    # run("project_b.model_data")
    # quick = ["--quick"] if args.quick else []
    # run("project_b.fit_model", *quick)
    # run("project_b.national", *quick)
    # run("project_b.calibrate", *quick)
    # run("project_b.report")
    # run("project_b.optimise")
    # run("project_b.next_experiment")
    # run("project_b.memo")


if __name__ == "__main__":
    main()
