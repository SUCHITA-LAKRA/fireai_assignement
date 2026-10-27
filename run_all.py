"""
One-command runner (same as `make all`, for machines without make, e.g. Windows).

    python run_all.py                  # default cutoff 1885
    python run_all.py --cutoff 1857    # re-run Project A at another cutoff
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
    args = parser.parse_args()

    # Project A — steps are added here as we build them
    run("project_a.data_prep")

    # Project B — added later


if __name__ == "__main__":
    main()
