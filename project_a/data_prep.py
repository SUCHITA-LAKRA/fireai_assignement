"""
Project A - Step 1: Load and filter the M5 data (pandas only).

What this script does
---------------------
1. Reads the 3 M5 files: sales, calendar, prices.
2. Keeps only FOODS items in California stores (CA_1..CA_4).
3. Turns the wide sales table (one column per day) into a long table
   (one row per item x store x day).
4. Adds calendar info (date, weekday, events, SNAP) and the weekly price.
5. Saves one clean file: ca_foods_long.parquet
6. Prints a few basic checks so we know the data loaded correctly.

How to run (from the repo root)
-------------------------------
    python -m project_a.data_prep

Paths come from config/project_a.yaml. Override with --data-dir / --out-dir.
Note: this step keeps ALL days (d_1..d_1941). The --cutoff is applied in
later steps, so the simulator can still replay the days after the cutoff.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Lets the file run with an IDE "Run" button (python project_a/data_prep.py),
# not only as `python -m project_a.data_prep` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_a.config import load_config  # noqa: E402

REQUIRED_FILES = ["sales_train_evaluation.csv", "calendar.csv", "sell_prices.csv"]


def check_input_files(data_dir: Path) -> None:
    """Stop early with a clear message if a data file is missing."""
    missing = [f for f in REQUIRED_FILES if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing in {data_dir}: {missing}\n"
            f"Put the M5 CSV files in that folder (see README)."
        )


def load_sales(data_dir: Path) -> pd.DataFrame:
    """Read the wide sales file and keep only CA FOODS rows."""
    sales = pd.read_csv(data_dir / "sales_train_evaluation.csv")

    is_foods = sales["cat_id"] == "FOODS"
    is_ca = sales["state_id"] == "CA"
    sales = sales.loc[is_foods & is_ca].reset_index(drop=True)

    print(f"Series after filter (item x store): {len(sales):,}")
    return sales


def wide_to_long(sales: pd.DataFrame) -> pd.DataFrame:
    """One column per day  ->  one row per item x store x day."""
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]

    # Text columns -> category BEFORE the melt. Otherwise the melt copies
    # every text value into 11 million rows and peaks at ~6 GB of RAM.
    sales = sales.copy()
    for col in id_cols:
        sales[col] = sales[col].astype("category")
    sales[day_cols] = sales[day_cols].astype("int32")

    long = sales.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )

    # Day number as an integer (d_1 -> 1). Makes cutoffs easy later.
    long["d"] = long["d"].astype("category")
    long["day"] = (long["d"].cat.rename_categories(
        lambda s: int(s.replace("d_", ""))).astype("int16"))
    # int32, not int16: sums across items/stores can exceed 32,767 and
    # pandas keeps the dtype when summing, so int16 would silently overflow.
    long["sales"] = long["sales"].astype("int32")
    return long


def load_calendar(data_dir: Path) -> pd.DataFrame:
    """Keep only the calendar columns we need for California."""
    cal = pd.read_csv(data_dir / "calendar.csv", parse_dates=["date"])
    keep = [
        "d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "snap_CA",
    ]
    return cal[keep]


def load_prices(data_dir: Path) -> pd.DataFrame:
    """Weekly prices for CA FOODS only."""
    prices = pd.read_csv(data_dir / "sell_prices.csv")
    is_ca = prices["store_id"].str.startswith("CA_")
    is_foods = prices["item_id"].str.startswith("FOODS_")
    prices = prices.loc[is_ca & is_foods].copy()
    prices["sell_price"] = prices["sell_price"].astype("float32")
    return prices


def build_long_table(data_dir: Path) -> pd.DataFrame:
    sales = load_sales(data_dir)
    long = wide_to_long(sales)
    cal = load_calendar(data_dir)
    prices = load_prices(data_dir)

    # Add calendar info by day number (int join: much lighter than text "d")
    cal = cal.copy()
    cal["day"] = cal["d"].str.replace("d_", "", regex=False).astype("int16")
    long = long.merge(cal.drop(columns="d"), on="day", how="left")

    # Add weekly price (left join: missing price = item not on sale that week).
    # Join keys as plain strings on the small price table only.
    prices = prices.copy()
    prices["store_id"] = prices["store_id"].astype(
        long["store_id"].dtype)
    prices["item_id"] = pd.Categorical(
        prices["item_id"], categories=long["item_id"].cat.categories)
    prices = prices.dropna(subset=["item_id"])
    long = long.merge(
        prices, on=["store_id", "item_id", "wm_yr_wk"], how="left"
    )

    # Save memory: repeated text columns -> category
    for col in ["id", "item_id", "dept_id", "cat_id", "store_id",
                "state_id", "d", "weekday",
                "event_name_1", "event_type_1",
                "event_name_2", "event_type_2"]:
        long[col] = long[col].astype("category")

    long = long.sort_values(["id", "day"]).reset_index(drop=True)
    return long


def basic_checks(long: pd.DataFrame) -> None:
    """Quick sanity checks. Read these numbers before moving on."""
    print("\n--- Basic checks ---")
    print(f"Rows:                 {len(long):,}")
    print(f"Series:               {long['id'].nunique():,}")
    print(f"Days:                 {long['day'].min()} to {long['day'].max()}")
    print(f"Dates:                {long['date'].min().date()} to "
          f"{long['date'].max().date()}")

    print("\nSeries per department x store:")
    print(long.groupby(["dept_id", "store_id"], observed=True)["id"]
              .nunique().unstack())

    zero_share = (long["sales"] == 0).mean()
    missing_price = long["sell_price"].isna().mean()
    print(f"\nShare of zero-sales days:     {zero_share:.1%}")
    print(f"Share of days with no price:  {missing_price:.1%}")

    # Days where total CA FOODS sales are (almost) zero -> store closures?
    daily_total = long.groupby("date")["sales"].sum()
    print("\nLowest 5 total-sales days (possible closures):")
    print(daily_total.nsmallest(5))


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=cfg["paths"]["raw_dir"])
    parser.add_argument("--out-dir", default=cfg["paths"]["processed_dir"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    check_input_files(data_dir)

    long = build_long_table(data_dir)
    basic_checks(long)

    out_path = out_dir / "ca_foods_long.parquet"
    long.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
