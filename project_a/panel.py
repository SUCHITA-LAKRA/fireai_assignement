"""
Load the CA FOODS long table as wide matrices (one row per item x store,
one column per day). Every later step works on these matrices: they are
~50x faster than group-bys on 11 million rows.

    panel = load_panel(cfg)
    panel.sales[i, d-1]   units sold by series i on day d   (int32)
    panel.price[i, d-1]   selling price that week (NaN = not listed)
    panel.meta            one row per series: id, item_id, dept_id, store_id
    panel.cal             one row per day: day, date, wday (1=Sat .. 7=Fri), ...
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Panel:
    sales: np.ndarray        # (n_series, n_days) int32
    price: np.ndarray        # (n_series, n_days) float32, NaN = not listed
    meta: pd.DataFrame       # n_series rows
    cal: pd.DataFrame        # n_days rows, day = 1..n_days

    @property
    def n_days(self) -> int:
        return self.sales.shape[1]


def load_panel(cfg: dict) -> Panel:
    path = cfg["paths"]["processed_dir"] / "ca_foods_long.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python -m project_a.data_prep` first")
    cols = ["id", "item_id", "dept_id", "store_id", "day", "sales", "sell_price"]
    long = pd.read_parquet(path, columns=cols)
    long = long.sort_values(["id", "day"], kind="stable")

    ids = long["id"].cat.remove_unused_categories().cat.categories
    n, n_days = len(ids), int(long["day"].max())
    if len(long) != n * n_days:
        raise ValueError("Long table is not a full item x day grid")

    sales = long["sales"].to_numpy(np.int32).reshape(n, n_days)
    price = long["sell_price"].to_numpy(np.float32).reshape(n, n_days)
    meta = (long.loc[long["day"] == 1, ["id", "item_id", "dept_id", "store_id"]]
            .reset_index(drop=True))
    for c in meta.columns:
        meta[c] = meta[c].astype(str)
    del long

    cal_cols = ["d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
                "event_name_1", "event_type_1", "snap_CA"]
    cal = pd.read_csv(cfg["paths"]["raw_dir"] / "calendar.csv",
                      parse_dates=["date"], usecols=cal_cols)
    cal["day"] = cal["d"].str.replace("d_", "", regex=False).astype(int)
    cal = cal[cal["day"] <= n_days].reset_index(drop=True)
    return Panel(sales=sales, price=price, meta=meta, cal=cal)
