"""
Project A - Step 3: Replay simulator.

Replays any policy through the test window (cutoff+1 .. cutoff+56) for the
California network: suppliers -> 1 DC -> 4 stores, every FOODS item.

Day loop (each simulated day t)
-------------------------------
1. Arrivals     supplier POs due today land at the DC; DC->store transfers
                sent 2 days ago land at the stores.
2. Demand       each store sells min(stock, demand); the rest is a lost sale
                (no backorders: a shopper who finds an empty shelf leaves).
3. Policy       policy(state) -> {"po": units per item (Mondays only),
                                  "transfers": units per item x store}.
4. Limits       transfers are cut, in this order, to (a) the store cap of 21
                days of average sales (on hand + in transit + new transfer),
                (b) what the DC has (shared pro rata between stores), and
                (c) the DC outbound cap of 1.25 x average daily CA FOODS sales
                (all transfers scaled down pro rata). POs on non-Mondays are
                ignored.
5. Lead times   each PO draws a lognormal supplier->DC lead time (mean 5/7/10
                days for FOODS_1/2/3, CV 0.4), rounded to whole days, min 1.
                One draw per department PO (one PO = one shipment). Stress
                scenarios multiply the lead time of POs placed in a window.
6. Costs        holding 25%/yr of price on DC + store + in-transit stock,
                lost sale 35% of price, $50 per department PO, $0.05 per unit
                moved DC->store.

How demand on censored days is handled
---------------------------------------
The test window is censored too, so "actual sales" is not "actual demand".
Demand comes from demand_replay.parquet (demand_recovery.py, run on all days):
  * ok days                      demand = recorded sales (fixed in every run)
  * stockout / disruption days   demand ~ Poisson(recovered rate) - drawn
                                 fresh for every seed, because we do not
                                 know what demand really was on those days
  * not_listed / dormant /closure demand = 0
So a policy is never rewarded for stock it could only have sold on days
Walmart itself was empty, but it is charged for demand that was really there.
The spread across seeds comes from lead-time draws and censored-day demand.

Starting state: every store and the DC hold 7 days of average sales per
item (average over listed days in the last 365 training days). Nothing is
in the pipeline.

How to run
----------
    python -m project_a.simulate --cutoff 1885                  # baseline
    python -m project_a.simulate --cutoff 1885 --policy project_a.policy:policy
"""

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_a.config import load_config  # noqa: E402

STORES = ["CA_1", "CA_2", "CA_3", "CA_4"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday"]


# ---------------------------------------------------------------------------
# Context: everything static about the test window, built once per cutoff
# ---------------------------------------------------------------------------
def to_grid(long: pd.DataFrame, col: str, items, n_days, day0) -> np.ndarray:
    """Long (id, day, col) -> array (n_items, n_stores, n_days)."""
    # parse each distinct id once ("FOODS_1_001_CA_1_evaluation")
    ids = long["id"].astype("category")
    cats = pd.Series(ids.cat.categories.astype(str))
    item_of = pd.Index(items).get_indexer(cats.str.rsplit("_", n=3).str[0])
    store_of = pd.Index(STORES).get_indexer(cats.str.split("_").str[-3:-1].str.join("_"))
    codes = ids.cat.codes.to_numpy()
    ii, ss = item_of[codes], store_of[codes]
    out = np.zeros((len(items), len(STORES), n_days), dtype=np.float32)
    out[ii, ss, long["day"].to_numpy() - day0] = long[col].to_numpy()
    return out


def build_context(cfg: dict, cutoff: int, history: str = "recovered",
                  history_days: int = 91) -> dict:
    """
    Static inputs for a simulation run.

    history = "recovered": the policy plans with recovered demand (default).
    history = "raw":       the policy plans with raw sales, to measure what
                           ignoring censoring costs.
    """
    proc = cfg["paths"]["processed_dir"]
    train = pd.read_parquet(proc / "demand_train.parquet")
    replay = pd.read_parquet(proc / "demand_replay.parquet")
    if int(train["day"].max()) != cutoff:
        raise ValueError(f"demand_train.parquet was built for cutoff "
                         f"{int(train['day'].max())}, not {cutoff}. Re-run "
                         "`python -m project_a.demand_recovery --cutoff ...`")

    item_ids = sorted(train["id"].astype(str).str.rsplit("_", n=3).str[0].unique())
    dept = np.array([i.rsplit("_", 1)[0] for i in item_ids])

    # --- training history (last history_days and last 365 days) ---------------
    lookback = cfg["capacity"]["dc_outbound_lookback_days"]
    first = cutoff - max(lookback, history_days) + 1
    tr = train[train["day"] >= first]
    n_tr = cutoff - first + 1
    listed = to_grid(tr.assign(x=~tr["reason"].isin(["not_listed", "closure",
                                                     "dormant"])),
                     "x", item_ids, n_tr, first) > 0
    sales_tr = to_grid(tr, "sales", item_ids, n_tr, first)
    demand_tr = to_grid(tr, "demand_est", item_ids, n_tr, first)

    yr = slice(n_tr - lookback, n_tr)
    n_listed = listed[:, :, yr].sum(axis=2)
    avg_sales = np.divide(sales_tr[:, :, yr].sum(axis=2), n_listed,
                          out=np.zeros(n_listed.shape), where=n_listed > 0)
    dc_cap = cfg["capacity"]["dc_outbound_multiplier"] * sales_tr[:, :, yr].sum() / lookback
    store_cap = np.maximum(1, np.ceil(cfg["capacity"]["store_max_days_of_sales"] * avg_sales))

    hist = demand_tr if history == "recovered" else sales_tr
    hist = np.where(listed, hist, np.nan)[:, :, -history_days:]

    # --- replay window ----------------------------------------------------------
    t0 = cutoff + 1
    h = int(replay["day"].max()) - cutoff
    demand_fixed = to_grid(replay.assign(x=np.where(replay["censored"], 0,
                                                    replay["demand_est"])),
                           "x", item_ids, h, t0)
    demand_rate = to_grid(replay.assign(x=np.where(replay["censored"],
                                                   replay["demand_est"], 0)),
                          "x", item_ids, h, t0)

    # prices: weekly store price, forward-filled; DC price = mean over stores
    p0 = cutoff - 119                       # 120 days back, for forward-filling
    prices = pd.read_parquet(proc / "ca_foods_long.parquet",
                             columns=["id", "day", "sell_price"],
                             filters=[("day", ">=", p0), ("day", "<=", cutoff + h)])
    p = to_grid(prices.assign(sell_price=prices["sell_price"].fillna(-1)),
                "sell_price", item_ids, cutoff + h - p0 + 1, p0)
    p[p < 0] = np.nan
    p = pd.DataFrame(p.reshape(-1, p.shape[2])).ffill(axis=1).to_numpy().reshape(p.shape)
    p = p[:, :, -h:]
    item_mean = np.nanmean(p, axis=(1, 2), keepdims=True)
    p = np.where(np.isnan(p), np.broadcast_to(item_mean, p.shape), p)
    p = np.nan_to_num(p, nan=0.0)
    dc_price = p.mean(axis=1)

    cal = pd.read_csv(cfg["paths"]["raw_dir"] / "calendar.csv")
    cal["day"] = cal["d"].str.replace("d_", "", regex=False).astype(int)
    weekday = cal.set_index("day").loc[t0:t0 + h - 1, "weekday"].to_numpy()

    lt = cfg["lead_times"]["supplier_to_dc"]
    return {
        "cutoff": cutoff, "first_day": t0, "horizon": h, "history": history,
        "items": item_ids, "stores": STORES, "dept": dept,
        "departments": sorted(set(dept)),
        "weekday": weekday,
        "price": p, "dc_price": dc_price,
        "avg_sales": avg_sales,                 # (items, stores), per day
        "store_cap": store_cap,
        "dc_outbound_cap": float(dc_cap),
        "demand_history": hist,                 # (items, stores, days); NaN = not listed
        "lead_time_mean": np.array([lt[d]["mean"] for d in dept], float),
        "lead_time_cv": np.array([lt[d]["cv"] for d in dept], float),
        "dc_to_store_days": cfg["lead_times"]["dc_to_store_days"],
        "dc_order_weekday": cfg["review"]["dc_order_weekday"],
        "costs": cfg["costs"],
        "target_fill_rate": cfg["service"]["target_fill_rate"],
        "start_days": cfg["starting_stock"]["days_of_avg_sales"],
        "lead_time_sampling": cfg["simulation"].get("lead_time_sampling", "po"),
        # replay demand: hidden from the policy
        "_demand_fixed": demand_fixed,
        "_demand_rate": demand_rate,
        "_cache": {},
    }


# ---------------------------------------------------------------------------
# One simulation run
# ---------------------------------------------------------------------------
def lognormal_days(mean, cv, rng, size=None):
    s2 = np.log(1 + cv ** 2)
    x = rng.lognormal(np.log(mean) - s2 / 2, np.sqrt(s2), size=size)
    return np.maximum(1, np.rint(x)).astype(int)


def ration_rows(q: np.ndarray, avail: np.ndarray) -> np.ndarray:
    """
    Share avail[i] units between the stores in row i in proportion to q,
    in whole units, using largest remainders. (Plain rounding down would turn
    every 1-unit request into 0 the moment there is any shortage.)
    """
    x = q * (avail / np.maximum(q.sum(axis=1), 1))[:, None]
    base = np.floor(x)
    left = (avail - base.sum(axis=1)).astype(int)
    rank = np.argsort(np.argsort(-(x - base), axis=1), axis=1)
    return base + (rank < left[:, None])


def ration_total(q: np.ndarray, total: float) -> np.ndarray:
    """Scale all of q down to `total` whole units, largest remainders first."""
    x = q * (total / q.sum())
    base = np.floor(x)
    left = int(total - base.sum())
    if left > 0:
        top = np.argpartition(-(x - base).ravel(), left)[:left]
        base.ravel()[top] += 1
    return base


def public_context(ctx: dict) -> dict:
    """What the policy may see: everything except the replay demand."""
    return {k: v for k, v in ctx.items() if not k.startswith("_") or k == "_cache"}


def simulate(ctx: dict, policy, seed: int, scenario: dict | None = None,
             record_daily: bool = False) -> dict:
    """
    scenario (optional), e.g. the stress test:
        {"dept": "FOODS_3", "start": 14, "end": 35, "lead_time_multiplier": 2.0,
         "warning_days": 14}
    start/end are 0-based offsets in the test window; POs placed in
    [start, end) get the multiplier. If warning_days is set, the policy sees
    the scenario in state["known_disruptions"] from day start - warning_days.
    """
    rng = np.random.default_rng(seed)
    n_i, n_s = len(ctx["items"]), len(STORES)
    h, l_ds = ctx["horizon"], ctx["dc_to_store_days"]
    c = ctx["costs"]
    dept = ctx["dept"]
    depts = ctx["departments"]

    demand = ctx["_demand_fixed"] + rng.poisson(ctx["_demand_rate"])

    store = np.rint(ctx["start_days"] * ctx["avg_sales"])
    dc = np.rint(ctx["start_days"] * ctx["avg_sales"].sum(axis=1))
    transit = np.zeros((n_i, n_s, h + l_ds + 1))           # arrivals by day index
    po_arrivals = np.zeros((n_i, h + 200))
    open_pos = []          # (placed t, dept, qty per item, arrival t per item)

    tot = dict(holding=0.0, lost_sales=0.0, po=0.0, transfer=0.0,
               demand=0.0, sold=0.0, lost_units=0.0, n_po=0, transfer_units=0.0)
    by_dept = {d: dict(demand=0.0, sold=0.0, cost=0.0) for d in depts}
    daily = []
    hold_rate = c["holding_rate_per_year"] / 365
    pub = public_context(ctx)

    for t in range(h):
        # 1. arrivals
        dc += po_arrivals[:, t]
        store += transit[:, :, t]

        # 2. demand
        d = demand[:, :, t]
        sold = np.minimum(store, d)
        lost = d - sold
        store -= sold
        price = ctx["price"][:, :, t]

        # 3. policy
        known = []
        if scenario and scenario.get("warning_days") is not None and \
                t >= scenario["start"] - scenario["warning_days"]:
            known = [scenario]
        state = {
            "t": t, "day": ctx["first_day"] + t, "weekday": ctx["weekday"][t],
            "is_po_day": ctx["weekday"][t] == ctx["dc_order_weekday"],
            "store_on_hand": store.copy(),
            "store_in_transit": transit[:, :, t + 1:].sum(axis=2),
            "store_pipeline": transit[:, :, t + 1:t + 1 + l_ds].copy(),
            "dc_on_hand": dc.copy(),
            # the policy sees what is on order, not when it will arrive
            "dc_on_order": sum((np.where(a > t, q, 0) for _, _, q, a in open_pos),
                               np.zeros(n_i)),
            "open_pos": [(p, dp, np.where(a > t, q, 0)) for p, dp, q, a in open_pos],
            "sales_today": sold.copy(), "lost_today": lost.copy(),
            "known_disruptions": known,
            "context": pub,
        }
        orders = policy(state) or {}

        # 4. transfers with limits
        q = np.maximum(0, np.rint(np.nan_to_num(orders.get("transfers",
                                                           np.zeros((n_i, n_s))))))
        room = np.maximum(0, ctx["store_cap"] - store - state["store_in_transit"])
        q = np.minimum(q, room)
        short = q.sum(axis=1) > dc
        if short.any():
            q[short] = ration_rows(q[short], np.floor(dc[short]))
        if q.sum() > ctx["dc_outbound_cap"]:
            q = ration_total(q, np.floor(ctx["dc_outbound_cap"]))
        dc -= q.sum(axis=1)
        transit[:, :, t + l_ds] += q

        # 5. purchase orders (Mondays only), one lead-time draw per department
        po_cost = 0.0
        if state["is_po_day"] and orders.get("po") is not None:
            po = np.maximum(0, np.rint(np.nan_to_num(orders["po"])))
            for dp in depts:
                m = (dept == dp) & (po > 0)
                if not m.any():
                    continue
                if ctx["lead_time_sampling"] == "line":
                    lt = lognormal_days(ctx["lead_time_mean"][m], ctx["lead_time_cv"][m], rng)
                else:
                    lt = np.full(m.sum(), lognormal_days(
                        ctx["lead_time_mean"][m][0], ctx["lead_time_cv"][m][0], rng))
                if scenario and scenario["dept"] == dp and \
                        scenario["start"] <= t < scenario["end"]:
                    lt = np.rint(lt * scenario["lead_time_multiplier"]).astype(int)
                qty = np.zeros(n_i)
                qty[m] = po[m]
                arrive = np.full(n_i, -1)
                arrive[m] = t + lt
                np.add.at(po_arrivals, (np.flatnonzero(m), t + lt), po[m])
                open_pos.append((t, dp, qty, arrive))
                po_cost += c["po_cost_per_department"]
                by_dept[dp]["cost"] += c["po_cost_per_department"]
                tot["n_po"] += 1
        # drop POs whose every line has arrived by tomorrow
        open_pos = [po_ for po_ in open_pos if (po_[3] > t + 1).any()]

        # 6. costs
        holding = hold_rate * ((store * price).sum()
                               + (dc * ctx["dc_price"][:, t]).sum()
                               + (transit[:, :, t + 1:].sum(axis=2) * price).sum())
        lost_cost = c["lost_sale_rate"] * (lost * price).sum()
        transfer_cost = c["transfer_cost_per_unit"] * q.sum()
        tot["holding"] += holding
        tot["lost_sales"] += lost_cost
        tot["po"] += po_cost
        tot["transfer"] += transfer_cost
        tot["demand"] += d.sum()
        tot["sold"] += sold.sum()
        tot["lost_units"] += lost.sum()
        tot["transfer_units"] += q.sum()
        for dp in depts:
            m = dept == dp
            by_dept[dp]["demand"] += d[m].sum()
            by_dept[dp]["sold"] += sold[m].sum()
            by_dept[dp]["cost"] += (hold_rate * ((store[m] * price[m]).sum()
                                                 + (dc[m] * ctx["dc_price"][m, t]).sum())
                                    + c["lost_sale_rate"] * (lost[m] * price[m]).sum()
                                    + c["transfer_cost_per_unit"] * q[m].sum())
        if record_daily:
            daily.append({"t": t, "day": ctx["first_day"] + t,
                          "demand": d.sum(), "sold": sold.sum(),
                          "store_stock": store.sum(), "dc_stock": dc.sum(),
                          "transfers": q.sum(), "holding": holding,
                          "lost_cost": lost_cost, "po_cost": po_cost})

    total = tot["holding"] + tot["lost_sales"] + tot["po"] + tot["transfer"]
    out = {"seed": seed, "total_cost": total,
           "fill_rate": tot["sold"] / max(tot["demand"], 1), **tot}
    for dp in depts:
        out[f"fill_rate_{dp}"] = by_dept[dp]["sold"] / max(by_dept[dp]["demand"], 1)
    if record_daily:
        out["daily"] = pd.DataFrame(daily)
    return out


def run_many(ctx, policy, n_seeds, base_seed, scenario=None) -> pd.DataFrame:
    rows = []
    for k in range(n_seeds):
        ctx["_cache"].clear()                      # policies start fresh each run
        r = simulate(ctx, policy, base_seed + k, scenario)
        rows.append(r)
    return pd.DataFrame(rows)


def summarise(runs: pd.DataFrame, name: str) -> dict:
    """Mean and 5-95% range across seeds."""
    out = {"policy": name}
    for col in ["total_cost", "holding", "lost_sales", "po", "transfer", "fill_rate",
                *[c for c in runs.columns if c.startswith("fill_rate_")]]:
        out[col] = runs[col].mean()
        out[f"{col}_p05"] = runs[col].quantile(0.05)
        out[f"{col}_p95"] = runs[col].quantile(0.95)
    return out


def load_policy(spec: str):
    module, fn = spec.split(":")
    return getattr(importlib.import_module(module), fn)


# ---------------------------------------------------------------------------
def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=int, default=cfg["time"]["default_cutoff"])
    parser.add_argument("--policy", default="project_a.policy:base_stock_policy",
                        help="module:function to replay")
    parser.add_argument("--seeds", type=int, default=cfg["simulation"]["n_seeds"])
    args = parser.parse_args()

    out_dir = cfg["paths"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    seed0 = cfg["simulation"]["random_seed"]
    pol = load_policy(args.policy)
    name = args.policy.split(":")[1]

    rows, all_runs = [], []
    # same policy planned on recovered demand vs raw sales = cost of ignoring censoring
    for history in ["recovered", "raw"]:
        ctx = build_context(cfg, args.cutoff, history=history)
        runs = run_many(ctx, pol, args.seeds, seed0)
        label = f"{name} (planned on {history} {'demand' if history == 'recovered' else 'sales'})"
        rows.append(summarise(runs, label))
        all_runs.append(runs.assign(policy=label))
        print(f"{label}: cost {runs.total_cost.mean():,.0f} "
              f"[{runs.total_cost.quantile(.05):,.0f} - {runs.total_cost.quantile(.95):,.0f}], "
              f"fill rate {runs.fill_rate.mean():.3f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / f"simulate_{name}.csv", index=False)
    pd.concat(all_runs).to_csv(out_dir / f"simulate_{name}_runs.csv", index=False)

    base, raw = summary.iloc[0], summary.iloc[1]
    print(f"\nTest window: d_{ctx['first_day']}..d_{ctx['first_day'] + ctx['horizon'] - 1} "
          f"({ctx['horizon']} days), {len(ctx['items'])} items x 4 stores, "
          f"{args.seeds} seeds")
    print(f"DC outbound cap: {ctx['dc_outbound_cap']:,.0f} units/day")
    cols = ["policy", "total_cost", "holding", "lost_sales", "po", "transfer",
            "fill_rate", "fill_rate_FOODS_1", "fill_rate_FOODS_2", "fill_rate_FOODS_3"]
    print(summary[cols].round(3).to_string(index=False))
    print(f"\nPlanning on raw sales instead of recovered demand changes cost by "
          f"{raw.total_cost - base.total_cost:+,.0f} and fill rate by "
          f"{raw.fill_rate - base.fill_rate:+.3f}")


if __name__ == "__main__":
    main()
