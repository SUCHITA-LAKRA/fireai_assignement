"""
Project A - Step 2: Recover true demand from censored sales.

M5 records SALES, not DEMAND. A zero can mean "nobody wanted it" or "it was
not on the shelf". This step labels every item x store x day with a reason
and an estimate of unconstrained demand.

Reason codes (one per item x store x day)
-----------------------------------------
ok                sales = demand
not_listed        before the item's first sale (no price, or priced but not
                  yet on the shelf). Demand = 0, and the day is not used.
closure           whole store closed (Christmas Day). Demand = 0.
stockout          run of zero (or near-zero*) sales too long to be chance for this item
                  (probability < alpha given its sales around the run).
                  Demand = the item's rate around the run x weekday effect.
dormant           same test, but the run is longer than max_stockout_days:
                  item most likely off the shelf for good. Demand = 0.
                  * near-zero = at most 10% of the item's median daily sales,
                    so a 120/day item selling 3/day for a month is caught.
store_disruption  CA_2 lost ~25% of its sales for 8 months (2014-10 to
                  2015-05) while the other stores did not. Demand = sales
                  scaled back up to CA_2's normal share of CA sales.

Two passes, so the --cutoff rule holds
--------------------------------------
1. TRAIN: uses days 1..cutoff only. Writes demand_train.parquet. This is all
   the forecast and the policy are allowed to see.
2. REPLAY: uses all days. Writes demand_replay.parquet for the test window
   (cutoff+1 .. cutoff+56). Only the simulator reads it, to score policies.

How to run
----------
    python -m project_a.demand_recovery --cutoff 1885
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_a.config import load_config  # noqa: E402
from project_a.panel import Panel, load_panel  # noqa: E402

REASONS = ["ok", "not_listed", "closure", "stockout", "dormant", "store_disruption"]
OK, NOT_LISTED, CLOSURE, STOCKOUT, DORMANT, DISRUPTION = range(len(REASONS))
CENSORED = {STOCKOUT, DISRUPTION}           # days where demand > sales


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def prefix(x: np.ndarray) -> np.ndarray:
    """Row-wise prefix sums with a leading 0: sum of x[r, a:b] = P[r,b]-P[r,a]."""
    out = np.zeros((x.shape[0], x.shape[1] + 1), dtype=np.float64)
    np.cumsum(x, axis=1, out=out[:, 1:])
    return out


def find_runs(mask: np.ndarray):
    """All runs of True in each row. Returns (row, start, end) with end exclusive."""
    n, t = mask.shape
    padded = np.zeros((n, t + 1), dtype=np.int8)
    padded[:, :t] = mask
    flat = np.r_[0, padded.ravel(), 0]
    step = np.diff(flat)
    starts = np.flatnonzero(step == 1)
    ends = np.flatnonzero(step == -1)
    row = starts // (t + 1)
    return row, starts - row * (t + 1), ends - row * (t + 1)


def window_sum(P: np.ndarray, row, start, end, w: int):
    """Sum over the w days before `start` plus the w days from `end` on."""
    t = P.shape[1] - 1
    b0 = np.maximum(start - w, 0)
    a1 = np.minimum(end + w, t)
    return (P[row, start] - P[row, b0]) + (P[row, a1] - P[row, end])


def first_sale_day(sales: np.ndarray) -> np.ndarray:
    """Index of the first day with a sale (n_days if never sold)."""
    any_sale = (sales > 0).any(axis=1)
    return np.where(any_sale, (sales > 0).argmax(axis=1), sales.shape[1])


def detect_closures(sales, meta, threshold) -> np.ndarray:
    """(n_stores_rows, n_days) bool: day where a store sold < threshold x its usual."""
    closed = np.zeros(sales.shape, dtype=bool)
    for store in meta["store_id"].unique():
        rows = (meta["store_id"] == store).to_numpy()
        tot = pd.Series(sales[rows].sum(axis=0), dtype=float)
        usual = tot.rolling(28, center=True, min_periods=7).median()
        day_closed = (tot < threshold * usual).to_numpy()
        closed[np.ix_(rows, day_closed)] = True
    return closed


def low_threshold(sales, eligible, frac):
    """
    Per series: a day is "low" when sales <= frac x the series' median sales
    on listed days. For slow sellers (median < 1/frac) this is exactly zero;
    for a fast seller doing 120/day, a day selling 3 is as suspicious as a 0.
    """
    med = np.nanmedian(np.where(eligible, sales, np.nan), axis=1)
    return np.floor(frac * np.nan_to_num(med))


def detect_stockouts(sales, eligible, alpha, w, n_passes=2, low_frac=0.1):
    """
    Flag runs of low-sales days that are too long to be chance.

    For each run of L low days, p0 = share of low days among the valid days
    in the w days before and after it (smoothed: (zeros+1)/(days+2)). If
    p0 ** L < alpha, the run is flagged. Pass 2 recomputes p0 with pass-1
    flags removed from the neighbourhood (a stockout next door would
    otherwise make a stockout look normal).

    Returns the flag matrix and a run table (row, start, end, length, p0, rate).
    """
    zero = eligible & (sales <= low_threshold(sales, eligible, low_frac)[:, None])
    row, start, end = find_runs(zero)
    length = end - start
    flagged = np.zeros(sales.shape, dtype=bool)
    log_alpha = np.log(alpha)

    for _ in range(n_passes):
        valid = eligible & ~flagged
        Pn = prefix(valid)
        Pz = prefix(valid & zero)
        Ps = prefix(np.where(valid, sales, 0))
        n_nb = window_sum(Pn, row, start, end, w)
        z_nb = window_sum(Pz, row, start, end, w)
        s_nb = window_sum(Ps, row, start, end, w)
        p0 = (z_nb + 1.0) / (n_nb + 2.0)
        hit = length * np.log(p0) < log_alpha

        flagged = np.zeros(sales.shape, dtype=bool)
        # paint flagged runs back onto the matrix
        r_hit, s_hit, e_hit = row[hit], start[hit], end[hit]
        if len(r_hit):
            lens = e_hit - s_hit
            rr = np.repeat(r_hit, lens)
            cc = np.repeat(s_hit, lens) + (np.arange(lens.sum()) -
                                           np.repeat(np.cumsum(lens) - lens, lens))
            flagged[rr, cc] = True

    rate = np.divide(s_nb, n_nb, out=np.zeros_like(s_nb), where=n_nb > 0)
    runs = pd.DataFrame({"row": row, "start": start, "end": end,
                         "length": length, "p0": p0, "rate": rate,
                         "n_neighbour_days": n_nb, "flagged": hit})
    return flagged, runs


def weekday_factors(sales, valid, meta, wday) -> np.ndarray:
    """(n_series, 7) multiplicative weekday effect per store x department."""
    f = np.ones((sales.shape[0], 7))
    groups = meta.groupby(["store_id", "dept_id"]).indices
    for _, rows in groups.items():
        s = np.where(valid[rows], sales[rows], 0).sum(axis=0)
        n = valid[rows].sum(axis=0)
        by_day = np.array([s[wday == k].sum() / max(n[wday == k].sum(), 1)
                           for k in range(1, 8)])
        if by_day.mean() > 0:
            f[rows] = by_day / by_day.mean()
    return f


def disruption_factors(sales, meta, dates, disruptions, valid):
    """
    For each store disruption: per department, CA_2's normal share of CA
    sales (56 days before the disruption) / its share during it. Sales on
    disrupted days x this factor = estimated demand.
    """
    mask = np.zeros(sales.shape, dtype=bool)
    factor = np.ones(sales.shape, dtype=np.float32)
    rows_out = []
    for d in disruptions:
        start, end = pd.Timestamp(d["start"]), pd.Timestamp(d["end"])
        during = ((dates >= start) & (dates <= end)).to_numpy()
        before = ((dates >= start - pd.Timedelta(days=56)) & (dates < start)).to_numpy()
        if not during.any() or not before.any():
            continue                      # disruption is outside the data range
        for dept in sorted(meta["dept_id"].unique()):
            in_dept = (meta["dept_id"] == dept).to_numpy()
            store = in_dept & (meta["store_id"] == d["store"]).to_numpy()
            s = np.where(valid, sales, 0)
            share_before = s[store][:, before].sum() / s[in_dept][:, before].sum()
            share_during = s[store][:, during].sum() / s[in_dept][:, during].sum()
            k = share_before / share_during
            mask[np.ix_(store, during)] = True
            factor[np.ix_(store, during)] = k
            rows_out.append({"store": d["store"], "dept_id": dept,
                             "share_before": share_before,
                             "share_during": share_during, "factor": k})
    return mask, factor, pd.DataFrame(rows_out)


# ---------------------------------------------------------------------------
# Main recovery
# ---------------------------------------------------------------------------
def recover(panel: Panel, last_day: int, cfg: dict) -> dict:
    """Run the recovery on days 1..last_day only."""
    rc = cfg["demand_recovery"]
    sales = panel.sales[:, :last_day]
    price = panel.price[:, :last_day]
    cal = panel.cal.iloc[:last_day]
    meta = panel.meta
    t = sales.shape[1]
    days = np.arange(t)

    reason = np.full(sales.shape, OK, dtype=np.int8)

    # 1. not listed: no price, or before the first sale
    first = first_sale_day(sales)
    not_listed = np.isnan(price) | (days[None, :] < first[:, None])
    reason[not_listed] = NOT_LISTED

    # 2. store closures
    closed = detect_closures(sales, meta, rc["closure_threshold"]) & ~not_listed
    reason[closed] = CLOSURE
    eligible = ~not_listed & ~closed

    # 3. stockouts
    flagged, runs = detect_stockouts(sales, eligible, rc["stockout_alpha"],
                                     rc["neighbour_window_days"],
                                     low_frac=rc["low_sales_fraction"])
    runs = runs[runs["flagged"]].copy()
    runs["kind"] = np.where(runs["length"] > rc["max_stockout_days"],
                            "dormant", "stockout")

    # 4. demand estimate
    valid = eligible & ~flagged
    wday = cal["wday"].to_numpy()
    f = weekday_factors(sales, valid, meta, wday)
    demand = np.where(eligible, sales, 0).astype(np.float32)

    for kind, code in [("stockout", STOCKOUT), ("dormant", DORMANT)]:
        r = runs[runs["kind"] == kind]
        if r.empty:
            continue
        lens = (r["end"] - r["start"]).to_numpy()
        rr = np.repeat(r["row"].to_numpy(), lens)
        cc = np.repeat(r["start"].to_numpy(), lens) + (
            np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens))
        reason[rr, cc] = code
        if code == STOCKOUT:
            demand[rr, cc] = np.repeat(r["rate"].to_numpy(), lens) * f[rr, wday[cc] - 1]
        else:
            demand[rr, cc] = 0.0

    # 5. store disruption: scale sales back to the store's normal share
    dis_mask, dis_factor, dis_table = disruption_factors(
        sales, meta, cal["date"], rc["store_disruptions"], valid)
    scale = dis_mask & np.isin(reason, [OK, STOCKOUT])   # days with demand > 0
    demand[scale] *= dis_factor[scale]
    reason[dis_mask & (reason == OK)] = DISRUPTION

    return {"sales": sales, "reason": reason, "demand": demand,
            "runs": runs, "disruption": dis_table, "weekday_factor": f,
            "valid": valid, "eligible": eligible, "last_day": last_day}


def to_long(res: dict, meta: pd.DataFrame, day_from: int, day_to: int) -> pd.DataFrame:
    """Wide result -> long table for days day_from..day_to (1-based, inclusive)."""
    sl = slice(day_from - 1, day_to)
    n = len(meta)
    n_days = day_to - day_from + 1
    out = pd.DataFrame({
        "id": pd.Categorical(np.repeat(meta["id"].to_numpy(), n_days)),
        "day": np.tile(np.arange(day_from, day_to + 1, dtype=np.int16), n),
        "sales": res["sales"][:, sl].ravel(),
        "demand_est": res["demand"][:, sl].ravel(),
        "reason": pd.Categorical.from_codes(res["reason"][:, sl].ravel(),
                                            categories=REASONS),
    })
    out["censored"] = out["reason"].isin([REASONS[c] for c in CENSORED])
    return out


# ---------------------------------------------------------------------------
# Checks: how good is the detector, and how big is the bias?
# ---------------------------------------------------------------------------
def velocity_bucket(rate):
    return pd.cut(rate, [-0.01, 0.5, 2, 5, np.inf],
                  labels=["<0.5/day", "0.5-2/day", "2-5/day", ">5/day"])


def injection_test(res: dict, cfg: dict, rng) -> pd.DataFrame:
    """
    Recall: plant fake stockouts (3, 7, 14 zero days) in clean stretches of
    real series and check how many the detector finds.
    False alarms: simulate stockout-free series (negative binomial with each
    series' own mean and variance) and count runs the detector flags.
    """
    rc = cfg["demand_recovery"]
    sales, valid = res["sales"], res["valid"]
    t = sales.shape[1]
    recent = slice(max(0, t - rc["recent_days"]), t)
    s_rec = np.where(valid[:, recent], sales[:, recent], 0)
    n_rec = valid[:, recent].sum(axis=1)
    mean = np.divide(s_rec.sum(1), n_rec, out=np.zeros(len(n_rec)), where=n_rec > 0)
    var = np.divide(((s_rec - mean[:, None]) ** 2 * valid[:, recent]).sum(1), n_rec,
                    out=np.zeros(len(n_rec)), where=n_rec > 0)
    w = rc["neighbour_window_days"]
    log_alpha = np.log(rc["stockout_alpha"])
    low = valid & (sales <= low_threshold(sales, res["eligible"],
                                          rc["low_sales_fraction"])[:, None])
    Pn, Pz = prefix(valid), prefix(low)

    rows = []
    cand = np.flatnonzero(n_rec > 200)
    for L in [3, 7, 14]:
        r = rng.choice(cand, rc["injection_test_samples"])
        a = rng.integers(max(w, t - rc["recent_days"]), t - w - L, size=len(r))
        ok = (Pn[r, a + L] - Pn[r, a]) == L          # planted days were clean
        r, a = r[ok], a[ok]
        # the planted run merges with natural zeros on each side
        zero = low
        start, end = a.copy(), a + L
        for i in range(len(r)):
            while start[i] > 0 and zero[r[i], start[i] - 1]:
                start[i] -= 1
            while end[i] < t and zero[r[i], end[i]]:
                end[i] += 1
        n_nb = window_sum(Pn, r, start, end, w)
        z_nb = window_sum(Pz, r, start, end, w)
        p0 = (z_nb + 1) / (n_nb + 2)
        found = (end - start) * np.log(p0) < log_alpha
        df = pd.DataFrame({"bucket": velocity_bucket(mean[r]), "found": found})
        for b, g in df.groupby("bucket", observed=True):
            rows.append({"test": f"recall: planted {L}-day stockout",
                         "velocity": str(b), "n": len(g), "value": g["found"].mean()})

    # false alarms on simulated clean series
    n_sim = 365
    sim_rows = cand
    m, v = mean[sim_rows], np.maximum(var[sim_rows], mean[sim_rows] + 1e-6)
    p = np.clip(m / v, 1e-3, 0.999)
    k = m * p / (1 - p)
    sim = rng.negative_binomial(np.maximum(k, 1e-3)[:, None], p[:, None],
                                size=(len(sim_rows), n_sim)).astype(np.int32)
    flags, _ = detect_stockouts(sim, np.ones(sim.shape, bool),
                                rc["stockout_alpha"], w,
                                low_frac=rc["low_sales_fraction"])
    real_rate = (res["reason"][sim_rows, recent] == STOCKOUT).mean()
    df = pd.DataFrame({"bucket": velocity_bucket(m), "fp": flags.mean(axis=1)})
    for b, g in df.groupby("bucket", observed=True):
        rows.append({"test": "false alarm: share of clean days flagged",
                     "velocity": str(b), "n": len(g), "value": g["fp"].mean()})
    rows.append({"test": "false alarm: share of clean days flagged",
                 "velocity": "all", "n": len(sim_rows), "value": flags.mean()})
    rows.append({"test": "real data: share of listed days flagged stockout",
                 "velocity": "all", "n": len(sim_rows), "value": real_rate})
    return pd.DataFrame(rows)


def bias_summary(res: dict, meta: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Raw sales vs recovered demand over the last `recent_days` before cutoff."""
    t = res["sales"].shape[1]
    recent = slice(max(0, t - cfg["demand_recovery"]["recent_days"]), t)
    reason = res["reason"][:, recent]
    listed = (reason != NOT_LISTED) & (reason != CLOSURE)
    rows = []
    for dept in sorted(meta["dept_id"].unique()) + ["ALL FOODS"]:
        m = np.ones(len(meta), bool) if dept == "ALL FOODS" else \
            (meta["dept_id"] == dept).to_numpy()
        raw = np.where(listed[m], res["sales"][m][:, recent], 0).sum()
        dem = np.where(listed[m], res["demand"][m][:, recent], 0).sum()
        rows.append({
            "department": dept,
            "series": int(m.sum()),
            "raw_units": int(raw),
            "recovered_units": round(float(dem)),
            "raw_understates_demand_by": 1 - raw / dem,
            "share_listed_days_stockout": (reason[m] == STOCKOUT).sum() / listed[m].sum(),
            "share_listed_days_dormant": (reason[m] == DORMANT).sum() / listed[m].sum(),
            "share_listed_days_disruption": (reason[m] == DISRUPTION).sum() / listed[m].sum(),
            "series_with_a_stockout": int(((reason[m] == STOCKOUT).any(axis=1)).sum()),
        })
    return pd.DataFrame(rows)


def forecast_bias_check(train: dict, replay: dict, meta, cutoff, test_days):
    """
    How wrong would a model trained on raw sales be? Simplest possible model
    (mean of the last 28 days) fitted once on raw sales and once on recovered
    demand, scored against the replay demand on the test window.
    """
    lo, hi = cutoff, min(cutoff + test_days, replay["sales"].shape[1])
    truth = replay["demand"][:, lo:hi]
    listed = (replay["reason"][:, lo:hi] != NOT_LISTED)
    last = slice(cutoff - 28, cutoff)
    ok = train["eligible"][:, last]
    n_ok = np.maximum(ok.sum(1), 1)
    f_raw = np.where(ok, train["sales"][:, last], 0).sum(1) / n_ok
    f_rec = np.where(ok, train["demand"][:, last], 0).sum(1) / n_ok
    rows = []
    for dept in sorted(meta["dept_id"].unique()) + ["ALL FOODS"]:
        m = np.ones(len(meta), bool) if dept == "ALL FOODS" else \
            (meta["dept_id"] == dept).to_numpy()
        tr = np.where(listed[m], truth[m], 0)
        for name, fc in [("raw sales", f_raw), ("recovered demand", f_rec)]:
            pred = np.where(listed[m], fc[m][:, None], 0)
            rows.append({"department": dept, "trained_on": name,
                         "forecast_units": pred.sum(), "test_demand_units": tr.sum(),
                         "bias": pred.sum() / tr.sum() - 1,
                         "wape": np.abs(pred - tr).sum() / tr.sum()})
    return pd.DataFrame(rows)


def plot(res: dict, panel: Panel, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sales, reason = res["sales"], res["reason"]
    t = sales.shape[1]
    dates = panel.cal["date"].iloc[:t]
    fig, ax = plt.subplots(2, 1, figsize=(11, 7))

    # (a) a fast seller with a clear stockout in the last year
    recent = slice(t - 365, t)
    so = (reason[:, recent] == STOCKOUT).sum(1)
    vel = sales[:, recent].mean(1)
    score = np.where((so >= 5) & (so <= 30), vel, -1)
    i = int(score.argmax())
    ax[0].plot(dates.iloc[recent], sales[i, recent], lw=0.8, color="#444", label="sales")
    ax[0].plot(dates.iloc[recent], res["demand"][i, recent], lw=0.8, color="#d62728",
               ls="--", label="recovered demand")
    flag = reason[i, recent] == STOCKOUT
    ax[0].fill_between(dates.iloc[recent], 0, sales[i, recent].max(), where=flag,
                       color="#d62728", alpha=0.15, label="flagged stockout")
    ax[0].set_title(f"{panel.meta['id'][i]}: zero-sales runs a fast seller cannot "
                    "produce by chance")
    ax[0].legend(loc="upper left", fontsize=8)

    # (b) CA_2 share of CA FOODS sales
    meta = panel.meta
    tot = pd.DataFrame({s: sales[(meta["store_id"] == s).to_numpy()].sum(0)
                        for s in sorted(meta["store_id"].unique())}, index=dates)
    share = (tot["CA_2"] / tot.sum(1)).rolling(28, center=True).mean()
    ax[1].plot(share.index, share.values, color="#1f77b4")
    dis = (reason[(meta["store_id"] == "CA_2").to_numpy()] == DISRUPTION).any(0)
    ax[1].fill_between(dates, share.min(), share.max(), where=dis, color="#ff7f0e",
                       alpha=0.2, label="flagged store disruption")
    ax[1].set_title("CA_2 share of California FOODS sales (28-day average)")
    ax[1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=int, default=cfg["time"]["default_cutoff"])
    args = parser.parse_args()

    out_dir = cfg["paths"]["output_dir"]
    proc = cfg["paths"]["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["simulation"]["random_seed"])

    panel = load_panel(cfg)
    cutoff = args.cutoff
    if not 365 < cutoff < panel.n_days:
        raise ValueError(f"--cutoff must be between 366 and {panel.n_days - 1}")
    test_end = min(cutoff + cfg["time"]["test_days"], panel.n_days)
    print(f"Cutoff d_{cutoff} ({panel.cal['date'].iloc[cutoff - 1].date()}); "
          f"replay window d_{cutoff + 1}..d_{test_end}")

    # Pass 1: training data only
    train = recover(panel, cutoff, cfg)
    to_long(train, panel.meta, 1, cutoff).to_parquet(proc / "demand_train.parquet",
                                                     index=False)
    # Pass 2: all days, kept only for the replay window
    replay = recover(panel, panel.n_days, cfg)
    to_long(replay, panel.meta, cutoff + 1, test_end).to_parquet(
        proc / "demand_replay.parquet", index=False)

    # Reports
    summary = bias_summary(train, panel.meta, cfg)
    summary.to_csv(out_dir / "demand_recovery_summary.csv", index=False)
    runs = train["runs"].assign(id=panel.meta["id"].to_numpy()[train["runs"]["row"]],
                                start_day=lambda d: d["start"] + 1,
                                end_day=lambda d: d["end"])
    runs[["id", "start_day", "end_day", "length", "p0", "rate", "kind"]].to_csv(
        out_dir / "demand_recovery_runs.csv", index=False)
    train["disruption"].to_csv(out_dir / "demand_recovery_disruption.csv", index=False)
    inj = injection_test(train, cfg, rng)
    inj.to_csv(out_dir / "demand_recovery_detector_check.csv", index=False)
    fb = forecast_bias_check(train, replay, panel.meta, cutoff, cfg["time"]["test_days"])
    fb.to_csv(out_dir / "demand_recovery_forecast_bias.csv", index=False)
    plot(train, panel, out_dir / "demand_recovery.png")

    pd.set_option("display.width", 160)
    print("\n--- Raw sales vs recovered demand, last "
          f"{cfg['demand_recovery']['recent_days']} training days ---")
    print(summary.round(3).to_string(index=False))
    print("\n--- Store disruption factors ---")
    print(train["disruption"].round(3).to_string(index=False))
    print("\n--- Detector check (recall on planted stockouts, false alarms) ---")
    print(inj.round(3).to_string(index=False))
    print("\n--- 28-day-mean forecast trained on raw vs recovered, scored on test ---")
    print(fb.round(3).to_string(index=False))
    print(f"\nSaved: {proc / 'demand_train.parquet'}, {proc / 'demand_replay.parquet'}, "
          f"reports in {out_dir}")


if __name__ == "__main__":
    main()
