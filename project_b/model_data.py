"""
Project B - Step 2: Build the model's input data and the hold-out split.

What this script does
---------------------
1. Reads the clean data (step 0) and the variable-role table (step 1), so the
   model inputs follow the causal graph exactly: impressions as media, spend
   only for ROI, organic, promo, two controls, population scaling.
2. Builds Meridian's input object twice:
     - geo level: 40 regions x 156 weeks  (the main model, step 3)
     - national:  the same data added up to one series (step 4)
3. Turns the hold-out flag from step 0 into Meridian's hold-out spec and
   checks it: Meridian hides only REVENUE in those weeks; ad spend stays in,
   so ads shown just before the hold-out still carry over into it (adstock).
4. Sanity checks: national totals equal the sum of regions, enough training
   weeks before the hold-out, overlap with the Channel3 test window.
5. Scores three simple "naive" forecasts on the hold-out weeks. The model
   must beat these in step 6, or it is not worth its complexity.

The functions here are imported by steps 3-7 (one place builds the data).

How to run (from the repo root, after steps 0 and 1)
----------------------------------------------------
    python -m project_b.model_data

The hold-out window comes from step 0 (`is_holdout`). To move it, re-run
step 0 with --holdout-start, then this step.

Outputs
-------
    outputs/project_b/step2_split_summary.csv     train / hold-out weeks and checks
    outputs/project_b/step2_model_inputs.csv      every column -> Meridian role
    outputs/project_b/step2_naive_benchmarks.csv  accuracy of naive forecasts on the hold-out
    outputs/project_b/step2_holdout_benchmarks.png
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # quieter TensorFlow start-up

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_b.config import load_config  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, INK, INK_2 = "#8c8b86", "#0b0b0b", "#52514e"

GEO, TIME = "geo", "time"
MAX_LAG = 8  # Meridian's default adstock window (weeks); used for a sanity check


# ---------------------------------------------------------------------------
# 1. Inputs from steps 0 and 1
# ---------------------------------------------------------------------------

def load_clean(cfg: dict) -> pd.DataFrame:
    path = Path(cfg["paths"]["processed_dir"]) / "meridian_geo_weekly.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run step 0 first")
    return pd.read_parquet(path)


def load_roles(cfg: dict) -> dict:
    """Read step 1's variable-role table and group columns by model role."""
    path = Path(cfg["paths"]["output_dir"]) / "step1_variable_roles.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run step 1 first")
    roles = pd.read_csv(path)

    def cols(role_text: str) -> list[str]:
        return roles.loc[roles["causal_role"] == role_text, "variable"].tolist()

    out = {
        "media": cols("treatment (paid media)"),
        "media_spend": cols("cost of the treatment"),
        "organic_media": cols("treatment (unpaid media)"),
        "non_media_treatments": cols("treatment (non-media)"),
        "controls": cols("confounder"),
        "kpi": cols("outcome")[0],
        "revenue_per_kpi": cols("outcome conversion")[0],
        "population": cols("scale")[0],
    }
    # Channel name = column name without the suffix (Channel0_impression -> Channel0)
    out["channels"] = [m.replace("_impression", "") for m in out["media"]]
    for k in ["media", "media_spend"]:
        if len(out[k]) != len(out["channels"]):
            raise ValueError(f"step 1 roles: {k} does not match the channel list")
    return out


def holdout_window(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last hold-out week, from step 0's flag."""
    weeks = df.loc[df["is_holdout"], TIME].drop_duplicates().sort_values()
    if weeks.empty:
        raise ValueError("No hold-out weeks flagged - re-run step 0")
    return weeks.iloc[0], weeks.iloc[-1]


# ---------------------------------------------------------------------------
# 2. National data: add the regions up
# ---------------------------------------------------------------------------

def national_frame(df: pd.DataFrame, roles: dict) -> pd.DataFrame:
    """One row per week. Volumes are summed; indices are population-weighted.

    Spend, impressions, organic and conversions are volumes -> sum.
    Controls and promo are intensities / indices -> population-weighted mean
    (a big region's sentiment counts for more people).
    Revenue per conversion -> total revenue / total conversions.
    """
    pop = roles["population"]
    volumes = roles["media"] + roles["media_spend"] + roles["organic_media"] + [roles["kpi"]]
    indices = roles["controls"] + roles["non_media_treatments"]
    d = df.copy()
    d["_rev"] = d[roles["kpi"]] * d[roles["revenue_per_kpi"]]
    for c in indices:
        d[f"_w_{c}"] = d[c] * d[pop]
    g = d.groupby(TIME)
    nat = g[volumes + ["_rev", pop]].sum()
    for c in indices:
        nat[c] = g[f"_w_{c}"].sum() / nat[pop]
    nat[roles["revenue_per_kpi"]] = nat["_rev"] / nat[roles["kpi"]]
    nat = nat.drop(columns=["_rev"]).reset_index()
    nat["is_holdout"] = g["is_holdout"].first().to_numpy()
    return nat


# ---------------------------------------------------------------------------
# 3. Meridian objects
# ---------------------------------------------------------------------------

def build_input_data(df: pd.DataFrame, roles: dict, level: str = "geo"):
    """Meridian InputData for the geo or national model."""
    from meridian.data import load

    d = df.copy()
    d[TIME] = pd.to_datetime(d[TIME]).dt.strftime("%Y-%m-%d")  # Meridian wants yyyy-mm-dd
    ch = roles["channels"]
    kwargs = dict(
        time=TIME,
        kpi=roles["kpi"],
        revenue_per_kpi=roles["revenue_per_kpi"],
        controls=roles["controls"],
        media=roles["media"],
        media_spend=roles["media_spend"],
        non_media_treatments=roles["non_media_treatments"],
        organic_media=roles["organic_media"],
    )
    if level == "geo":
        kwargs.update(geo=GEO, population=roles["population"])
    elif level != "national":
        raise ValueError("level must be 'geo' or 'national'")

    loader = load.DataFrameDataLoader(
        df=d,
        coord_to_columns=load.CoordToColumns(**kwargs),
        # Our KPI is conversions; revenue_per_kpi turns it into money, so
        # every ROI comes out in currency. (Meridian's own docstring example
        # spells this 'non-revenue'; the code requires 'non_revenue'.)
        kpi_type="non_revenue",
        media_to_channel=dict(zip(roles["media"], ch)),
        media_spend_to_channel=dict(zip(roles["media_spend"], ch)),
    )
    return loader.load()


def holdout_spec(start: pd.Timestamp, end: pd.Timestamp):
    """Meridian hold-out: the same date range for every region (inclusive)."""
    from meridian.model import spec

    return spec.HoldoutSpec(spec=[spec.DateRange(start.strftime("%Y-%m-%d"),
                                                 end.strftime("%Y-%m-%d"))])


# ---------------------------------------------------------------------------
# 4. Checks
# ---------------------------------------------------------------------------

def split_checks(df: pd.DataFrame, nat: pd.DataFrame, roles: dict, cfg: dict,
                 start: pd.Timestamp, end: pd.Timestamp, geo_data, nat_data) -> pd.DataFrame:
    from meridian.model import model, spec

    rows = []

    def add(check, value, ok, note=""):
        rows.append({"check": check, "value": value, "ok": ok, "note": note})

    weeks = np.sort(df[TIME].unique())
    n_hold = int(df.loc[df["is_holdout"], TIME].nunique())
    n_before = int((weeks < np.datetime64(start)).sum())
    n_after = int((weeks > np.datetime64(end)).sum())
    add("hold-out window", f"{start.date()} to {end.date()}", True)
    add("hold-out weeks", n_hold, n_hold == cfg["holdout"]["weeks"])
    add("training weeks (before / after hold-out)", f"{n_before} / {n_after}", n_before > MAX_LAG,
        f"need > {MAX_LAG} weeks before the hold-out so adstock has history")

    # Meridian resolves the same mask we flagged in step 0
    for level, data, expected in [("geo", geo_data, n_hold * df[GEO].nunique()),
                                  ("national", nat_data, n_hold)]:
        m = model.Meridian(input_data=data, model_spec=spec.ModelSpec(
            holdout=holdout_spec(start, end)))
        got = int(np.asarray(m.holdout_id).sum())
        add(f"Meridian hold-out mask ({level})", f"{got} of {np.asarray(m.holdout_id).size} cells",
            got == expected, f"expected {expected}")

    add("geo input shape (regions, weeks, channels)", str(tuple(geo_data.media.shape)),
        tuple(geo_data.media.shape) == (df[GEO].nunique(), len(weeks), len(roles["channels"])))
    add("national input shape (weeks, channels)", str(tuple(nat_data.media.shape)),
        nat_data.media.shape[-2:] == (len(weeks), len(roles["channels"])))

    # National = sum of regions
    for col in [roles["kpi"]] + roles["media_spend"][:1]:
        diff = abs(nat[col].sum() - df[col].sum()) / df[col].sum()
        add(f"national total = sum of regions ({col})", f"{diff:.1e} relative gap", diff < 1e-9)
    rev_geo = (df[roles["kpi"]] * df[roles["revenue_per_kpi"]]).sum()
    rev_nat = (nat[roles["kpi"]] * nat[roles["revenue_per_kpi"]]).sum()
    add("national revenue = sum of regions", f"{abs(rev_nat - rev_geo) / rev_geo:.1e} relative gap",
        abs(rev_nat - rev_geo) / rev_geo < 1e-9)

    # Overlap with the Channel3 experiment window (matters for step 5)
    e = cfg["experiment"]
    e_start = pd.Timestamp(e["start"])
    e_end = e_start + pd.Timedelta(weeks=e["weeks"] - 1)
    overlap = max(0, (min(end, e_end) - max(start, e_start)).days // 7 + 1)
    add("hold-out overlaps the Channel3 test window", f"{overlap} weeks", True,
        "if > 0, step 5 must not judge the test using hidden weeks" if overlap else "")
    return pd.DataFrame(rows)


def model_inputs_table(roles: dict) -> pd.DataFrame:
    rows = []
    for m, s, c in zip(roles["media"], roles["media_spend"], roles["channels"]):
        rows.append([c, m, "media (impressions; adstock + saturation)"])
        rows.append([c, s, "media_spend (only to compute ROI)"])
    for o in roles["organic_media"]:
        rows.append([o.replace("_impression", ""), o, "organic_media (adstock + saturation)"])
    for t in roles["non_media_treatments"]:
        rows.append([t, t, "non_media_treatments"])
    for c in roles["controls"]:
        rows.append([c, c, "controls"])
    rows += [["KPI", roles["kpi"], "kpi (kpi_type = non_revenue)"],
             ["KPI", roles["revenue_per_kpi"], "revenue_per_kpi (turns KPI into money)"],
             ["scale", roles["population"], "population (geo model only)"]]
    return pd.DataFrame(rows, columns=["name", "column", "meridian_role"])


# ---------------------------------------------------------------------------
# 5. Naive benchmarks on the hold-out
# ---------------------------------------------------------------------------

def naive_benchmarks(df: pd.DataFrame, roles: dict, start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Three no-model forecasts of each region's weekly revenue in the hold-out.

    - last 13 weeks:   each region's average over the 13 weeks before the hold-out
    - same week last year: each region's revenue 52 weeks earlier
    - training average: each region's average over all training weeks
    Scored at region-week level (wMAPE) and national weekly level (MAPE, R2).
    """
    d = df[[GEO, TIME, "is_holdout"]].copy()
    d["revenue"] = df[roles["kpi"]] * df[roles["revenue_per_kpi"]]
    d = d.sort_values([GEO, TIME])
    train = d[~d["is_holdout"]]
    before = train[train[TIME] < start]

    last13 = before.groupby(GEO).tail(13).groupby(GEO)["revenue"].mean()
    train_mean = train.groupby(GEO)["revenue"].mean()
    d["same_week_last_year"] = d.groupby(GEO)["revenue"].shift(52)
    d["last_13_weeks"] = d[GEO].map(last13)
    d["training_average"] = d[GEO].map(train_mean)

    hold = d[d["is_holdout"]].copy()
    rows = []
    names = ["last_13_weeks", "same_week_last_year", "training_average"]
    nat_actual = hold.groupby(TIME)["revenue"].sum()
    for n in names:
        err = hold[n] - hold["revenue"]
        nat_pred = hold.groupby(TIME)[n].sum()
        nat_err = nat_pred - nat_actual
        ss_res = float((nat_err ** 2).sum())
        ss_tot = float(((nat_actual - nat_actual.mean()) ** 2).sum())
        rows.append({
            "benchmark": n,
            "region_week_wMAPE": round(float(err.abs().sum() / hold["revenue"].sum()), 4),
            "national_weekly_MAPE": round(float((nat_err.abs() / nat_actual).mean()), 4),
            "national_R2": round(1 - ss_res / ss_tot, 3),
            "national_total_bias": round(float(nat_err.sum() / nat_actual.sum()), 4),
        })
    nat_lines = pd.concat([nat_actual.rename("actual")] +
                          [hold.groupby(TIME)[n].sum() for n in names], axis=1)
    return pd.DataFrame(rows), nat_lines


def chart_benchmarks(df: pd.DataFrame, roles: dict, nat_lines: pd.DataFrame,
                     start: pd.Timestamp, end: pd.Timestamp, out: Path) -> None:
    rev = (df[roles["kpi"]] * df[roles["revenue_per_kpi"]]).groupby(df[TIME]).sum() / 1e6
    lo = start - pd.Timedelta(weeks=30)
    hi = end + pd.Timedelta(weeks=8)
    rev = rev[(rev.index >= lo) & (rev.index <= hi)]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axvspan(start - pd.Timedelta(days=3), end + pd.Timedelta(days=3), color=GREY, alpha=0.15, lw=0)
    ax.plot(rev.index, rev.values, color=INK, lw=2, label="Actual revenue")
    ax.plot(nat_lines.index, nat_lines["last_13_weeks"] / 1e6, color=BLUE, lw=2,
            label="Naive: last 13 weeks' average")
    ax.plot(nat_lines.index, nat_lines["same_week_last_year"] / 1e6, color=ORANGE, lw=2,
            label="Naive: same week last year")
    ax.text(start, ax.get_ylim()[0], " held-out weeks", color=INK_2, fontsize=9, va="bottom")
    ax.set_title("Hold-out weeks: the bar the model has to beat (weekly revenue, all regions, millions)",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncols=3)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.15)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config()
    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_clean(cfg)
    roles = load_roles(cfg)
    start, end = holdout_window(df)
    print(f"Hold-out (from step 0): {start.date()} to {end.date()}")

    nat = national_frame(df, roles)
    geo_data = build_input_data(df, roles, "geo")
    nat_data = build_input_data(nat, roles, "national")
    print(f"Geo input: {geo_data.media.shape} (regions, weeks, channels); "
          f"national input: {nat_data.media.shape}")

    checks = split_checks(df, nat, roles, cfg, start, end, geo_data, nat_data)
    inputs = model_inputs_table(roles)
    bench, nat_lines = naive_benchmarks(df, roles, start)

    checks.to_csv(out_dir / "step2_split_summary.csv", index=False)
    inputs.to_csv(out_dir / "step2_model_inputs.csv", index=False)
    bench.to_csv(out_dir / "step2_naive_benchmarks.csv", index=False)
    chart_benchmarks(df, roles, nat_lines, start, end, out_dir / "step2_holdout_benchmarks.png")

    with pd.option_context("display.width", 160, "display.max_colwidth", 70):
        print("\n--- Split and data checks ---")
        print(checks.to_string(index=False))
        print("\n--- Model inputs ---")
        print(inputs.to_string(index=False))
        print("\n--- Naive forecasts on the hold-out (the bar to beat in step 6) ---")
        print(bench.to_string(index=False))
    if not checks["ok"].all():
        raise SystemExit("Some checks failed - see the table above")
    print(f"\nSaved: tables and chart in {out_dir}")


if __name__ == "__main__":
    main()
