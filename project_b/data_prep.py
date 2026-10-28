"""
Project B - Step 0: Load, check and prepare the Meridian geo data.

What this script does
---------------------
1. Reads geo_all_channels.csv (40 geos x 156 weeks).
2. Hard checks that stop the run if the data is broken: missing columns,
   missing values, duplicate geo-weeks, gaps in the weekly calendar,
   negative spend / impressions / conversions.
3. Adds the columns later steps need: revenue, per-capita versions,
   and an `is_holdout` flag for the 13 validation weeks.
4. Runs the data-quality checks behind DATA_ISSUES.md and saves them
   as a table, so every number in that file is reproducible.
5. Checks the Channel3 experiment against the data: did spend actually
   drop in Geo0-Geo7 during the test weeks?
6. Saves charts for the memo and one clean file for modelling.

How to run (from the repo root)
-------------------------------
    python -m project_b.data_prep
    python -m project_b.data_prep --holdout-start 2023-07-03

Outputs
-------
    data/processed/meridian_geo_weekly.parquet   clean data + holdout flag
    outputs/project_b/step0_checks.csv           every check, value, verdict
    outputs/project_b/step0_experiment_check.csv Channel3 spend before/during test
    outputs/project_b/step0_*.png                charts
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no screen needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Lets the file run with an IDE "Run" button, not only with `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_b.config import load_config  # noqa: E402

# Chart colours (fixed roles, same in every Project B chart)
BLUE = "#2a78d6"     # test geos / main series
ORANGE = "#eb6834"   # control geos / second series
GREY = "#8c8b86"     # shading, reference lines
INK = "#0b0b0b"
INK_2 = "#52514e"


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------

def expected_columns(cfg: dict) -> list[str]:
    c = cfg["columns"]
    cols = [c["geo"], c["time"], c["kpi"], c["revenue_per_kpi"], c["population"]]
    for ch in c["paid_channels"]:
        cols += [f"{ch}_impression", f"{ch}_spend"]
    cols += [f"{ch}_impression" for ch in c["organic_channels"]]
    cols += c["non_media_treatments"] + c["controls"]
    return cols


def load_data(path: Path, cfg: dict) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Input file in {path}")
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")]) #Drop the column with blank header

    missing = [c for c in expected_columns(cfg) if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from the data: {missing}")

    c = cfg["columns"]
    df[c["time"]] = pd.to_datetime(df[c["time"]])
    # Sort geos by number (Geo2 before Geo10), then by week.
    df["geo_num"] = df[c["geo"]].str.extract(r"(\d+)").astype(int)
    df = df.sort_values(["geo_num", c["time"]]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. Hard checks - stop the run if any fail
# ---------------------------------------------------------------------------

def hard_checks(df: pd.DataFrame, cfg: dict) -> None:
    c = cfg["columns"]
    geo, time = c["geo"], c["time"]
    problems = []

    n_geo, n_week = df[geo].nunique(), df[time].nunique()
    if len(df) != n_geo * n_week:
        problems.append(f"Not a full geo x week grid: {len(df)} rows vs " f"{n_geo} x {n_week} as mentioned in Project")
    if df.duplicated([geo, time]).any():
        problems.append("Duplicate geo-week rows")
    if df[expected_columns(cfg)].isna().any().any():
        problems.append("Missing values present")

    gaps = df[time].drop_duplicates().sort_values().diff().dropna().dt.days
    if not (gaps == 7).all():
        problems.append(f"Weeks are not evenly 7 days apart: {gaps.unique()}")

    must_be_non_negative = ([f"{ch}_spend" for ch in c["paid_channels"]]
                            + [f"{ch}_impression" for ch in c["paid_channels"]]
                            + [f"{ch}_impression" for ch in c["organic_channels"]]
                            + c["non_media_treatments"]
                            + [c["kpi"], c["revenue_per_kpi"], c["population"]])
    negative = [col for col in must_be_non_negative if (df[col] < 0).any()]
    if negative:
        problems.append(f"Negative values in: {negative}")

    if (df.groupby(geo)[c["population"]].nunique() > 1).any():
        problems.append("Population changes over time within a geo")

    if problems:
        raise ValueError("Data failed hard checks:\n  - " + "\n  - ".join(problems))
    print(f"Hard checks passed: {n_geo} geos x {n_week} weeks = {len(df):,} rows, "
          f"{df[time].min().date()} to {df[time].max().date()}")


# ---------------------------------------------------------------------------
# 3. Derived columns
# ---------------------------------------------------------------------------

def add_columns(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = cfg["columns"]
    df["revenue"] = df[c["kpi"]] * df[c["revenue_per_kpi"]]
    df["total_paid_spend"] = df[[f"{ch}_spend" for ch in c["paid_channels"]]].sum(axis=1)
    # Per 1,000 people: removes geo size, so big and small geos are comparable.
    per_k = df[c["population"]] / 1000
    df["revenue_per_1k_pop"] = df["revenue"] / per_k
    for ch in c["paid_channels"]:
        df[f"{ch}_spend_per_1k_pop"] = df[f"{ch}_spend"] / per_k
    return df


def resolve_holdout(df: pd.DataFrame, cfg: dict, start: str | None) -> pd.Timestamp:
    """Return the first holdout week. Default = the last `weeks` weeks."""
    time = cfg["columns"]["time"]
    n = cfg["holdout"]["weeks"]
    weeks = np.sort(df[time].unique())
    start = start or cfg["holdout"]["default_start"]
    if start is None:
        return pd.Timestamp(weeks[-n])

    ts = pd.Timestamp(start)
    if ts not in set(pd.to_datetime(weeks)):
        # Snap to the next week in the data (weeks start on Mondays here).
        later = [w for w in pd.to_datetime(weeks) if w >= ts]
        if not later:
            raise ValueError(f"--holdout-start {start} is after the last week in the data")
        print(f"Note: {start} is not a week start in the data; using {later[0].date()}")
        ts = later[0]
    if (pd.to_datetime(weeks) >= ts).sum() < n:
        print(f"Warning: fewer than {n} weeks after {ts.date()}; holdout is shorter")
    return ts


def add_holdout_flag(df: pd.DataFrame, cfg: dict, start: pd.Timestamp) -> pd.DataFrame:
    time = cfg["columns"]["time"]
    end = start + pd.Timedelta(weeks=cfg["holdout"]["weeks"])
    df["is_holdout"] = (df[time] >= start) & (df[time] < end)
    n_weeks = df.loc[df["is_holdout"], time].nunique()
    n_train = df.loc[~df["is_holdout"], time].nunique()
    print(f"Holdout: {n_weeks} weeks from {start.date()}; training on {n_train} weeks")
    return df


# ---------------------------------------------------------------------------
# 4. Data-quality checks (feed DATA_ISSUES.md)
# ---------------------------------------------------------------------------

def quality_checks(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = cfg["columns"]
    geo, time = c["geo"], c["time"]
    chans = c["paid_channels"]
    rows = []

    def add(area, check, value, note):
        rows.append({"area": area, "check": check, "value": value, "note": note})

    # Cost per 1,000 impressions: is it ever different across geos or weeks?
    for ch in chans:
        s, imp = df[f"{ch}_spend"], df[f"{ch}_impression"]
        cpm = (s / imp * 1000)[imp > 0]
        mismatch = int(((s > 0) & (imp == 0)).sum() + ((imp > 0) & (s == 0)).sum())
        add("spend vs impressions", f"{ch} CPM range",
            f"{cpm.min():.4f} to {cpm.max():.4f}",
            f"{mismatch} rows where only one of spend/impressions is zero")

    # Zero-spend weeks
    for ch in chans:
        add("zero spend", f"{ch} share of geo-weeks with zero spend",
            round(float((df[f"{ch}_spend"] == 0).mean()), 3), "")

    # Geo size drives everything
    g = df.groupby(geo).agg(pop=(c["population"], "first"), rev=("revenue", "sum"),
                            spend=("total_paid_spend", "sum"))
    add("geo size", "corr(population, total revenue) across geos",
        round(float(g["pop"].corr(g["rev"])), 3), "")
    add("geo size", "corr(population, total paid spend) across geos",
        round(float(g["pop"].corr(g["spend"])), 3), "")
    add("geo size", "largest / smallest population",
        round(float(g["pop"].max() / g["pop"].min()), 1), "")

    # Channels moving together (harder to separate their effects)
    nat = df.groupby(time)[[f"{ch}_spend" for ch in chans]].sum()
    corr_nat = nat.corr().where(~np.eye(len(chans), dtype=bool))
    corr_geo = df[[f"{ch}_spend" for ch in chans]].corr().where(
        ~np.eye(len(chans), dtype=bool))
    i, j = np.unravel_index(np.nanargmax(corr_nat.values), corr_nat.shape)
    add("collinearity", "highest spend correlation between two channels (national weekly)",
        round(float(corr_nat.values[i, j]), 2), f"{chans[i]} vs {chans[j]}")
    add("collinearity", "highest spend correlation between two channels (geo-week)",
        round(float(np.nanmax(corr_geo.values)), 2), "")

    # Units of the outcome
    conv_pc = (df[c["kpi"]] / df[c["population"]])
    add("outcome", "conversions per person per week (min / median / max)",
        f"{conv_pc.min():.1f} / {conv_pc.median():.1f} / {conv_pc.max():.1f}",
        "far above 1 - treat as an index, not people")
    rpc = df[c["revenue_per_kpi"]]
    add("outcome", "revenue_per_conversion range",
        f"{rpc.min():.5f} to {rpc.max():.5f}", f"coefficient of variation {rpc.std() / rpc.mean():.4f}")
    add("outcome", "total revenue / total paid spend (all weeks)",
        round(float(df["revenue"].sum() / df["total_paid_spend"].sum()), 2),
        "upper bound on blended return - most revenue is baseline, not media")
    yearly = df.groupby(df[time].dt.year)["revenue"].sum() / df.groupby(
        df[time].dt.year)[time].nunique()
    add("outcome", "average weekly revenue by year",
        "; ".join(f"{y}: {v / 1e6:.2f}M" for y, v in yearly.items()), "no clear trend")

    # Promo and controls
    p = df[c["non_media_treatments"][0]]
    add("promo", "share of geo-weeks with Promo > 0", round(float((p > 0).mean()), 3),
        f"{p[p > 0].nunique()} distinct non-zero values - continuous, not 0/1")
    for ctl in c["controls"]:
        add("controls", f"{ctl} mean / std",
            f"{df[ctl].mean():.2f} / {df[ctl].std():.2f}", "already standardised; negatives are fine")

    # Heavy tails in spend
    for ch in chans:
        s = df[f"{ch}_spend"]
        add("outliers", f"{ch} spend max / 99th percentile",
            round(float(s.max() / s.quantile(0.99)), 2), "kept - no sign of data error")

    # Budget context
    last13 = df[time] > df[time].max() - pd.Timedelta(weeks=13)
    spend13 = df.loc[last13, "total_paid_spend"].sum()
    add("budget", "total paid spend, last 13 weeks", round(float(spend13)), "")
    add("budget", "next-quarter budget vs last 13 weeks",
        f"{cfg['budget']['next_quarter_total'] / spend13 - 1:+.1%}", "budget is a cut")
    for ch in chans:
        add("budget", f"{ch} spend last 13 weeks",
            round(float(df.loc[last13, f"{ch}_spend"].sum())), "")

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 5. Experiment check: did Channel3 spend actually drop in the test geos?
# ---------------------------------------------------------------------------

def experiment_check(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    c, e = cfg["columns"], cfg["experiment"]
    geo, time = c["geo"], c["time"]
    ch = e["channel"]
    start = pd.Timestamp(e["start"])
    end = start + pd.Timedelta(weeks=e["weeks"])
    pre_start = start - pd.Timedelta(weeks=e["pre_period_weeks"])

    is_test = df[geo].isin(e["test_geos"])
    period = np.select([(df[time] >= pre_start) & (df[time] < start),
                        (df[time] >= start) & (df[time] < end)],
                       ["before", "during"], default="other")
    sub = df.assign(group=np.where(is_test, "test", "control"), period=period)
    sub = sub[sub["period"] != "other"]

    summary = (sub.groupby(["group", "period"])
                  .agg(geo_weeks=(f"{ch}_spend", "size"),
                       zero_spend_weeks=(f"{ch}_spend", lambda s: int((s == 0).sum())),
                       spend_per_geo_week=(f"{ch}_spend", "mean"),
                       spend_per_1k_pop=(f"{ch}_spend_per_1k_pop", "mean"),
                       revenue_per_1k_pop=("revenue_per_1k_pop", "mean"))
                  .reindex(pd.MultiIndex.from_product(
                      [["test", "control"], ["before", "during"]]))
                  .reset_index().rename(columns={"level_0": "group", "level_1": "period"}))

    # Longest run of consecutive zero-spend weeks in ANY geo: was the channel
    # ever switched off for a sustained period, anywhere, at any time?
    def longest_zero_run(s: pd.Series) -> int:
        z = (s.to_numpy() == 0).astype(int)
        best = run = 0
        for v in z:
            run = run + 1 if v else 0
            best = max(best, run)
        return best

    runs = df.groupby(geo)[f"{ch}_spend"].apply(longest_zero_run)

    pop_rank = df.groupby(geo)[c["population"]].first().rank(method="min").astype(int)
    test_ranks = pop_rank.loc[e["test_geos"]].sort_values()

    t_b, t_d = summary.iloc[0], summary.iloc[1]
    c_b, c_d = summary.iloc[2], summary.iloc[3]
    did_spend = ((t_d.spend_per_1k_pop - t_b.spend_per_1k_pop)
                 - (c_d.spend_per_1k_pop - c_b.spend_per_1k_pop))
    did_rev = ((t_d.revenue_per_1k_pop - t_b.revenue_per_1k_pop)
               - (c_d.revenue_per_1k_pop - c_b.revenue_per_1k_pop))

    # Placebo: run the same revenue comparison for every other 8-week window.
    # If many windows show a gap this big, the test-window gap is just noise.
    n = e["weeks"]
    pre = e["pre_period_weeks"]
    weekly = (df.assign(group=np.where(is_test, "test", "control"))
                .groupby([time, "group"])["revenue_per_1k_pop"].mean().unstack())
    gap = weekly["test"] - weekly["control"]
    placebo = pd.Series({gap.index[k]: gap.iloc[k:k + n].mean() - gap.iloc[k - pre:k].mean()
                         for k in range(pre, len(gap) - n + 1)})
    share_as_big = float((placebo.abs() >= abs(did_rev)).mean())

    findings = pd.DataFrame([
        {"check": f"test geo-weeks with zero {ch} spend during test",
         "value": f"{int(t_d.zero_spend_weeks)} of {int(t_d.geo_weeks)}",
         "expected_if_switched_off": f"{int(t_d.geo_weeks)} of {int(t_d.geo_weeks)}"},
        {"check": f"test geos: {ch} spend change, before -> during",
         "value": f"{t_d.spend_per_geo_week / t_b.spend_per_geo_week - 1:+.0%}",
         "expected_if_switched_off": "-100%"},
        {"check": f"control geos: {ch} spend change, before -> during",
         "value": f"{c_d.spend_per_geo_week / c_b.spend_per_geo_week - 1:+.0%}",
         "expected_if_switched_off": "about 0%"},
        {"check": f"difference-in-differences, {ch} spend per 1k pop per week",
         "value": f"{did_spend:+.2f}",
         "expected_if_switched_off": f"about {-t_b.spend_per_1k_pop:.2f}"},
        {"check": "difference-in-differences, revenue per 1k pop per week",
         "value": f"{did_rev:+.2f}",
         "expected_if_switched_off": "negative if the channel works"},
        {"check": "placebo: share of all other 8-week windows with a revenue gap at least this big",
         "value": f"{share_as_big:.0%} of {len(placebo)} windows (std {placebo.std():.1f})",
         "expected_if_switched_off": "a small share (e.g. < 10%) if the drop is real"},
        {"check": f"longest run of zero {ch} spend weeks, any geo, any time",
         "value": f"{runs.max()} weeks ({', '.join(runs[runs == runs.max()].index)})",
         "expected_if_switched_off": f">= {e['weeks']} weeks in each test geo"},
        {"check": "population rank of test geos (1 = smallest of 40)",
         "value": ", ".join(f"{g}:{r}" for g, r in test_ranks.items()),
         "expected_if_switched_off": "a mix of sizes if geos were chosen at random"},
    ])
    return summary, findings


# ---------------------------------------------------------------------------
# 6. Charts
# ---------------------------------------------------------------------------

def _style(ax):
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)


def chart_national(df: pd.DataFrame, cfg: dict, holdout_start: pd.Timestamp,
                   out: Path) -> None:
    """Two panels, shared time axis: revenue on top, paid spend below."""
    time = cfg["columns"]["time"]
    nat = df.groupby(time)[["revenue", "total_paid_spend"]].sum() / 1e6
    end = holdout_start + pd.Timedelta(weeks=cfg["holdout"]["weeks"])

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    for ax, col, title in [(axes[0], "revenue", "Weekly revenue, all 40 geos (millions)"),
                           (axes[1], "total_paid_spend", "Weekly paid media spend, all 5 channels (millions)")]:
        ax.axvspan(holdout_start, end - pd.Timedelta(days=1), color=GREY, alpha=0.15, lw=0)
        ax.plot(nat.index, nat[col], color=BLUE, lw=2)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_ylim(bottom=0)
        _style(ax)
    axes[0].text(holdout_start, axes[0].get_ylim()[1] * 0.05, "held-out weeks ",
                 color=INK_2, fontsize=9, va="bottom", ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_experiment(df: pd.DataFrame, cfg: dict, out: Path) -> None:
    """Channel3 spend per 1k people: test geos vs control geos, around the test."""
    c, e = cfg["columns"], cfg["experiment"]
    geo, time, ch = c["geo"], c["time"], e["channel"]
    start = pd.Timestamp(e["start"])
    end = start + pd.Timedelta(weeks=e["weeks"])
    window = (df[time] >= start - pd.Timedelta(weeks=16)) & (df[time] < end + pd.Timedelta(weeks=12))
    sub = df[window].assign(group=np.where(df.loc[window, geo].isin(e["test_geos"]),
                                           "Test geos (Geo0–Geo7)", "Control geos (32)"))
    weekly = sub.groupby([time, "group"])[f"{ch}_spend_per_1k_pop"].mean().unstack()

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axvspan(start, end - pd.Timedelta(days=1), color=GREY, alpha=0.15, lw=0)
    ax.text(start, 0, " reported test: spend switched off", color=INK_2,
            fontsize=9, va="bottom")
    ax.plot(weekly.index, weekly["Test geos (Geo0–Geo7)"], color=BLUE, lw=2,
            label="Test geos (Geo0–Geo7)")
    ax.plot(weekly.index, weekly["Control geos (32)"], color=ORANGE, lw=2,
            label="Control geos (32)")
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_title(f"{ch} spend per 1,000 people, weekly average — "
                 "it never drops to zero in the test geos",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_ylim(bottom=-0.5)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_spend_mix(df: pd.DataFrame, cfg: dict, out: Path) -> None:
    """Spend per channel, all weeks vs last 13 weeks, plus zero-spend share."""
    c = cfg["columns"]
    chans = c["paid_channels"]
    time = c["time"]
    last13 = df[time] > df[time].max() - pd.Timedelta(weeks=13)
    spend13 = np.array([df.loc[last13, f"{ch}_spend"].sum() / 1e6 for ch in chans])
    zero = np.array([(df[f"{ch}_spend"] == 0).mean() * 100 for ch in chans])

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    y = np.arange(len(chans))[::-1]
    axes[0].barh(y, spend13, color=BLUE, height=0.6)
    for yi, v in zip(y, spend13):
        axes[0].text(v, yi, f" {v:.1f}M", va="center", fontsize=9, color=INK)
    axes[0].set_title("Spend in the last 13 weeks (millions)", loc="left", fontsize=11)
    axes[1].barh(y, zero, color=BLUE, height=0.6)
    for yi, v in zip(y, zero):
        axes[1].text(v, yi, f" {v:.0f}%", va="center", fontsize=9, color=INK)
    axes[1].set_title("Geo-weeks with zero spend", loc="left", fontsize=11)
    for ax in axes:
        ax.set_yticks(y, chans)
        _style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
        ax.set_xlim(0, ax.get_xlim()[1] * 1.15)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", default=cfg["paths"]["raw_file"])
    parser.add_argument("--holdout-start", default=None,
                        help="first held-out week, e.g. 2023-07-03 (default: last 13 weeks)")
    args = parser.parse_args()

    out_dir = Path(cfg["paths"]["output_dir"])
    proc_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(Path(args.data_file), cfg)
    hard_checks(df, cfg)
    df = add_columns(df, cfg)
    holdout_start = resolve_holdout(df, cfg, args.holdout_start)
    df = add_holdout_flag(df, cfg, holdout_start)

    checks = quality_checks(df, cfg)
    checks.to_csv(out_dir / "step0_checks.csv", index=False)

    summary, findings = experiment_check(df, cfg)
    summary.to_csv(out_dir / "step0_experiment_summary.csv", index=False)
    findings.to_csv(out_dir / "step0_experiment_check.csv", index=False)

    chart_national(df, cfg, holdout_start, out_dir / "step0_national_weekly.png")
    chart_experiment(df, cfg, out_dir / "step0_channel3_experiment.png")
    chart_spend_mix(df, cfg, out_dir / "step0_spend_mix.png")

    out_path = proc_dir / "meridian_geo_weekly.parquet"
    df.to_parquet(out_path, index=False)

    with pd.option_context("display.width", 140, "display.max_colwidth", 60):
        print("\n--- Data-quality checks ---")
        print(checks.to_string(index=False))
        print("\n--- Channel3 experiment: did spend actually drop? ---")
        print(summary.round(2).to_string(index=False))
        print()
        print(findings.to_string(index=False))
    print(f"\nSaved: {out_path}\nSaved: charts and tables in {out_dir}")


if __name__ == "__main__":
    main()
