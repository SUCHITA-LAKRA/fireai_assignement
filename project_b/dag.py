"""
Project B - Step 1: Causal structure (DAG) before modelling.

What this script does
---------------------
1. Writes down our assumed causal graph: what causes what among media,
   promo, organic, controls, region size, season and revenue.
2. Tests the assumptions the graph depends on, using the data:
   a. Do controls / promo / organic move together with paid spend?
      (if yes and they also affect revenue, they are confounders: leaving
      them out would bias the media effect)
   b. Is each control a confounder or a mediator? (the sign test below)
   c. How much does the media effect change as each variable is added?
      (the "bias path": a quick linear check, not the final model)
   d. How much of each variable moves with time (season) vs region?
3. Saves the variable-role table that step 3 uses to build the model.
4. Draws the DAG and the bias-path chart.

How to run (from the repo root, after step 0)
---------------------------------------------
    python -m project_b.dag

Outputs
-------
    outputs/project_b/step1_dag.png              the causal graph
    outputs/project_b/step1_bias_path.png        media effect as variables are added
    outputs/project_b/step1_variable_roles.csv   every variable: role, how it enters the model, why
    outputs/project_b/step1_confounding_checks.csv  correlations behind the graph
    outputs/project_b/step1_bias_path.csv        the numbers behind the bias chart

Important: only the TRAINING weeks are used (hold-out weeks excluded), so
nothing here peeks at the weeks we later forecast.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_b.config import load_config  # noqa: E402

# Colours by role (same meaning in every Project B chart)
TEAL = "#1E6F6C"     # treatment (things we decide: media, promo)
GOLD = "#A77A30"     # confounder (affects both spend and revenue)
BLUE = "#2A6FA8"     # mediator (on the path from spend to revenue)
PURPLE = "#7C4B8E"   # outcome
GREY = "#8A948E"     # unobserved
INK = "#1F2926"
INK_2 = "#5D6A63"

CONTROLS_LABEL = {"competitor_sales_control": "competitor sales",
                  "sentiment_score_control": "sentiment"}


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------

def load_training(cfg: dict) -> pd.DataFrame:
    path = Path(cfg["paths"]["processed_dir"]) / "meridian_geo_weekly.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run step 0 first: "
                                f"python -m project_b.data_prep")
    df = pd.read_parquet(path)
    df = df[~df["is_holdout"]].copy()
    c = cfg["columns"]
    per_k = df[c["population"]] / 1000
    # Per 1,000 people, so region size does not drive the comparisons
    df["paid_spend_pc"] = df["total_paid_spend"] / per_k
    df["organic_pc"] = df[f"{c['organic_channels'][0]}_impression"] / per_k
    df["revenue_pc"] = df["revenue_per_1k_pop"]
    df = df.sort_values([c["geo"], c["time"]])
    df["paid_spend_pc_lag1"] = df.groupby(c["geo"])["paid_spend_pc"].shift(1)
    return df.dropna(subset=["paid_spend_pc_lag1"])


def remove_geo_and_week(df: pd.DataFrame, col: str, geo: str, time: str) -> pd.Series:
    """Subtract each region's average and each week's average.

    What is left is how this region-week differs from usual, after taking
    out region size / baseline and anything that hit all regions that week
    (season, holidays, national news). Correlations on this are not driven
    by big regions or by a shared calendar.
    """
    x = df[col]
    return (x - x.groupby(df[geo]).transform("mean")
            - x.groupby(df[time]).transform("mean") + x.mean())


# ---------------------------------------------------------------------------
# 2. Checks behind the graph
# ---------------------------------------------------------------------------

def confounding_checks(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = cfg["columns"]
    geo, time = c["geo"], c["time"]
    candidates = c["controls"] + c["non_media_treatments"] + ["organic_pc"]
    cols = candidates + ["paid_spend_pc", "paid_spend_pc_lag1", "revenue_pc"]
    w = pd.DataFrame({col: remove_geo_and_week(df, col, geo, time) for col in cols})

    rows = []
    for v in candidates:
        x = df[v]
        rows.append({
            "variable": v,
            "corr_with_paid_spend_same_week": round(float(w[v].corr(w["paid_spend_pc"])), 3),
            "corr_with_paid_spend_last_week": round(float(w[v].corr(w["paid_spend_pc_lag1"])), 3),
            "corr_with_revenue": round(float(w[v].corr(w["revenue_pc"])), 3),
            "share_of_variation_from_week": round(float(
                x.groupby(df[time]).transform("mean").var() / x.var()), 2),
            "share_of_variation_from_region": round(float(
                x.groupby(df[geo]).transform("mean").var() / x.var()), 2),
        })
    for v in ["paid_spend_pc", "revenue_pc"]:
        x = df[v]
        rows.append({"variable": v,
                     "share_of_variation_from_week": round(float(
                         x.groupby(df[time]).transform("mean").var() / x.var()), 2),
                     "share_of_variation_from_region": round(float(
                         x.groupby(df[geo]).transform("mean").var() / x.var()), 2)})

    # Yearly seasonality: does the national series repeat every 52 weeks?
    nat = df.groupby(time)[["paid_spend_pc", "revenue_pc"]].mean()
    for v in ["paid_spend_pc", "revenue_pc"]:
        rows.append({"variable": f"{v} (national)",
                     "yearly_repeat_autocorr_52w": round(float(nat[v].autocorr(52)), 2)})
    # Spend autocorrelation tells us how strong the "last week" test is
    rows.append({"variable": "paid_spend_pc (within region)",
                 "week_to_week_autocorr": round(float(
                     df.groupby(geo)["paid_spend_pc"].apply(lambda s: s.autocorr(1)).mean()), 2)})
    return pd.DataFrame(rows)


def bias_path(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Linear effect of paid spend on revenue as we add variables one by one.

    An ordinary least-squares regression (statsmodels):
        revenue per 1k people ~ paid spend per 1k people + [added variables]
                                + one dummy per region + one dummy per week
    The region dummies take out region size / baseline; the week dummies take
    out anything that hit every region that week (season, holidays).
    This is a rough same-week check, NOT the final ROI (no adstock, no
    saturation). What matters is how much the spend effect MOVES when a
    variable is added.
    """
    import statsmodels.formula.api as smf

    c = cfg["columns"]
    geo, time = c["geo"], c["time"]
    steps = [
        ("Spend only", []),
        ("+ competitor sales", ["competitor_sales_control"]),
        ("+ sentiment", ["competitor_sales_control", "sentiment_score_control"]),
        ("+ promo", ["competitor_sales_control", "sentiment_score_control", "Promo"]),
        ("+ organic", ["competitor_sales_control", "sentiment_score_control", "Promo",
                       "organic_pc"]),
    ]
    data = df.assign(week=df[time].astype(str))
    rows = []
    for label, extra in steps:
        formula = " + ".join(["revenue_pc ~ paid_spend_pc", *extra, f"C({geo})", "C(week)"])
        fit = smf.ols(formula, data=data).fit()
        low, high = fit.conf_int(alpha=0.10).loc["paid_spend_pc"]
        rows.append({"model": label, "spend_effect": round(float(fit.params["paid_spend_pc"]), 3),
                     "ci90_low": round(float(low), 3), "ci90_high": round(float(high), 3)})
    return pd.DataFrame(rows)


def mediator_or_confounder(checks: pd.DataFrame, path: pd.DataFrame) -> pd.DataFrame:
    """Sign test for each control.

    If ads CAUSED the control (mediator) and the control then raised sales,
    adding it would LOWER the spend effect (it would soak up part of the
    ad's effect). If the control is a common cause (confounder), adding it
    can move the spend effect either way - often up, when it pushes spend
    up and revenue down. We look at which direction the data shows.
    """
    effect = dict(zip(path["model"], path["spend_effect"]))
    moves = {
        "competitor_sales_control": effect["+ competitor sales"] - effect["Spend only"],
        "sentiment_score_control": effect["+ sentiment"] - effect["+ competitor sales"],
        "Promo": effect["+ promo"] - effect["+ sentiment"],
        "organic_pc": effect["+ organic"] - effect["+ promo"],
    }
    rows = []
    for _, r in checks.dropna(subset=["corr_with_paid_spend_same_week"]).iterrows():
        v = r["variable"]
        rows.append({"variable": v,
                     "moves_with_spend": r["corr_with_paid_spend_same_week"],
                     "relation_to_revenue": r["corr_with_revenue"],
                     "spend_effect_change_when_added": round(moves[v], 3)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. The variable-role table (step 3 builds the model from this)
# ---------------------------------------------------------------------------

def variable_roles(cfg: dict) -> pd.DataFrame:
    c = cfg["columns"]
    rows = []
    for ch in c["paid_channels"]:
        rows.append([f"{ch}_impression", "treatment (paid media)",
                     "media input, with adstock + saturation",
                     "What we want to measure. Impressions are what people see; "
                     "spend causes impressions."])
        rows.append([f"{ch}_spend", "cost of the treatment",
                     "media spend, used only to turn effects into ROI",
                     "Price per 1,000 impressions is fixed, so spend and impressions are "
                     "copies. Putting both in as drivers would double-count."])
    rows += [
        [f"{c['organic_channels'][0]}_impression", "treatment (unpaid media)",
         "organic media, with adstock + saturation",
         "Can affect revenue; no ROI because it costs nothing. Moves with paid spend, "
         "but adding it barely changes the paid effect, so whether it is a mediator "
         "hardly matters."],
        ["Promo", "treatment (non-media)", "non-media treatment, same week",
         "Planned together with media (moves with spend) and lifts revenue. Left out, "
         "its lift would be credited to ads."],
        ["competitor_sales_control", "confounder", "control variable",
         "Rises with our spend and pushes our revenue down. Left out, ads look "
         "harmful."],
        ["sentiment_score_control", "confounder", "control variable",
         "Rises with our spend; adding it RAISES the spend effect, the opposite of what "
         "a mediator would do, so it is treated as a common cause, not a result of ads."],
        ["population", "scale", "population scaling (per-person modelling)",
         "Region size drives both spend and revenue (corr 0.997 / 0.97)."],
        ["geo", "confounder (baseline per region)", "region intercepts, hierarchical",
         "Regions differ in baseline sales for reasons unrelated to ads."],
        ["time", "confounder (season, calendar)", "time effects (smooth trend / knots)",
         "Spend repeats yearly (autocorr 0.49 at 52 weeks); a shared calendar can "
         "drive both spend and revenue."],
        ["conversions", "outcome", "KPI",
         "What ads try to move."],
        ["revenue_per_conversion", "outcome conversion", "revenue per KPI (turns "
         "conversions into money)",
         "Not a cause of anything; almost constant. Lets ROI be in money."],
    ]
    return pd.DataFrame(rows, columns=["variable", "causal_role", "enters_model_as", "why"])


# ---------------------------------------------------------------------------
# 4. Charts
# ---------------------------------------------------------------------------

def _node(ax, xy, text, colour, w=2.3, h=0.78, dashed=False, sub=None):
    x, y = xy
    box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.14",
                         fc=colour + "1A", ec=colour, lw=1.6,
                         ls=(0, (4, 3)) if dashed else "-")
    ax.add_patch(box)
    ty = y + (0.12 if sub else 0)
    ax.text(x, ty, text, ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=INK)
    if sub:
        ax.text(x, y - 0.2, sub, ha="center", va="center", fontsize=8.5, color=INK_2)


def _arrow(ax, a, b, colour=GREY, rad=0.0, dashed=False, lw=1.3):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13,
                                 color=colour, lw=lw, shrinkA=2, shrinkB=2,
                                 connectionstyle=f"arc3,rad={rad}",
                                 ls=(0, (4, 3)) if dashed else "-"))


def chart_dag(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    W, H = 2.5, 0.8

    # Row 1: confounders (affect our spend AND our revenue)
    conf = [((1.6, 6.5), "Season / week", "time effects"),
            ((4.55, 6.5), "Region size", "population, baseline"),
            ((7.5, 6.5), "Competitor sales", "control"),
            ((10.45, 6.5), "Sentiment", "control")]
    for xy, t, sub in conf:
        _node(ax, xy, t, GOLD, w=W, h=H, sub=sub)
    # A shared band: every confounder feeds both spend and revenue
    band_y = 5.25
    ax.add_patch(FancyBboxPatch((0.6, band_y - 0.2), 10.8, 0.4,
                                boxstyle="round,pad=0.01,rounding_size=0.18",
                                fc=GOLD + "14", ec=GOLD, lw=1.0))
    ax.text(6.0, band_y, "each of these moves our spend AND our revenue  →  must be in the model",
            ha="center", va="center", fontsize=9.5, color=GOLD, fontweight="bold")
    for xy, _, _ in conf:
        _arrow(ax, (xy[0], xy[1] - H / 2), (xy[0], band_y + 0.2), GOLD)

    # Row 2: treatment path
    spend, imp, rev = (2.4, 3.8), (6.3, 3.8), (10.1, 2.5)
    _node(ax, spend, "Paid spend", TEAL, w=W, h=H, sub="Channel0–4")
    _node(ax, imp, "Impressions", BLUE, w=W, h=H, sub="Channel0–4 · fixed price")
    _node(ax, rev, "Revenue", PURPLE, w=2.8, h=1.0, sub="conversions × revenue/conv")
    _arrow(ax, (spend[0], band_y - 0.2), (spend[0], spend[1] + H / 2), GOLD, lw=2)
    _arrow(ax, (rev[0], band_y - 0.2), (rev[0], rev[1] + 0.5), GOLD, lw=2)
    _arrow(ax, (spend[0] + W / 2, spend[1]), (imp[0] - W / 2, imp[1]), TEAL, lw=2.2)
    _arrow(ax, (imp[0] + W / 2, imp[1]), (rev[0] - 1.4, rev[1] + 0.3), TEAL, lw=2.2)
    ax.text(8.25, 3.62, "adstock: this week's\nand past weeks' ads", fontsize=8.5,
            color=TEAL, ha="left", va="bottom")

    # Row 3: other things we do
    promo, org, plan = (6.3, 2.2), (6.3, 0.95), (2.4, 1.6)
    _node(ax, promo, "Promo", TEAL, w=W, h=H, sub="non-media treatment")
    _node(ax, org, "Organic", TEAL, w=W, h=H, sub="unpaid media")
    _node(ax, plan, "Campaign calendar", GREY, w=W, h=H, dashed=True, sub="not in the data")
    _arrow(ax, (promo[0] + W / 2, promo[1]), (rev[0] - 1.4, rev[1] - 0.05), TEAL, lw=2.2)
    _arrow(ax, (org[0] + W / 2, org[1]), (rev[0] - 1.4, rev[1] - 0.35), TEAL, lw=2.2)
    _arrow(ax, (plan[0], plan[1] + H / 2), (spend[0], spend[1] - H / 2), GREY, dashed=True)
    _arrow(ax, (plan[0] + W / 2, plan[1] + 0.15), (promo[0] - W / 2, promo[1]), GREY, dashed=True)
    _arrow(ax, (plan[0] + W / 2, plan[1] - 0.15), (org[0] - W / 2, org[1]), GREY, dashed=True)
    ax.text(0.6, 0.62, "planned together: spend, promo and organic\nrise and fall in the same weeks",
            fontsize=8.5, color=INK_2, ha="left", va="top")

    # Legend
    lx, ly = 5.1, 0.25
    for i, (lab, col, dash) in enumerate([("treatment", TEAL, False), ("mediator", BLUE, False),
                                          ("confounder", GOLD, False), ("outcome", PURPLE, False),
                                          ("not observed", GREY, True)]):
        x = lx + i * 1.38
        ax.add_patch(FancyBboxPatch((x, ly - 0.1), 0.28, 0.2,
                                    boxstyle="round,pad=0.01,rounding_size=0.05",
                                    fc=col + "1A", ec=col, lw=1.2,
                                    ls=(0, (3, 2)) if dash else "-"))
        ax.text(x + 0.38, ly, lab, va="center", fontsize=8.5, color=INK_2)

    ax.set_title("Assumed causal graph: what drives revenue, and what drives our spend",
                 loc="left", fontsize=12.5, color=INK, pad=4)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_bias_path(path: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.8))
    y = np.arange(len(path))[::-1]
    vals = path["spend_effect"].to_numpy()
    colours = ["#B23B3B" if v < 0 else TEAL for v in vals]
    ax.barh(y, vals, color=colours, height=0.55)
    ax.errorbar(vals, y, xerr=[vals - path["ci90_low"], path["ci90_high"] - vals],
                fmt="none", ecolor=INK_2, elinewidth=1, capsize=3)
    for yi, v, hi, lo in zip(y, vals, path["ci90_high"], path["ci90_low"]):
        x = hi + 0.03 if v >= 0 else lo - 0.03
        ax.text(x, yi, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right",
                fontsize=10, color=INK, fontweight="bold")
    ax.axvline(0, color=INK_2, lw=1)
    ax.set_yticks(y, path["model"])
    ax.set_xlim(min(-0.6, path["ci90_low"].min() - 0.2), path["ci90_high"].max() + 0.25)
    ax.set_xlabel("Extra revenue per 1 of paid spend, same week (per 1,000 people; rough linear check). "
                  "Each row adds one variable to the row above.",
                  fontsize=9, color=INK_2)
    ax.set_title("Leave out competitor sales, sentiment and promo, and ads look harmful",
                 loc="left", fontsize=11.5, color=INK)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors=INK_2, labelsize=9.5)
    ax.grid(axis="x", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
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

    df = load_training(cfg)
    print(f"Using {df[cfg['columns']['time']].nunique()} training weeks "
          f"(hold-out weeks excluded)")

    checks = confounding_checks(df, cfg)
    path = bias_path(df, cfg)
    sign = mediator_or_confounder(checks, path)
    roles = variable_roles(cfg)

    checks.to_csv(out_dir / "step1_confounding_checks.csv", index=False)
    path.to_csv(out_dir / "step1_bias_path.csv", index=False)
    sign.to_csv(out_dir / "step1_mediator_or_confounder.csv", index=False)
    roles.to_csv(out_dir / "step1_variable_roles.csv", index=False)

    chart_dag(out_dir / "step1_dag.png")
    chart_bias_path(path, out_dir / "step1_bias_path.png")

    with pd.option_context("display.width", 160, "display.max_colwidth", 70):
        print("\n--- Do these move with paid spend / revenue? (region & week averages removed) ---")
        print(checks.to_string(index=False))
        print("\n--- Bias path: spend effect as variables are added ---")
        print(path.to_string(index=False))
        print("\n--- Mediator or confounder? ---")
        print(sign.to_string(index=False))
        print("\n--- Variable roles (used by step 3) ---")
        print(roles[["variable", "causal_role", "enters_model_as"]].to_string(index=False))
    print(f"\nSaved: charts and tables in {out_dir}")


if __name__ == "__main__":
    main()
