"""
Project B - Step 7: Split next quarter's budget across Channel0-Channel4.

What this script does
---------------------
Uses the calibrated model (step 5). "Next quarter" is modelled as a repeat of
the last 13 weeks: each channel keeps its recent week-by-week and region-by-
region pattern, scaled up or down. (Meridian's standard assumption; we have no
plan for next quarter's flighting.)

1. The rule: each channel must stay within +/-30% of what it got in the last
   13 weeks. So the total can only be between 70% and 130% of 20.77M,
   i.e. 14.5M to 27.0M. A budget outside that range is IMPOSSIBLE under the
   rule, and the script says so instead of quietly breaking it.
2. For each budget (12M, 18M, 24M) it finds the split that maximises expected
   extra revenue (Meridian's optimiser, posterior mean), within the rule.
   - 12M is below 14.5M, so two honest options are shown: the same cut for
     every channel (-42%), and the best split if channels may be cut up to
     -70% (the +30% cap kept).
3. For every plan it computes the extra revenue from all 2,000 posterior
   draws: mean, 90% range, and how likely the plan beats the simple
   alternative (every channel scaled by the same %).
4. SATURATION STRESS TEST (step 4: the data barely learns how fast returns
   fall). The 2,000 draws are split into thirds by their saturation point:
   "returns fall fast", "middle", "returns fall slowly". The 18M optimisation
   is re-run on each third. If the recommended split holds across all three,
   the advice doesn't hinge on the saturation assumption.

How to run (from the repo root, after step 5)
---------------------------------------------
    python -m project_b.optimise          # ~15 min on 2 CPUs

Outputs
-------
    project_b/budget_allocation.csv                 recommended spend per channel and budget (deliverable)
    outputs/project_b/step7_scenario_summary.csv    one line per plan: total extra revenue, ROI, gain vs simple cut
    outputs/project_b/step7_saturation_stress.csv   18M split under fast / middle / slow saturation
    outputs/project_b/step7_budget_steps.csv        what each extra rupee of budget buys, 12M -> 18M -> 20.8M -> 24M
    outputs/project_b/step7_allocation.png
    outputs/project_b/step7_saturation_stress.png
"""

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="arviz")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_b.config import REPO_ROOT, load_config  # noqa: E402
from project_b.report import load_final_model  # noqa: E402

TEAL, BLUE, ORANGE = "#1E6F6C", "#2a78d6", "#eb6834"
GREY, INK, INK_2 = "#8c8b86", "#0b0b0b", "#52514e"
EPS = 0.01    # keep inside the rule: the optimiser rounds spend to steps of ~10,000


# ---------------------------------------------------------------------------
# 1. Set-up: current spend and the rule
# ---------------------------------------------------------------------------

def current_spend(mmm, weeks: int) -> tuple[np.ndarray, list[str], tuple[str, str]]:
    """Spend per channel over the last `weeks` weeks, channel names, (start, end)."""
    spend = np.asarray(mmm.input_data.media_spend)[:, -weeks:, :].sum(axis=(0, 1))
    chans = [str(c) for c in mmm.input_data.media_channel.values]
    times = [str(t) for t in mmm.input_data.time.values]
    return spend, chans, (times[-weeks], times[-1])


def meridian_bounds(budget: float, current: np.ndarray, down: float, up: float):
    """Turn "each channel within -down / +up of CURRENT spend" into Meridian's terms.

    Meridian measures its bounds from a proportional split of the NEW budget
    (budget x current share). Channel c may go from (1-down)*current_c to
    (1+up)*current_c; relative to budget x share_c that is
        lower = 1 - (1-down) x total_current / budget
        upper = (1+up) x total_current / budget - 1
    Returns None if the budget can't be met within the rule.
    """
    ratio = current.sum() / budget
    lower = 1 - (1 - down) * ratio - EPS
    upper = (1 + up) * ratio - 1 - EPS
    if lower < 0 or upper < 0:
        return None
    return float(min(lower, 1.0)), float(upper)


# ---------------------------------------------------------------------------
# 2. Extra revenue of any plan, from every posterior draw
# ---------------------------------------------------------------------------

def plan_draws(mmm, analyzer, current: np.ndarray, plan: np.ndarray, window) -> np.ndarray:
    """Extra revenue in the window, per draw and channel, if channel c spends plan[c].

    Each channel's media (and spend) is scaled by plan[c] / current[c] in every
    week, so its pattern is kept - the same assumption the optimiser makes.
    """
    from meridian import backend
    from meridian.analysis import tensors

    k = (plan / current).astype(np.float32)
    media = np.asarray(mmm.input_data.media) * k
    spend = np.asarray(mmm.input_data.media_spend) * k
    weeks = [str(t) for t in mmm.input_data.time.values]
    sel = weeks[weeks.index(window[0]): weeks.index(window[1]) + 1]
    new = tensors.DataTensors(media=backend.to_tensor(media, dtype=backend.float_dtype),
                              media_spend=backend.to_tensor(spend, dtype=backend.float_dtype))
    inc = np.asarray(analyzer.incremental_outcome(new_data=new, selected_times=sel,
                                                  include_non_paid_channels=False))
    return inc.reshape(-1, len(plan))


def optimise(mmm, analyzer, budget: float, bounds, window) -> np.ndarray:
    from meridian.analysis import optimizer

    opt = optimizer.BudgetOptimizer(meridian=mmm, analyzer=analyzer)
    res = opt.optimize(start_date=window[0], end_date=window[1], budget=budget,
                       spend_constraint_lower=bounds[0], spend_constraint_upper=bounds[1],
                       confidence_level=0.9)
    return np.asarray(res.optimized_data.spend.values, dtype=float)


# ---------------------------------------------------------------------------
# 3. Saturation subsets
# ---------------------------------------------------------------------------

def saturation_subsets(mmm, n_groups: int = 3) -> dict:
    """Split the posterior draws into groups by their average half-saturation point.

    Low half-saturation point = returns fall fast; high = returns fall slowly.
    Returns {label: InferenceData with only those draws}.
    """
    import arviz as az

    post = mmm.inference_data.posterior
    stacked = post.stack(sample=("chain", "draw"))
    ec = stacked["ec_m"].mean("media_channel").values
    cuts = np.quantile(ec, np.linspace(0, 1, n_groups + 1))
    labels = ["returns fall fast", "middle", "returns fall slowly"]
    out = {}
    for i, lab in enumerate(labels):
        keep = (ec >= cuts[i]) & (ec <= cuts[i + 1]) if i == n_groups - 1 else (ec >= cuts[i]) & (ec < cuts[i + 1])
        sub = stacked.isel(sample=np.where(keep)[0])
        sub = sub.drop_vars(["chain", "draw", "sample"]).rename({"sample": "draw"})
        sub = sub.expand_dims(chain=[0]).assign_coords(draw=np.arange(sub.sizes["draw"]))
        sub = sub.transpose("chain", "draw", ...)
        out[lab] = (az.InferenceData(posterior=sub), float(np.median(ec[keep])))
    return out


# ---------------------------------------------------------------------------
# 4. Charts
# ---------------------------------------------------------------------------

def _style(ax, grid="x"):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis=grid, color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)


def chart_allocation(alloc: pd.DataFrame, chans: list[str], out: Path) -> None:
    # Only the recommended plans: "same % for every channel" is a flat bar by definition
    plans = [p for p in alloc["plan"].unique() if "recommended" in p or "best split" in p]
    colours = [ORANGE, TEAL, BLUE]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    y = np.arange(len(chans))[::-1]
    h = 0.8 / len(plans)
    for i, (p, c) in enumerate(zip(plans, colours)):
        d = alloc[alloc["plan"] == p].set_index("channel").loc[chans]
        off = 0.4 - h * (i + 0.5)
        ax.barh(y + off, d["change_vs_last_13_weeks"] * 100, height=h * 0.9, color=c, label=p)
    ax.axvspan(-30, 30, color=GREY, alpha=0.08, lw=0)
    for x in (-30, 30):
        ax.axvline(x, color=INK_2, lw=1, ls=(0, (4, 3)))
    ax.text(30.5, y.max() + 0.48, "+30% rule", fontsize=8.5, color=INK_2)
    ax.text(-29.5, y.max() + 0.48, "−30% rule", fontsize=8.5, color=INK_2)
    ax.axvline(0, color=INK_2, lw=1)
    ax.set_yticks(y, chans)
    ax.set_ylim(y.min() - 0.6, y.max() + 0.7)
    ax.set_xlabel("Change vs last 13 weeks' spend (%)", fontsize=9, color=INK_2)
    ax.set_title("Recommended change in spend per channel, by budget (12M breaks the rule: see text)",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", bbox_to_anchor=(1.0, -0.13), ncols=3)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_stress(stress: pd.DataFrame, chans: list[str], out: Path) -> None:
    groups = list(stress["saturation_group"].unique())
    colours = [ORANGE, TEAL, BLUE]
    fig, ax = plt.subplots(figsize=(10, 4))
    y = np.arange(len(chans))[::-1]
    h = 0.8 / len(groups)
    for i, (g, c) in enumerate(zip(groups, colours)):
        d = stress[stress["saturation_group"] == g].set_index("channel").loc[chans]
        ax.barh(y + 0.4 - h * (i + 0.5), d["change_vs_last_13_weeks"] * 100, height=h * 0.9,
                color=c, label=g)
    for x in (-30, 30):
        ax.axvline(x, color=INK_2, lw=1, ls=(0, (4, 3)))
    ax.axvline(0, color=INK_2, lw=1)
    ax.set_yticks(y, chans)
    ax.set_xlabel("Change vs last 13 weeks' spend (%), 18M budget", fontsize=9, color=INK_2)
    ax.set_title("Saturation stress test: does the 18M split change if returns fall faster or slower?",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1.0, -0.13), ncols=3)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    from meridian.analysis import analyzer as an

    cfg = load_config()
    b = cfg["budget"]
    out_dir = Path(cfg["paths"]["output_dir"])
    mmm, which = load_final_model(cfg)
    a = an.Analyzer(model_context=mmm.model_context, inference_data=mmm.inference_data)
    current, chans, window = current_spend(mmm, b["weeks"])
    down = up = b["max_change_vs_last_13w"]
    total_now = current.sum()
    print(f"Model: {which}. Window for 'next quarter': {window[0]} to {window[1]}")
    print(f"Last 13 weeks' spend: {total_now / 1e6:.2f}M. Rule allows "
          f"{(1 - down) * total_now / 1e6:.2f}M to {(1 + up) * total_now / 1e6:.2f}M in total.")

    # --- plans -------------------------------------------------------------
    plans = {"last 13 weeks (actual)": (total_now, current.copy(), "reference")}
    for budget in b["scenarios"]:
        tag = f"{budget / 1e6:g}M"
        same_cut = current * budget / total_now
        bounds = meridian_bounds(budget, current, down, up)
        if bounds is None:
            plans[f"{tag}: same % for every channel"] = (budget, same_cut, "breaks the +/-30% rule")
            relaxed = meridian_bounds(budget, current, 0.70, up)
            print(f"{tag}: impossible within +/-30% (needs at least {(1 - down) * total_now / 1e6:.2f}M);"
                  " showing the same-% cut and the best split with cuts allowed down to -70%")
            plans[f"{tag}: best split, cuts up to -70%"] = (
                budget, optimise(mmm, a, budget, relaxed, window), "rule relaxed to -70% / +30%")
        else:
            plans[f"{tag}: same % for every channel"] = (budget, same_cut, "within rule" if
                                                        abs(budget / total_now - 1) <= down else "breaks rule")
            print(f"Optimising {tag} ...")
            plans[f"{tag}: recommended"] = (budget, optimise(mmm, a, budget, bounds, window), "within rule")

    # --- extra revenue for every plan, all draws ---------------------------
    draws = {p: plan_draws(mmm, a, current, spend, window) for p, (_, spend, _) in plans.items()}
    alloc_rows, summary_rows = [], []
    for p, (budget, spend, rule) in plans.items():
        d = draws[p]
        tot = d.sum(axis=1)
        same = [q for q in plans if q.startswith(p.split(":")[0] + ":") and "same %" in q]
        base = draws[same[0]].sum(axis=1) if same and same[0] != p else None
        summary_rows.append({
            "plan": p, "budget": round(budget), "rule": rule,
            "extra_revenue_mean": tot.mean(), "extra_revenue_p05": np.quantile(tot, .05),
            "extra_revenue_p95": np.quantile(tot, .95), "roi_mean": (tot / budget).mean(),
            "gain_vs_same_cut_mean": None if base is None else (tot - base).mean(),
            "gain_vs_same_cut_p05": None if base is None else np.quantile(tot - base, .05),
            "gain_vs_same_cut_p95": None if base is None else np.quantile(tot - base, .95),
            "prob_beats_same_cut": None if base is None else float((tot > base).mean()),
        })
        for i, ch in enumerate(chans):
            alloc_rows.append({
                "plan": p, "budget": round(budget), "channel": ch,
                "spend_last_13_weeks": round(current[i]), "recommended_spend": round(spend[i]),
                "change_vs_last_13_weeks": spend[i] / current[i] - 1,
                "within_30pct_rule": bool(abs(spend[i] / current[i] - 1) <= down + 1e-9),
                "extra_revenue_mean": d[:, i].mean(), "extra_revenue_p05": np.quantile(d[:, i], .05),
                "extra_revenue_p95": np.quantile(d[:, i], .95),
                "roi_mean": (d[:, i] / spend[i]).mean(),
            })
    alloc = pd.DataFrame(alloc_rows)
    summary = pd.DataFrame(summary_rows)

    # Hard check: every plan that claims to follow the rule really does
    claimed = alloc["plan"].map({p: r for p, (_, _, r) in plans.items()}) == "within rule"
    broken = alloc[claimed & ~alloc["within_30pct_rule"]]
    if len(broken):
        raise SystemExit(f"Rule broken in a 'within rule' plan:\n{broken}")

    # What each extra rupee of BUDGET buys, moving up from one budget to the next
    # (best plan at each level, plus the current 20.77M as it was spent)
    best = summary[summary["plan"].str.contains("recommended|best split|actual")].copy()
    best = best.sort_values("budget").reset_index(drop=True)
    tot = {r.plan: draws[r.plan].sum(axis=1) for r in best.itertuples()}
    steps = []
    for lo, hi in zip(best.itertuples(), best.iloc[1:].itertuples()):
        extra = (tot[hi.plan] - tot[lo.plan]) / (hi.budget - lo.budget)
        steps.append({"from_plan": lo.plan, "to_plan": hi.plan,
                      "extra_budget": hi.budget - lo.budget,
                      "extra_revenue_mean": (tot[hi.plan] - tot[lo.plan]).mean(),
                      "revenue_per_extra_1_mean": extra.mean(),
                      "revenue_per_extra_1_p05": np.quantile(extra, .05),
                      "revenue_per_extra_1_p95": np.quantile(extra, .95),
                      "prob_extra_budget_pays_back": float((extra > 1).mean())})
    budget_steps = pd.DataFrame(steps)

    # --- saturation stress test at the main budget ------------------------
    main_budget = b["next_quarter_total"]
    bounds = meridian_bounds(main_budget, current, down, up)
    stress_rows = []
    for label, (idata, ec_med) in saturation_subsets(mmm).items():
        print(f"Stress test ({label}) ...")
        sub_a = an.Analyzer(model_context=mmm.model_context, inference_data=idata)
        spend = optimise(mmm, sub_a, main_budget, bounds, window)
        for i, ch in enumerate(chans):
            stress_rows.append({"saturation_group": label, "median_half_saturation_point": round(ec_med, 2),
                                "channel": ch, "recommended_spend": round(spend[i]),
                                "change_vs_last_13_weeks": spend[i] / current[i] - 1})
    stress = pd.DataFrame(stress_rows)

    # --- save --------------------------------------------------------------
    for t in (alloc, summary, stress, budget_steps):
        num = t.select_dtypes("number").columns
        t[num] = t[num].round(3)
    alloc.to_csv(REPO_ROOT / "project_b" / "budget_allocation.csv", index=False)
    summary.to_csv(out_dir / "step7_scenario_summary.csv", index=False)
    stress.to_csv(out_dir / "step7_saturation_stress.csv", index=False)
    budget_steps.to_csv(out_dir / "step7_budget_steps.csv", index=False)
    chart_allocation(alloc, chans, out_dir / "step7_allocation.png")
    chart_stress(stress, chans, out_dir / "step7_saturation_stress.png")

    with pd.option_context("display.width", 230, "display.max_columns", 30):
        print("\n--- Plans: total extra revenue in the 13 weeks ---")
        print(summary.to_string(index=False))
        print("\n--- Spend per channel ---")
        print(alloc[["plan", "channel", "spend_last_13_weeks", "recommended_spend",
                     "change_vs_last_13_weeks", "within_30pct_rule", "extra_revenue_mean",
                     "roi_mean"]].to_string(index=False))
        print("\n--- What each extra rupee of budget buys ---")
        print(budget_steps.to_string(index=False))
        print("\n--- Saturation stress test (18M) ---")
        print(stress.to_string(index=False))
    print("\nSaved: project_b/budget_allocation.csv, tables and charts in outputs/project_b")


if __name__ == "__main__":
    main()
