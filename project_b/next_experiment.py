"""
Project B - Step 8: Design the next experiment.

The question: which channel's uncertainty costs the most money, and what
test (regions, length) would reduce it, with a power calculation?

What this script does
---------------------
Uses the calibrated model (step 5). Money = extra revenue minus spend over
the 13 weeks. Channels add up independently in the model, so each can be
scored on its own.

Part 1 - What our uncertainty costs, per channel (value of PERFECT information)
   We set each channel's budget (within +/-30%) from the AVERAGE belief. If we
   knew the true return, we could set it better. The average gain over the
   2,000 posterior draws is what not knowing costs per quarter.

Part 2 - How noisy would a real test be?  (power from our own history)
   Design: switch the channel OFF in a random, size-balanced set of regions;
   the other regions are the comparison.
   Analysis: time-based regression. Learn, over the 52 weeks before the test,
   how treated-region revenue tracks control-region revenue; predict what the
   treated regions would have earned; lift = actual - predicted;
   ROI = lift / spend switched off. It doesn't use the MMM, so it can check it.
   Noise: run thousands of FAKE tests on past weeks where nothing was switched
   off. The spread of their "lift" is what a real test must beat.
       standard error of ROI = noise / spend switched off

Part 3 - What would THIS test be worth?  (value of SAMPLE information)
   For each channel and design, simulate the test result from every posterior
   draw, update our belief, re-decide the budget, and measure the average
   gain. Multiply by how many quarters the result stays useful, and compare
   with the expected cost of the test (revenue lost minus spend saved).

Choosing the test
   1. Report the channel whose uncertainty costs most (part 1) and the best
      test available for it.
   2. Recommend the CHEAPEST design (any channel) that reaches 80% power to
      detect a true ROI of 1 at 90% confidence; if none does, the most
      powerful one. A test that can't detect a channel paying for itself
      can't settle the budget question.

How to run (from the repo root, after step 6)
---------------------------------------------
    python -m project_b.next_experiment

Outputs
-------
    outputs/project_b/step8_value_of_information.csv   per channel: cost of uncertainty
    outputs/project_b/step8_test_noise.csv             noise of a test, by regions and weeks
    outputs/project_b/step8_designs.csv                every channel x design: precision, power, value, cost
    outputs/project_b/step8_recommended_design.csv     the chosen test, regions and settings
    outputs/project_b/step8_value_of_information.png
    outputs/project_b/step8_designs.png               power curves: costliest-uncertainty channel and recommended channel
"""

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_b import model_data as md  # noqa: E402
from project_b.config import load_config  # noqa: E402
from project_b.optimise import current_spend, plan_draws  # noqa: E402
from project_b.report import load_final_model  # noqa: E402

TEAL, BLUE, ORANGE = "#1E6F6C", "#2a78d6", "#eb6834"
GREY, INK, INK_2 = "#8c8b86", "#0b0b0b", "#52514e"
GRID = np.round(np.arange(0.70, 1.301, 0.05), 2)   # spend multipliers allowed by the rule
Z90, Z80 = stats.norm.ppf(0.95), stats.norm.ppf(0.80)


# ---------------------------------------------------------------------------
# Part 1: value of each budget choice, per draw
# ---------------------------------------------------------------------------

def value_matrix(mmm, cfg: dict):
    """value[draw, grid point, channel] = extra revenue - spend at that spend level."""
    from meridian.analysis import analyzer as an

    a = an.Analyzer(model_context=mmm.model_context, inference_data=mmm.inference_data)
    current, chans, window = current_spend(mmm, cfg["budget"]["weeks"])
    # Channels are additive, so scaling all channels by k at once gives each
    # channel's own response at k: one model call per grid point.
    revenue = np.stack([plan_draws(mmm, a, current, current * k, window) for k in GRID], axis=1)
    value = revenue - GRID[None, :, None] * current[None, None, :]
    roi = np.asarray(a.roi()).reshape(-1, len(chans))   # average ROI per draw (what a switch-off measures)
    return value, roi, current, chans


def perfect_information(value: np.ndarray, chans: list[str], current: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, ch in enumerate(chans):
        v = value[:, :, i]
        k0 = int(np.argmax(v.mean(axis=0)))
        evpi = v.max(axis=1).mean() - v[:, k0].mean()
        best = GRID[np.argmax(v, axis=1)]
        rows.append({"channel": ch, "spend_last_13_weeks": round(float(current[i])),
                     "choice_from_average_belief": f"{GRID[k0] - 1:+.0%}",
                     "share_of_draws_wanting_a_cut": float((best < 1).mean()),
                     "share_of_draws_wanting_an_increase": float((best > 1).mean()),
                     "value_of_perfect_information_per_quarter": evpi,
                     "as_pct_of_channel_spend": evpi / current[i]})
    return pd.DataFrame(rows).sort_values("value_of_perfect_information_per_quarter",
                                          ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Part 2: test noise from fake experiments on history
# ---------------------------------------------------------------------------

def size_balanced_assignment(pop: pd.Series, n_treated: int, rng) -> list[str]:
    """Sort regions by population into n_treated equal blocks; pick one per block at random."""
    order = pop.sort_values().index.to_list()
    size = len(order) // n_treated
    return [str(rng.choice(order[b * size:(b + 1) * size])) for b in range(n_treated)]


def tbr_lift(y_treated: np.ndarray, y_control: np.ndarray, start: int, weeks: int, pre: int) -> float:
    """Time-based regression: fit treated ~ a + b x control on the pre-period, sum actual - predicted."""
    X = np.c_[np.ones(pre), y_control[start - pre:start]]
    a, b = np.linalg.lstsq(X, y_treated[start - pre:start], rcond=None)[0]
    during = slice(start, start + weeks)
    return float((y_treated[during] - (a + b * y_control[during])).sum())


def test_noise(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Spread of 'lift' in fake tests where nothing was switched off (training weeks only)."""
    e = cfg["next_experiment"]
    rng = np.random.default_rng(cfg["random_seed"])
    d = df[~df["is_holdout"]]
    rev = (d["conversions"] * d["revenue_per_conversion"]).groupby([d[md.GEO], d[md.TIME]]).sum().unstack()
    pop = d.groupby(md.GEO)["population"].first()
    n_weeks, pre = rev.shape[1], e["pre_period_weeks"]
    rows = []
    for g in e["treated_region_options"]:
        for w in e["week_options"]:
            lifts = []
            for _ in range(e["n_fake_experiments"]):
                is_t = rev.index.isin(size_balanced_assignment(pop, g, rng))
                s = int(rng.integers(pre, n_weeks - w + 1))
                lifts.append(tbr_lift(rev[is_t].sum(0).to_numpy(), rev[~is_t].sum(0).to_numpy(), s, w, pre))
            rows.append({"treated_regions": g, "weeks": w, "noise_sd_of_lift": float(np.std(lifts)),
                         "bias_of_lift": float(np.mean(lifts))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 3: value of THIS test (sample information), per channel and design
# ---------------------------------------------------------------------------

def sample_information(v: np.ndarray, roi: np.ndarray, se: float, rng, n_sims: int = 5000) -> float:
    """Average gain in decision value from seeing a test result with error `se`.

    For a 'true' draw t, the test reports roi[t] + noise. We re-weight all
    draws by how well they explain that result (Bayes), pick the best spend
    level under the updated belief, and score it with draw t's true values.
    """
    k0 = int(np.argmax(v.mean(axis=0)))
    truth = rng.integers(0, len(roi), n_sims)
    seen = roi[truth] + rng.normal(0, se, n_sims)
    w = np.exp(-0.5 * ((seen[:, None] - roi[None, :]) / se) ** 2)        # (sims, draws)
    w /= w.sum(axis=1, keepdims=True)
    choice = np.argmax(w @ v, axis=1)                                    # best spend after the test
    # Information can't hurt in expectation; small negatives are simulation noise
    return max(0.0, float(v[truth, choice].mean() - v[truth, k0].mean()))


def designs(value, roi, current, chans, noise, df, cfg) -> pd.DataFrame:
    e = cfg["next_experiment"]
    rng = np.random.default_rng(cfg["random_seed"] + 7)
    d = df[~df["is_holdout"]]
    pop = d.groupby(md.GEO)["population"].first()
    rows = []
    for i, ch in enumerate(chans):
        weekly = d.groupby(md.GEO)[f"{ch}_spend"].mean()        # typical weekly spend per region
        for n in noise.itertuples():
            share = np.mean([weekly[size_balanced_assignment(pop, n.treated_regions, rng)].sum()
                             for _ in range(300)])
            off = share * n.weeks
            se = n.noise_sd_of_lift / off
            evsi = sample_information(value[:, :, i], roi[:, i], se, rng)
            cost = (roi[:, i].mean() - 1) * off      # revenue lost minus spend saved
            rows.append({
                "channel": ch, "treated_regions": n.treated_regions, "weeks": n.weeks,
                "spend_switched_off": off, "roi_standard_error": se,
                "roi_90pct_margin": Z90 * se,
                "smallest_roi_detectable_80pct_power": (Z90 + Z80) * se,
                "power_if_true_roi_is_1": float(stats.norm.cdf(1 / se - Z90)),
                "current_roi_sd": float(roi[:, i].std()),
                "roi_range_shrinks_by": 1 - 1 / np.sqrt(1 + roi[:, i].var() / se ** 2),
                "value_per_quarter": evsi,
                "value_over_horizon": evsi * e["info_horizon_quarters"],
                "expected_test_cost": cost,
                "net_value": evsi * e["info_horizon_quarters"] - cost,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _style(ax, grid="x"):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis=grid, color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)


def chart_voi(voi: pd.DataFrame, best: pd.DataFrame, out: Path) -> None:
    """Per channel: cost of uncertainty (perfect info) vs what the best test recovers."""
    v = voi.set_index("channel")
    b = best.set_index("channel").loc[v.index]
    y = np.arange(len(v))[::-1]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.barh(y + 0.18, v["value_of_perfect_information_per_quarter"] / 1e3, height=0.34, color=GREY,
            label="What not knowing costs (perfect information)")
    ax.barh(y - 0.18, b["value_per_quarter"] / 1e3, height=0.34, color=TEAL,
            label="What the best realistic test recovers")
    for yi, a1, a2 in zip(y, v["value_of_perfect_information_per_quarter"] / 1e3, b["value_per_quarter"] / 1e3):
        ax.text(a1, yi + 0.18, f"  {a1:,.0f}k", va="center", fontsize=8.5, color=INK)
        ax.text(a2, yi - 0.18, f"  {a2:,.0f}k", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, v.index)
    ax.set_xlim(0, v["value_of_perfect_information_per_quarter"].max() / 1e3 * 1.22)
    ax.set_xlabel("Per quarter, thousands (extra revenue minus spend)", fontsize=9, color=INK_2)
    ax.set_title("Which channel's uncertainty costs the most, and how much a test can recover",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", bbox_to_anchor=(1.0, -0.17), ncols=2)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_designs(des: pd.DataFrame, channels: list[str], chosen: pd.Series, out: Path) -> None:
    """Power by test length, one panel per channel, one line per number of regions."""
    fig, axes = plt.subplots(1, len(channels), figsize=(10, 3.8), sharey=True)
    colours = {8: ORANGE, 10: TEAL, 20: BLUE}
    for ax, ch in zip(np.atleast_1d(axes), channels):
        d = des[des["channel"] == ch]
        for g, x in d.groupby("treated_regions"):
            ax.plot(x["weeks"], x["power_if_true_roi_is_1"] * 100, "o-", color=colours.get(g, GREY),
                    lw=2, ms=5, label=f"{g} regions off")
        ax.axhline(80, color=INK_2, lw=1, ls=(0, (4, 3)))
        ax.text(d["weeks"].min(), 82, "80% target", fontsize=8.5, color=INK_2)
        if ch == chosen["channel"]:
            ax.plot(chosen["weeks"], chosen["power_if_true_roi_is_1"] * 100, "o", ms=14, mfc="none",
                    mec=INK, mew=1.6)
        ax.set_title(f"{ch}", loc="left", fontsize=10.5, color=INK)
        ax.set_xticks(sorted(d["weeks"].unique()))
        ax.set_xlabel("Weeks switched off", fontsize=9, color=INK_2)
        ax.set_ylim(0, 100)
        _style(ax, "y")
    np.atleast_1d(axes)[0].set_ylabel("Power to detect ROI = 1 (%)", fontsize=9, color=INK_2)
    np.atleast_1d(axes)[-1].legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle("How likely a switch-off test detects a channel that pays for itself (90% confidence)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config()
    out_dir = Path(cfg["paths"]["output_dir"])
    df = md.load_clean(cfg)
    mmm, which = load_final_model(cfg)
    print(f"Model: {which}")

    print("Part 1: value of each budget choice per draw ...")
    value, roi, current, chans = value_matrix(mmm, cfg)
    voi = perfect_information(value, chans, current)

    print("Part 2: fake experiments on past weeks ...")
    noise = test_noise(df, cfg)

    print("Part 3: value of each realistic test ...")
    des = designs(value, roi, current, chans, noise, df, cfg)
    # Best test per channel = the most powerful one we allow
    best_per_channel = des.loc[des.groupby("channel")["power_if_true_roi_is_1"].idxmax()]
    top = voi.iloc[0]["channel"]                                    # costliest uncertainty
    top_test = best_per_channel.set_index("channel").loc[top]
    powered = des[des["power_if_true_roi_is_1"] >= 0.80]
    chosen = (powered.sort_values("expected_test_cost").iloc[0] if len(powered)
              else des.sort_values("power_if_true_roi_is_1").iloc[-1])

    rng = np.random.default_rng(cfg["random_seed"] + 1)
    pop = df.groupby(md.GEO)["population"].first()
    regions = sorted(size_balanced_assignment(pop, int(chosen["treated_regions"]), rng),
                     key=lambda g: int(g.replace("Geo", "")))
    start = pd.Timestamp(df[md.TIME].max()) + pd.Timedelta(weeks=1)
    i = chans.index(chosen["channel"])
    design = pd.DataFrame([
        ("channel whose uncertainty costs most", f"{top}: about {voi.iloc[0]['value_of_perfect_information_per_quarter'] / 1e3:,.0f}k per quarter"),
        (f"best test for {top}", f"{int(top_test['treated_regions'])} regions x {int(top_test['weeks'])} weeks: "
                                 f"ROI to +/- {top_test['roi_90pct_margin']:.2f}, power {top_test['power_if_true_roi_is_1']:.0%}, "
                                 f"expected cost {top_test['expected_test_cost'] / 1e3:,.0f}k vs information worth "
                                 f"{top_test['value_over_horizon'] / 1e3:,.0f}k over a year -> not worth running"),
        ("recommended test: channel", chosen["channel"]),
        ("why this channel", "the cheapest test that reaches 80% power to detect a true ROI of 1; "
                             "its large spend gives the strongest signal, and it re-runs the vendor test properly"),
        ("design", "switch the channel OFF in the treated regions; every other region and channel "
                   "carries on as normal"),
        ("treated regions", ", ".join(regions)),
        ("how regions are chosen", "sort the 40 regions by population into equal blocks; pick one per "
                                   "block at random (re-draw with a new seed for the real test)"),
        ("control regions", f"the other {40 - int(chosen['treated_regions'])}"),
        ("weeks switched off", int(chosen["weeks"])),
        ("extra weeks measured after", 2),
        ("why the extra weeks", "ads keep working for a few weeks after they stop (adstock)"),
        ("analysis", f"time-based regression on the {cfg['next_experiment']['pre_period_weeks']} weeks "
                     "before the test; ROI = lift / spend switched off"),
        ("earliest start", start.strftime("%Y-%m-%d")),
        ("spend switched off", round(float(chosen["spend_switched_off"]))),
        ("ROI measured to within (90%)", f"+/- {chosen['roi_90pct_margin']:.2f}"),
        ("smallest ROI it reliably detects (80% power)", round(float(chosen["smallest_roi_detectable_80pct_power"]), 2)),
        ("power if the true ROI is 1", f"{chosen['power_if_true_roi_is_1']:.0%}"),
        ("current belief about ROI", f"{roi[:, i].mean():.2f} (90%: {np.quantile(roi[:, i], .05):.2f}-"
                                     f"{np.quantile(roi[:, i], .95):.2f})"),
        ("range shrinks by (combined with the model)", f"{chosen['roi_range_shrinks_by']:.0%}"),
        ("value over a year", round(float(chosen["value_over_horizon"]))),
        ("expected cost (revenue lost - spend saved)", round(float(chosen["expected_test_cost"]))),
        ("net value", round(float(chosen["net_value"]))),
        ("success check before trusting it", "spend data must show ZERO spend for this channel in "
                                             "every treated region and test week (the last test failed this)"),
        ("feed the result back", "as an ROI prior for those weeks, exactly as in step 5"),
    ], columns=["item", "value"])

    for t in (voi, noise, des):
        num = t.select_dtypes("number").columns
        t[num] = t[num].round(3)
    voi.to_csv(out_dir / "step8_value_of_information.csv", index=False)
    noise.to_csv(out_dir / "step8_test_noise.csv", index=False)
    des.to_csv(out_dir / "step8_designs.csv", index=False)
    design.to_csv(out_dir / "step8_recommended_design.csv", index=False)
    chart_voi(voi, best_per_channel, out_dir / "step8_value_of_information.png")
    chart_designs(des, list(dict.fromkeys([top, chosen["channel"]])), chosen, out_dir / "step8_designs.png")

    with pd.option_context("display.width", 230, "display.max_columns", 30, "display.max_colwidth", 100):
        print("\n--- Part 1: what our uncertainty costs per quarter ---")
        print(voi.to_string(index=False))
        print("\n--- Part 2: noise of a test (fake tests on past weeks) ---")
        print(noise.to_string(index=False))
        print("\n--- Part 3: most powerful test per channel ---")
        print(best_per_channel[["channel", "treated_regions", "weeks", "spend_switched_off",
                                "roi_90pct_margin", "power_if_true_roi_is_1", "roi_range_shrinks_by",
                                "value_per_quarter", "value_over_horizon", "expected_test_cost",
                                "net_value"]].to_string(index=False))
        print("\n(Value of a test = how much better a year of budget decisions gets. It is small for "
              "every channel because the data already agrees on cutting; see docs/DECISIONS_PROJECT_B_DETAIL.md, B40.)")
        print("\n--- Recommended test ---")
        print(design.to_string(index=False))
    print("\nSaved: tables and charts in outputs/project_b")


if __name__ == "__main__":
    main()
