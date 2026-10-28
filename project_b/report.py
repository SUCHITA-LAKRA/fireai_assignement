"""
Project B - Step 6: Report effectiveness: ROI, marginal ROI and the 13-week forecast.

What this script does
---------------------
Uses the calibrated model from step 5 (falls back to step 3's if absent).

1. AVERAGE ROI per channel, all weeks:
       extra revenue the channel created / what was spent on it
   "Was this channel worth it, on average, over 3 years?"
2. MARGINAL ROI per channel at CURRENT spend (the last 13 weeks):
       extra revenue from spending 1% more / that extra 1% of spend
   "Is the NEXT rupee worth it?" Budget moves depend on this, not on the
   average. With diminishing returns, marginal ROI is below average ROI.
3. For both: mean, 90% range, chance it is above 1.
4. How much of the marginal ROI comes from the data rather than the priors
   (step 4 found the saturation point is not learned).
5. Each channel's share of revenue, and the share from all paid media.
6. FORECAST on the 13 held-out weeks: predicted vs actual weekly revenue
   with 90% bands, accuracy, and a comparison with step 2's naive forecasts.

How to run (from the repo root, after step 5)
---------------------------------------------
    python -m project_b.report

Outputs
-------
    project_b/roi_table.csv                    ROI and marginal ROI per channel (deliverable)
    project_b/forecast_13w.csv                 hold-out forecast, week by week (deliverable)
    outputs/project_b/step6_roi_mroi.png
    outputs/project_b/step6_forecast_accuracy.csv   model vs naive benchmarks
    outputs/project_b/step6_forecast_13w.png
    outputs/project_b/step6_mroi_learned.csv   how much the data narrowed marginal ROI vs its prior
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

from project_b import fit_model as fm  # noqa: E402
from project_b import model_data as md  # noqa: E402
from project_b.config import REPO_ROOT, load_config  # noqa: E402

TEAL, BLUE, ORANGE = "#1E6F6C", "#2a78d6", "#eb6834"
GREY, INK, INK_2 = "#8c8b86", "#0b0b0b", "#52514e"


def load_final_model(cfg: dict):
    """The calibrated model (step 5) if it exists, else the step 3 model."""
    try:
        return fm.load_model(cfg, "mmm_calibrated"), "calibrated (step 5)"
    except FileNotFoundError:
        print("NOTE: calibrated model not found - using the step 3 model")
        return fm.load_model(cfg, "mmm_geo"), "uncalibrated (step 3)"


def last_weeks(mmm, n: int) -> list[str]:
    """The last n weeks of data: 'current spend' for marginal ROI and the budget."""
    return [str(t) for t in mmm.input_data.time.values[-n:]]


# ---------------------------------------------------------------------------
# 1. ROI and marginal ROI
# ---------------------------------------------------------------------------

def draws_per_channel(x, channels: list[str]) -> pd.DataFrame:
    x = np.asarray(x)
    return pd.DataFrame(x.reshape(-1, len(channels)), columns=channels)


def roi_table(mmm, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    from meridian.analysis import analyzer

    a = analyzer.Analyzer(mmm)
    chans = [str(c) for c in mmm.input_data.media_channel.values]
    current = last_weeks(mmm, cfg["budget"]["weeks"])

    roi = draws_per_channel(a.roi(), chans)
    mroi = draws_per_channel(a.marginal_roi(selected_times=current), chans)
    mroi_prior = draws_per_channel(a.marginal_roi(use_posterior=False, selected_times=current), chans)
    inc = draws_per_channel(a.incremental_outcome(include_non_paid_channels=False), chans)

    d = mmm.input_data
    revenue = float((np.asarray(d.kpi) * np.asarray(d.revenue_per_kpi)).sum())
    spend_all = np.asarray(d.media_spend).sum(axis=(0, 1))
    t_idx = len(d.time) - cfg["budget"]["weeks"]
    spend_now = np.asarray(d.media_spend)[:, t_idx:, :].sum(axis=(0, 1))

    def q(x, p):
        return float(np.quantile(x, p))

    rows, learned = [], []
    for i, ch in enumerate(chans):
        r, m = roi[ch].to_numpy(), mroi[ch].to_numpy()
        share = inc[ch].to_numpy() / revenue
        rows.append({
            "channel": ch,
            "spend_all_weeks": round(float(spend_all[i])),
            "spend_last_13_weeks": round(float(spend_now[i])),
            "roi_mean": r.mean(), "roi_p05": q(r, .05), "roi_p95": q(r, .95),
            "roi_prob_above_1": float((r > 1).mean()),
            "mroi_mean": m.mean(), "mroi_p05": q(m, .05), "mroi_p95": q(m, .95),
            "mroi_prob_above_1": float((m > 1).mean()),
            "share_of_revenue_mean": share.mean(),
            "share_of_revenue_p05": q(share, .05), "share_of_revenue_p95": q(share, .95),
        })
        pw = q(m, .95) - q(m, .05)
        prw = q(mroi_prior[ch], .95) - q(mroi_prior[ch], .05)
        learned.append({"channel": ch, "mroi_prior_p05": q(mroi_prior[ch], .05),
                        "mroi_prior_p95": q(mroi_prior[ch], .95),
                        "mroi_posterior_p05": q(m, .05), "mroi_posterior_p95": q(m, .95),
                        "learned_from_data": 1 - pw / prw})
    # All paid media together
    tot_inc = inc.sum(axis=1).to_numpy()
    tot_r = tot_inc / spend_all.sum()
    tot_share = tot_inc / revenue
    rows.append({"channel": "All paid media", "spend_all_weeks": round(float(spend_all.sum())),
                 "spend_last_13_weeks": round(float(spend_now.sum())),
                 "roi_mean": tot_r.mean(), "roi_p05": q(tot_r, .05), "roi_p95": q(tot_r, .95),
                 "roi_prob_above_1": float((tot_r > 1).mean()),
                 "share_of_revenue_mean": tot_share.mean(), "share_of_revenue_p05": q(tot_share, .05),
                 "share_of_revenue_p95": q(tot_share, .95)})
    out = pd.DataFrame(rows)
    num = out.select_dtypes("number").columns.drop(["spend_all_weeks", "spend_last_13_weeks"])
    out[num] = out[num].round(3)
    return out, pd.DataFrame(learned).round(3)


# ---------------------------------------------------------------------------
# 2. Forecast on the hold-out weeks
# ---------------------------------------------------------------------------

def forecast(mmm, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    _, lines = fm.posterior_predictive(mmm, seed=cfg["model"]["sampling"]["seed"])
    h = lines["hold"]
    fc = pd.DataFrame({"week": lines["time"][h].strftime("%Y-%m-%d"),
                       "actual_revenue": lines["actual"][h],
                       "predicted_revenue": lines["mean"][h],
                       "predicted_p05": lines["lo"][h], "predicted_p95": lines["hi"][h]})
    fc["error_pct"] = (fc["predicted_revenue"] / fc["actual_revenue"] - 1)
    fc["actual_inside_90pct_band"] = ((fc["actual_revenue"] >= fc["predicted_p05"]) &
                                      (fc["actual_revenue"] <= fc["predicted_p95"]))

    acc = fm.fit_accuracy(mmm)
    test = acc[acc["evaluation_set"] == "Test"]

    def get(metric, gran):
        return float(test[(test["metric"] == metric) & (test["geo_granularity"] == gran)]["value"].iloc[0])

    rows = [{"forecast": "Model (calibrated MMM)",
             "region_week_wMAPE": get("wMAPE", "geo"),
             "national_weekly_MAPE": get("MAPE", "national"),
             "national_R2": get("R_Squared", "national"),
             "national_total_bias": float(fc["predicted_revenue"].sum() / fc["actual_revenue"].sum() - 1),
             "weeks_inside_90pct_band": f"{int(fc['actual_inside_90pct_band'].sum())} of {len(fc)}"}]
    bench = Path(cfg["paths"]["output_dir"]) / "step2_naive_benchmarks.csv"
    if bench.exists():
        names = {"last_13_weeks": "Naive: last 13 weeks' average",
                 "same_week_last_year": "Naive: same week last year",
                 "training_average": "Naive: training average"}
        for _, r in pd.read_csv(bench).iterrows():
            rows.append({"forecast": names.get(r["benchmark"], r["benchmark"]),
                         "region_week_wMAPE": r["region_week_wMAPE"],
                         "national_weekly_MAPE": r["national_weekly_MAPE"],
                         "national_R2": r["national_R2"], "national_total_bias": r["national_total_bias"],
                         "weeks_inside_90pct_band": "-"})
    accuracy = pd.DataFrame(rows)
    num = accuracy.select_dtypes("number").columns
    accuracy[num] = accuracy[num].round(4)
    fc = fc.round({"actual_revenue": 0, "predicted_revenue": 0, "predicted_p05": 0,
                   "predicted_p95": 0, "error_pct": 4})
    return fc, accuracy, lines


# ---------------------------------------------------------------------------
# 3. Charts
# ---------------------------------------------------------------------------

def _style(ax, grid="x"):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis=grid, color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)


def chart_roi_mroi(t: pd.DataFrame, out: Path) -> None:
    t = t[t["channel"] != "All paid media"].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    y = np.arange(len(t))[::-1]
    for ax, key, title, c in [(axes[0], "roi", "Average ROI, all weeks", TEAL),
                              (axes[1], "mroi", "Marginal ROI at current spend", BLUE)]:
        ax.hlines(y, t[f"{key}_p05"], t[f"{key}_p95"], color=c, lw=6, alpha=0.9)
        ax.plot(t[f"{key}_mean"], y, "o", color=c, ms=8, mec="white", mew=1.5)
        for yi, m, hi in zip(y, t[f"{key}_mean"], t[f"{key}_p95"]):
            ax.text(hi + 0.06, yi, f"{m:.2f}", va="center", fontsize=9, color=INK)
        ax.axvline(1, color=INK_2, lw=1, ls=(0, (4, 3)))
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlim(0, max(t["roi_p95"].max(), t["mroi_p95"].max()) + 0.45)
        ax.set_xlabel("Revenue per 1 spent (dot = average, bar = 90% range)", fontsize=8.5, color=INK_2)
        _style(ax)
    axes[0].set_yticks(y, t["channel"])
    for ax in axes:
        ax.set_ylim(y.min() - 0.5, y.max() + 0.7)
        ax.text(1.03, y.max() + 0.45, "break-even (1)", fontsize=8.5, color=INK_2, va="center")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_forecast(lines: dict, cfg: dict, out: Path) -> None:
    t, h = lines["time"], lines["hold"]
    idx = np.where(h)[0]
    lo_i = max(0, idx[0] - 26)
    hi_i = min(len(t), idx[-1] + 1)
    sl = slice(lo_i, hi_i)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axvspan(t[idx[0]] - pd.Timedelta(days=3), t[idx[-1]] + pd.Timedelta(days=3),
               color=GREY, alpha=0.15, lw=0)
    ax.text(t[idx[0]], 0.02, " held-out weeks (model never saw this revenue)",
            transform=ax.get_xaxis_transform(), color=INK_2, fontsize=9)
    ax.fill_between(t[sl], lines["lo"][sl] / 1e6, lines["hi"][sl] / 1e6, color=BLUE, alpha=0.2,
                    lw=0, label="Model: 90% band")
    ax.plot(t[sl], lines["mean"][sl] / 1e6, color=BLUE, lw=2, label="Model: forecast")
    ax.plot(t[sl], lines["actual"][sl] / 1e6, color=INK, lw=1.6, label="Actual")
    ax.set_title("Forecast vs actual weekly revenue, all regions (millions)", loc="left",
                 fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncols=3)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.15)
    _style(ax, "y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config()
    out_dir = Path(cfg["paths"]["output_dir"])
    mmm, which = load_final_model(cfg)
    print(f"Model: {which}")

    roi, learned = roi_table(mmm, cfg)
    fc, accuracy, lines = forecast(mmm, cfg)

    roi.to_csv(REPO_ROOT / "project_b" / "roi_table.csv", index=False)
    fc.to_csv(REPO_ROOT / "project_b" / "forecast_13w.csv", index=False)
    accuracy.to_csv(out_dir / "step6_forecast_accuracy.csv", index=False)
    learned.to_csv(out_dir / "step6_mroi_learned.csv", index=False)
    chart_roi_mroi(roi, out_dir / "step6_roi_mroi.png")
    chart_forecast(lines, cfg, out_dir / "step6_forecast_13w.png")

    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print("\n--- ROI and marginal ROI (revenue per 1 spent) ---")
        print(roi.to_string(index=False))
        print("\n--- Marginal ROI: how much came from the data (0 = all prior) ---")
        print(learned.to_string(index=False))
        print("\n--- Forecast on the held-out weeks ---")
        print(fc.to_string(index=False))
        print("\n--- Forecast accuracy: model vs naive ---")
        print(accuracy.to_string(index=False))
    print("\nSaved: project_b/roi_table.csv, project_b/forecast_13w.csv, charts in outputs/project_b")


if __name__ == "__main__":
    main()
