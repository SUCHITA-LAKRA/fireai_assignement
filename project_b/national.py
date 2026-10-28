"""
Project B - Step 4: The same model on national totals, compared with the geo model.

What this script does
---------------------
1. Adds the 40 regions up into one national series (step 2's national data)
   and fits EXACTLY the same model as step 3: same priors, same knots, same
   hold-out, same sampler settings. Only the level of detail changes.
2. Checks it the same way: convergence, fit accuracy, posterior-predictive.
3. Compares ROI per channel: national vs geo (step 3).
4. Measures WHY they can differ, straight from the data:
     - how many data points each model learns from
     - how much of each channel's spend variation exists only between
       regions (invisible once you add regions up)
     - how strongly channels move together nationally vs by region
5. Says which model to trust and why.

How to run (from the repo root, after step 3)
---------------------------------------------
    python -m project_b.national
    python -m project_b.national --quick     # smoke test, not for results
    python -m project_b.national --reuse     # re-run the comparison without re-fitting

Outputs
-------
    outputs/project_b/step4_geo_vs_national_roi.csv / .png   ROI side by side
    outputs/project_b/step4_why_they_differ.csv              data facts behind the differences
    outputs/project_b/step4_accuracy_comparison.csv          same-weeks accuracy, both models
    outputs/project_b/step4_how_much_learned.csv             how much each model narrowed its priors
    outputs/project_b/step4_national_convergence.csv, _parameters.csv,
      _fit_accuracy.csv, _posterior_predictive.csv / .png    national model checks
"""

import argparse
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
from project_b.config import load_config  # noqa: E402

TEAL, ORANGE, GREY, INK, INK_2 = "#1E6F6C", "#eb6834", "#8c8b86", "#0b0b0b", "#52514e"


# ---------------------------------------------------------------------------
# 1. ROI side by side
# ---------------------------------------------------------------------------

def roi_draws(mmm) -> pd.DataFrame:
    """ROI draws per channel (all weeks), one column per channel."""
    post = mmm.inference_data.posterior["roi_m"]
    chans = [str(c) for c in post.coords["media_channel"].values]
    return pd.DataFrame(post.values.reshape(-1, len(chans)), columns=chans)


def compare_roi(geo, nat) -> pd.DataFrame:
    g, n = roi_draws(geo), roi_draws(nat)
    rows = []
    for ch in g.columns:
        gl, gh = np.quantile(g[ch], [0.05, 0.95])
        nl, nh = np.quantile(n[ch], [0.05, 0.95])
        rows.append({
            "channel": ch,
            "geo_mean": g[ch].mean(), "geo_p05": gl, "geo_p95": gh,
            "national_mean": n[ch].mean(), "national_p05": nl, "national_p95": nh,
            "difference_national_minus_geo": n[ch].mean() - g[ch].mean(),
            "national_range_vs_geo_range": (nh - nl) / (gh - gl),
            "ranges_overlap": bool(max(gl, nl) <= min(gh, nh)),
            "geo_prob_roi_above_1": float((g[ch] > 1).mean()),
            "national_prob_roi_above_1": float((n[ch] > 1).mean()),
        })
    out = pd.DataFrame(rows)
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(3)
    return out


# ---------------------------------------------------------------------------
# 2. Why they differ - facts from the data (training weeks only)
# ---------------------------------------------------------------------------

def time_trend_share(series: np.ndarray, knots: list[int], keep: np.ndarray) -> float:
    """Share of a weekly series' variation that the model's time trend can absorb.

    Builds the same piecewise-straight-line trend as the model (straight lines
    between knots), fits it to the series, and returns R^2. Whatever the trend
    absorbs, the media coefficients can't use.
    """
    n = len(series)
    basis = np.zeros((n, len(knots)))
    for t in range(n):
        if t <= knots[0]:
            basis[t, 0] = 1
        elif t >= knots[-1]:
            basis[t, -1] = 1
        else:
            j = np.searchsorted(knots, t) - 1
            w = (t - knots[j]) / (knots[j + 1] - knots[j])
            basis[t, j], basis[t, j + 1] = 1 - w, w
    y, X = series[keep], basis[keep]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def why_they_differ(df: pd.DataFrame, roles: dict) -> pd.DataFrame:
    train = df[~df["is_holdout"]].copy()
    per_k = train[roles["population"]] / 1000
    rows = [{"fact": "data points the model learns from",
             "geo_model": f"{len(train):,} region-weeks",
             "national_model": f"{train[md.TIME].nunique()} weeks",
             "so_what": "40x fewer observations nationally -> wider, less stable estimates"}]

    # Share of each channel's spend variation that only exists between regions
    for ch, col in zip(roles["channels"], roles["media_spend"]):
        x = train[col] / per_k  # spend per 1k people
        week_mean = x.groupby(train[md.TIME]).transform("mean")
        geo_mean = x.groupby(train[md.GEO]).transform("mean")
        within = x - week_mean - geo_mean + x.mean()   # region-specific weekly variation
        share = float(within.var() / (x - x.mean()).var())
        rows.append({"fact": f"{ch}: share of spend variation that exists only between regions",
                     "geo_model": f"{share:.0%} (used)", "national_model": "0% (added away)",
                     "so_what": "the geo model sees regions spending differently in the same week; "
                                "national totals hide this"})

    # Channels moving together
    nat = train.groupby(md.TIME)[roles["media_spend"]].sum()
    corr_nat = nat.corr().where(~np.eye(len(roles["channels"]), dtype=bool))
    geo_pc = train[roles["media_spend"]].div(per_k, axis=0)
    within_pc = geo_pc - geo_pc.groupby(train[md.TIME]).transform("mean") \
        - geo_pc.groupby(train[md.GEO]).transform("mean") + geo_pc.mean()
    corr_within = within_pc.corr().where(~np.eye(len(roles["channels"]), dtype=bool))
    i, j = np.unravel_index(np.nanargmax(corr_nat.values), corr_nat.shape)
    rows.append({"fact": "highest spend correlation between two channels",
                 "geo_model": f"{np.nanmax(corr_within.values):.2f} (region-specific movement)",
                 "national_model": f"{corr_nat.values[i, j]:.2f} "
                                   f"({roles['channels'][i]} vs {roles['channels'][j]})",
                 "so_what": "channels that rise and fall together are hard to tell apart; "
                            "nationally they move together much more"})
    # Time trend absorbs part of national spend movement (not of geo movement)
    knots = fm.make_knots(df, 13)
    keep = ~df.groupby(md.TIME)["is_holdout"].first().to_numpy()
    nat_all = df.groupby(md.TIME)[roles["media_spend"]].sum()
    for ch, col in zip(roles["channels"], roles["media_spend"]):
        share = time_trend_share(nat_all[col].to_numpy(), knots, keep)
        rows.append({"fact": f"{ch}: share of national spend movement the time trend can absorb",
                     "geo_model": "small (time trend is shared; regions still differ)",
                     "national_model": f"{share:.0%}",
                     "so_what": "what the trend absorbs can't be credited to the channel; "
                                "in the national model that's the only variation there is"})
    nat_rev = (df[roles["kpi"]] * df[roles["revenue_per_kpi"]]).groupby(df[md.TIME]).sum()
    rows.append({"fact": "revenue: share of national movement the time trend can absorb",
                 "geo_model": "-", "national_model": f"{time_trend_share(nat_rev.to_numpy(), knots, keep):.0%}",
                 "so_what": "spend has a seasonal pattern, revenue barely does"})

    # Overlap between our spend and the competitor confounder
    comp = roles["controls"][0]
    total_pc = train[roles["media_spend"]].sum(axis=1) / per_k
    nat_total = train.groupby(md.TIME)[roles["media_spend"]].sum().sum(axis=1)
    nat_comp = (train[comp] * train[roles["population"]]).groupby(train[md.TIME]).sum() / \
        train.groupby(md.TIME)[roles["population"]].sum()
    w_spend = total_pc - total_pc.groupby(train[md.TIME]).transform("mean") \
        - total_pc.groupby(train[md.GEO]).transform("mean") + total_pc.mean()
    w_comp = train[comp] - train[comp].groupby(train[md.TIME]).transform("mean") \
        - train[comp].groupby(train[md.GEO]).transform("mean") + train[comp].mean()
    rows.append({"fact": "correlation of total paid spend with competitor sales (the main confounder)",
                 "geo_model": f"{w_spend.corr(w_comp):.2f} (region-specific movement)",
                 "national_model": f"{nat_total.corr(nat_comp):.2f}",
                 "so_what": "nationally, ads and the confounder move almost in lock-step, so the "
                            "model struggles to split credit between them"})
    rows.append({"fact": "how the media effect is allowed to vary",
                 "geo_model": "each region its own strength (log-normal around a shared average)",
                 "national_model": "one strength (Meridian switches to a normal prior)",
                 "so_what": "saturation is applied to totals; regions at different saturation "
                            "levels get averaged, which bends the curve"})
    return pd.DataFrame(rows)


def how_much_learned(geo, nat) -> pd.DataFrame:
    """How much the data narrowed each parameter's range, compared with its prior.

    learned = 1 - (posterior 90% width / prior 90% width).
    0% = the data taught nothing (posterior = prior); 100% = pinned down exactly.
    """
    rows = []
    for param, meaning in [("roi_m", "ROI"), ("alpha_m", "adstock decay"),
                           ("ec_m", "half-saturation point")]:
        for name, m in [("geo", geo), ("national", nat)]:
            post, prior = m.inference_data.posterior[param], m.inference_data.prior[param]
            for ch in post.coords["media_channel"].values:
                pw = np.subtract(*np.quantile(post.sel(media_channel=ch).values, [0.95, 0.05]))
                rw = np.subtract(*np.quantile(prior.sel(media_channel=ch).values, [0.95, 0.05]))
                rows.append({"parameter": meaning, "channel": str(ch), "model": name,
                             "learned": 1 - pw / rw})
    out = pd.DataFrame(rows).pivot_table(index=["parameter", "channel"], columns="model",
                                         values="learned").reset_index()
    out.columns.name = None
    return out.rename(columns={"geo": "geo_learned", "national": "national_learned"}).round(2)


def accuracy_comparison(geo, nat) -> pd.DataFrame:
    """National-level accuracy of both models on the same weeks."""
    g = fm.fit_accuracy(geo)
    n = fm.fit_accuracy(nat)
    g = g[g["geo_granularity"] == "national"].assign(model="geo model, added up")
    n = n[n["geo_granularity"] == "national"].assign(model="national model")
    out = pd.concat([g, n])
    return out.pivot_table(index=["metric", "evaluation_set"], columns="model",
                           values="value").reset_index().round(4)


# ---------------------------------------------------------------------------
# 3. Chart
# ---------------------------------------------------------------------------

def chart_roi(cmp: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    y = np.arange(len(cmp))[::-1]
    off = 0.17
    ax.hlines(y + off, cmp["geo_p05"], cmp["geo_p95"], color=TEAL, lw=5, label="Geo model (40 regions)")
    ax.plot(cmp["geo_mean"], y + off, "o", color=TEAL, ms=8, mec="white", mew=1.5)
    ax.hlines(y - off, cmp["national_p05"], cmp["national_p95"], color=ORANGE, lw=5,
              label="National model (totals)")
    ax.plot(cmp["national_mean"], y - off, "o", color=ORANGE, ms=8, mec="white", mew=1.5)
    for yi, r in zip(y, cmp.itertuples()):
        ax.text(r.geo_p95 + 0.08, yi + off, f"{r.geo_mean:.2f}", va="center", fontsize=9, color=INK)
        ax.text(r.national_p95 + 0.08, yi - off, f"{r.national_mean:.2f}", va="center",
                fontsize=9, color=INK)
    ax.axvline(1, color=INK_2, lw=1, ls=(0, (4, 3)))
    ax.text(1.03, y.min() - 0.62, "break-even (ROI 1)", color=INK_2, fontsize=8.5, va="center")
    ax.set_yticks(y, cmp["channel"])
    ax.set_ylim(y.min() - 0.85, y.max() + 0.55)
    ax.set_xlim(0, max(cmp["national_p95"].max(), cmp["geo_p95"].max()) + 0.6)
    ax.set_xlabel("Revenue per 1 spent (dot = average, bar = 90% range)", fontsize=9, color=INK_2)
    ax.set_title("ROI by channel: geo model vs national model", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors=INK_2, labelsize=9)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="few draws, for testing only")
    parser.add_argument("--reuse", action="store_true",
                        help="load the saved national model instead of re-fitting")
    args = parser.parse_args()
    out_dir = Path(cfg["paths"]["output_dir"])

    df = md.load_clean(cfg)
    roles = md.load_roles(cfg)
    geo = fm.load_model(cfg, "mmm_geo")          # step 3, not re-fitted

    if args.reuse:
        nat = fm.load_model(cfg, "mmm_national")
    else:
        nat = fm.build_model(cfg, df, "national")    # same spec, national data
        fm.fit(nat, cfg, quick=args.quick)
        fm.save_model(nat, cfg, "mmm_national")
    res = fm.run_checks_and_save(nat, cfg, "step4_national", out_dir)

    cmp = compare_roi(geo, nat)
    why = why_they_differ(df, roles)
    acc = accuracy_comparison(geo, nat)
    learned = how_much_learned(geo, nat)
    learned.to_csv(out_dir / "step4_how_much_learned.csv", index=False)
    cmp.to_csv(out_dir / "step4_geo_vs_national_roi.csv", index=False)
    why.to_csv(out_dir / "step4_why_they_differ.csv", index=False)
    acc.to_csv(out_dir / "step4_accuracy_comparison.csv", index=False)
    chart_roi(cmp, out_dir / "step4_geo_vs_national_roi.png")

    with pd.option_context("display.width", 200, "display.max_colwidth", 80):
        print("\n--- National model convergence ---")
        print(res["conv"].to_string(index=False))
        print("\n--- ROI: geo vs national ---")
        print(cmp.to_string(index=False))
        print("\n--- Why they can differ (training weeks) ---")
        print(why.to_string(index=False))
        print("\n--- How much the data narrowed each range vs the prior (0% = learned nothing) ---")
        print(learned.to_string(index=False))
        print("\n--- National-level accuracy, both models ---")
        print(acc.to_string(index=False))
        print("\n--- National posterior-predictive coverage ---")
        print(res["ppc"].to_string(index=False))
    if not res["conv"]["ok"].all():
        print("WARNING: national model did not fully converge - see the convergence table.")
    if args.quick:
        print("NOTE: --quick run. Numbers are for testing only.")
    print(f"\nSaved: tables and charts in {out_dir}")


if __name__ == "__main__":
    main()
