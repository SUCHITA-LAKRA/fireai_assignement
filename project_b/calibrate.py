"""
Project B - Step 5: Calibrate the model with the Channel3 experiment.

What this script does
---------------------
1. Re-states the step 0 check against our own data: did Channel3 spend
   actually drop in Geo0-Geo7 during the test weeks? (It didn't.)
2. Asks the UNCALIBRATED model (step 3) what the test should have found:
   Channel3's ROI in exactly those regions and weeks, vs the vendor's 0.8.
3. Feeds the vendor's result into the model as a PRIOR on Channel3's ROI
   during the test weeks (Meridian's ROI calibration). How much we trust it
   is a single number, `trust_weight` in the config (a "power prior"):
       vendor's result as a log-normal (median 0.8, spread from 0.3-1.3),
       its uncertainty multiplied by 1/sqrt(trust_weight),
       combined with our usual ROI prior (precision-weighted average).
   - our choice: trust_weight = 0.25   (the "calibrated" model, used from here on)
   - for comparison: trust_weight = 1  (believe the vendor fully)
   - trust_weight = 0 is just the step 3 model
4. Compares ROI for every channel across the three, plus convergence and
   hold-out accuracy (calibration should not make the fit worse).
5. Saves the calibrated model; steps 6-7 use it.

How to run (from the repo root, after step 3)
---------------------------------------------
    python -m project_b.calibrate             # two fits, ~20-25 min on 2 CPUs
    python -m project_b.calibrate --quick     # smoke test, not for results
    python -m project_b.calibrate --reuse     # re-compare without re-fitting

Outputs
-------
    project_b/posterior_calibrated.nc                 posterior of the calibrated model
    data/processed/mmm_calibrated.pkl                 calibrated model (steps 6-7 load it)
    outputs/project_b/step5_experiment_vs_data.csv    the experiment checked against data and model
    outputs/project_b/step5_roi_comparison.csv / .png ROI: uncalibrated vs weighted vs full trust
    outputs/project_b/step5_channel3_prior_posterior.png
    outputs/project_b/step5_fit_comparison.csv        convergence and hold-out accuracy, all three
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
from project_b.config import REPO_ROOT, load_config  # noqa: E402

TEAL, BLUE, ORANGE = "#1E6F6C", "#2a78d6", "#eb6834"
GREY, INK, INK_2 = "#8c8b86", "#0b0b0b", "#52514e"
VARIANT_COLOURS = {"uncalibrated": GREY, "calibrated": TEAL, "full trust": ORANGE}


# ---------------------------------------------------------------------------
# 1. The experiment as a prior
# ---------------------------------------------------------------------------

def test_window(cfg: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    e = cfg["experiment"]
    start = pd.Timestamp(e["start"])
    return start, start + pd.Timedelta(weeks=e["weeks"] - 1)  # last week (inclusive)


def vendor_lognormal(cfg: dict) -> tuple[float, float]:
    """The vendor's result as a log-normal: (log-median, log-sd).

    Median = the reported 0.8. Spread = the reported 90% interval's width on
    the log scale: (log 1.3 - log 0.3) / (2 x 1.645) = 0.446.
    """
    e = cfg["experiment"]
    lo, hi = e["reported_ci90"]
    return float(np.log(e["reported_iroas"])), float((np.log(hi) - np.log(lo)) / (2 * 1.645))


def channel_prior(cfg: dict, trust: float) -> tuple[float, float]:
    """Usual ROI prior combined with the vendor's result at weight `trust`.

    Both are normal on the log scale, so combining them is a precision-weighted
    average (precision = 1 / sd^2). Weight `trust` scales the vendor's
    precision: trust 0.25 counts it as a quarter of an experiment, i.e. its
    sd is doubled.
    """
    usual = cfg["model"]["priors"]["roi_m"]
    m0, s0 = float(usual["loc"]), float(usual["scale"])
    if trust <= 0:
        return m0, s0
    mv, sv = vendor_lognormal(cfg)
    p0, pv = 1 / s0 ** 2, trust / sv ** 2
    return (p0 * m0 + pv * mv) / (p0 + pv), float(np.sqrt(1 / (p0 + pv)))


def calibrated_roi_prior(cfg: dict, channels: list[str], trust: float):
    """Per-channel log-normal ROI prior; only the test channel is changed."""
    from meridian import backend

    tfd, f = backend.tfd, backend.np_float_dtype
    usual = cfg["model"]["priors"]["roi_m"]
    loc = np.full(len(channels), usual["loc"], dtype=f)
    scale = np.full(len(channels), usual["scale"], dtype=f)
    i = channels.index(cfg["experiment"]["channel"])
    loc[i], scale[i] = channel_prior(cfg, trust)
    return tfd.LogNormal(loc, scale, name="roi_m")


def build_calibrated(cfg: dict, df: pd.DataFrame, trust: float):
    """Geo model (step 3 spec) + Channel3 ROI prior that applies to the test weeks."""
    from meridian.model import spec

    roles = md.load_roles(cfg)
    start, end = test_window(cfg)
    calibration = spec.CalibrationSpec(spec=[spec.ChannelCalibrationSpec(
        channels=[cfg["experiment"]["channel"]],
        date_ranges=[spec.DateRange(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))])])
    prior = fm.make_prior(cfg, {"roi_m": calibrated_roi_prior(cfg, roles["channels"], trust)})
    return fm.build_model(cfg, df, "geo", prior=prior, roi_calibration=calibration)


# ---------------------------------------------------------------------------
# 2. ROI from any model, over any regions / weeks
# ---------------------------------------------------------------------------

def roi_draws(mmm, geos=None, times=None) -> pd.DataFrame:
    """ROI draws per channel = extra revenue from the channel / its spend.

    `times` limits it to ads run in those weeks (their effect is followed
    into later weeks); `geos` limits it to those regions.
    """
    from meridian.analysis import analyzer

    a = analyzer.Analyzer(mmm)
    r = np.asarray(a.roi(use_posterior=True, selected_geos=geos, selected_times=times))
    chans = [str(c) for c in mmm.input_data.media_channel.values]
    return pd.DataFrame(r.reshape(-1, len(chans)), columns=chans)


def summarise(draws: pd.DataFrame, label: str, scope: str) -> pd.DataFrame:
    rows = []
    for ch in draws.columns:
        x = draws[ch].to_numpy()
        rows.append({"model": label, "scope": scope, "channel": ch, "mean": x.mean(),
                     "p05": np.quantile(x, 0.05), "p95": np.quantile(x, 0.95),
                     "prob_above_1": float((x > 1).mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. The experiment checked against the data and the model
# ---------------------------------------------------------------------------

def experiment_vs_data(cfg: dict, df: pd.DataFrame, uncal) -> pd.DataFrame:
    e = cfg["experiment"]
    start, end = test_window(cfg)
    out_dir = Path(cfg["paths"]["output_dir"])
    rows = []

    step0 = out_dir / "step0_experiment_check.csv"
    if step0.exists():
        for _, r in pd.read_csv(step0).iterrows():
            rows.append({"evidence": "data (step 0)", "check": r["check"], "value": r["value"],
                         "expected_if_test_was_real": r["expected_if_switched_off"]})

    weeks = [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="7D")]
    in_test = roi_draws(uncal, geos=e["test_geos"], times=weeks)[e["channel"]]
    everywhere = roi_draws(uncal, times=weeks)[e["channel"]]
    vendor_mean = float(e["reported_iroas"])
    rows += [
        {"evidence": "model (step 3, no experiment used)",
         "check": f"{e['channel']} ROI in the test regions and weeks",
         "value": f"{in_test.mean():.2f} (90%: {in_test.quantile(.05):.2f}-{in_test.quantile(.95):.2f})",
         "expected_if_test_was_real": f"vendor: {vendor_mean} (90%: {e['reported_ci90'][0]}-{e['reported_ci90'][1]})"},
        {"evidence": "model (step 3, no experiment used)",
         "check": f"{e['channel']} ROI in the test weeks, all regions",
         "value": f"{everywhere.mean():.2f} (90%: {everywhere.quantile(.05):.2f}-{everywhere.quantile(.95):.2f})",
         "expected_if_test_was_real": "-"},
        {"evidence": "model (step 3, no experiment used)",
         "check": "chance the model's test-cell ROI is at or below the vendor's 0.8",
         "value": f"{(in_test <= vendor_mean).mean():.0%}",
         "expected_if_test_was_real": "-"},
    ]
    hold = df.loc[df["is_holdout"], md.TIME]
    overlap = int(((hold >= start) & (hold <= end)).groupby(hold).first().sum()) if len(hold) else 0
    rows.append({"evidence": "set-up", "check": "hold-out weeks inside the test window",
                 "value": f"{overlap} weeks", "expected_if_test_was_real": "0 is cleanest"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Charts
# ---------------------------------------------------------------------------

def _style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="x", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)


def chart_roi(table: pd.DataFrame, out: Path) -> None:
    t = table[table["scope"] == "all weeks"]
    variants = list(dict.fromkeys(t["model"]))
    chans = list(dict.fromkeys(t["channel"]))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    y0 = np.arange(len(chans))[::-1]
    offs = np.linspace(0.24, -0.24, len(variants))
    for v, off in zip(variants, offs):
        d = t[t["model"] == v].set_index("channel").loc[chans]
        c = VARIANT_COLOURS.get(v.split(" (")[0], BLUE)
        ax.hlines(y0 + off, d["p05"], d["p95"], color=c, lw=4.5, label=v)
        ax.plot(d["mean"], y0 + off, "o", color=c, ms=7, mec="white", mew=1.3)
        for yi, m, hi in zip(y0 + off, d["mean"], d["p95"]):
            ax.text(hi + 0.06, yi, f"{m:.2f}", va="center", fontsize=8.5, color=INK)
    ax.axvline(1, color=INK_2, lw=1, ls=(0, (4, 3)))
    ax.text(1.03, y0.min() - 0.62, "break-even (ROI 1)", color=INK_2, fontsize=8.5, va="center")
    ax.set_yticks(y0, chans)
    ax.set_ylim(y0.min() - 0.85, y0.max() + 0.5)
    ax.set_xlim(0, t["p95"].max() + 0.5)
    ax.set_xlabel("ROI over all weeks (dot = average, bar = 90% range)", fontsize=9, color=INK_2)
    ax.set_title("ROI with and without the Channel3 experiment", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1.0, -0.13), ncols=3)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_channel3(models: dict, cfg: dict, out: Path) -> None:
    """Channel3 ROI in the test weeks: prior and posterior for each variant."""
    ch = cfg["experiment"]["channel"]
    fig, axes = plt.subplots(1, len(models), figsize=(10, 3.4), sharey=True)
    bins = np.linspace(0, 4, 61)
    for ax, (label, m) in zip(np.atleast_1d(axes), models.items()):
        idx = [str(c) for c in m.input_data.media_channel.values].index(ch)
        pr = m.inference_data.prior["roi_m"].values.reshape(-1, len(m.input_data.media_channel))[:, idx]
        po = m.inference_data.posterior["roi_m"].values.reshape(-1, len(m.input_data.media_channel))[:, idx]
        c = VARIANT_COLOURS.get(label.split(" (")[0], BLUE)
        # Each draw weighs 1/(n x bin width): values above 4 fall off the chart
        # instead of piling up at the edge.
        width = bins[1] - bins[0]
        ax.hist(pr, bins=bins, weights=np.full(len(pr), 1 / (len(pr) * width)),
                color=GREY, alpha=0.35, label="prior")
        ax.hist(po, bins=bins, weights=np.full(len(po), 1 / (len(po) * width)),
                color=c, alpha=0.8, label="posterior")
        ax.axvline(cfg["experiment"]["reported_iroas"], color=INK, lw=1, ls=(0, (3, 2)))
        ax.set_title(label, fontsize=10, color=INK, loc="left")
        scope = "all weeks" if label.startswith("uncalibrated") else "the test weeks"
        ax.set_xlabel(f"{ch} ROI, {scope}", fontsize=8.5, color=INK_2)
        ax.text(0.83, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] else 1, " vendor 0.8",
                fontsize=8, color=INK_2)
        _style(ax)
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(f"{ch}: what each model assumed (grey) and concluded (colour)", x=0.01,
                 ha="left", fontsize=11, color=INK)
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
    parser.add_argument("--reuse", action="store_true", help="load saved models, don't re-fit")
    args = parser.parse_args()
    out_dir = Path(cfg["paths"]["output_dir"])
    e = cfg["experiment"]

    df = md.load_clean(cfg)
    uncal = fm.load_model(cfg, "mmm_geo")

    # Fit the chosen calibrated model and the comparison(s)
    trusts = [e["trust_weight"]] + [w for w in e.get("trust_weights_to_compare", []) if w != e["trust_weight"]]
    for w in [0.0] + trusts:
        loc, sc = channel_prior(cfg, w)
        print(f"trust {w:g}: {e['channel']} ROI prior in test weeks = LogNormal({loc:.3f}, {sc:.3f}) -> "
              f"median {np.exp(loc):.2f}, 90% {np.exp(loc - 1.645 * sc):.2f}-{np.exp(loc + 1.645 * sc):.2f}")
    fitted = {}
    for w in trusts:
        name = "mmm_calibrated" if w == e["trust_weight"] else f"mmm_trust_{w:g}"
        if args.reuse:
            fitted[w] = fm.load_model(cfg, name)
        else:
            print(f"\n=== Fitting with trust_weight = {w:g} ===")
            m = build_calibrated(cfg, df, w)
            fm.fit(m, cfg, quick=args.quick)
            fm.save_model(m, cfg, name)
            fitted[w] = m

    def label(w):
        if w == e["trust_weight"]:
            return f"calibrated (trust {w:g})"
        return "full trust (1)" if w == 1 else f"trust {w:g}"
    models = {"uncalibrated (trust 0)": uncal, **{label(w): m for w, m in fitted.items()}}

    # Checks for each calibrated fit (same outputs as step 3, own prefix)
    chosen = fitted[e["trust_weight"]]
    res = fm.run_checks_and_save(chosen, cfg, "step5_calibrated", out_dir)
    chosen.inference_data.to_netcdf(str(REPO_ROOT / "project_b" / "posterior_calibrated.nc"))

    # ROI comparison: all weeks, and Channel3 in the test weeks
    start, end = test_window(cfg)
    weeks = [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="7D")]
    tables = []
    for lab, m in models.items():
        tables.append(summarise(roi_draws(m), lab, "all weeks"))
        tw = roi_draws(m, times=weeks)[[e["channel"]]]
        tables.append(summarise(tw, lab, "test weeks"))
        tg = roi_draws(m, geos=e["test_geos"], times=weeks)[[e["channel"]]]
        tables.append(summarise(tg, lab, "test regions and weeks"))
    roi = pd.concat(tables, ignore_index=True)
    num = ["mean", "p05", "p95", "prob_above_1"]
    roi[num] = roi[num].round(3)
    roi.to_csv(out_dir / "step5_roi_comparison.csv", index=False)

    # Fit quality for all three
    fit_rows = []
    for lab, m in models.items():
        conv = fm.convergence(m, cfg)
        acc = fm.fit_accuracy(m)
        test = acc[(acc["evaluation_set"] == "Test")]
        get = lambda metric, gran: float(test[(test["metric"] == metric) &  # noqa: E731
                                              (test["geo_granularity"] == gran)]["value"].iloc[0])
        fit_rows.append({"model": lab, "max_rhat": conv["max_rhat"].max(),
                         "all_converged": bool(conv["ok"].all()),
                         "holdout_national_MAPE": get("MAPE", "national"),
                         "holdout_national_R2": get("R_Squared", "national"),
                         "holdout_region_week_wMAPE": get("wMAPE", "geo")})
    fits = pd.DataFrame(fit_rows).round(4)
    fits.to_csv(out_dir / "step5_fit_comparison.csv", index=False)

    evidence = experiment_vs_data(cfg, df, uncal)
    evidence.to_csv(out_dir / "step5_experiment_vs_data.csv", index=False)

    chart_roi(roi, out_dir / "step5_roi_comparison.png")
    chart_channel3(models, cfg, out_dir / "step5_channel3_prior_posterior.png")

    with pd.option_context("display.width", 200, "display.max_colwidth", 70):
        print("\n--- The experiment checked against data and model ---")
        print(evidence.to_string(index=False))
        print("\n--- ROI comparison ---")
        print(roi.to_string(index=False))
        print("\n--- Fit quality ---")
        print(fits.to_string(index=False))
        print("\n--- Calibrated model: posterior-predictive coverage ---")
        print(res["ppc"].to_string(index=False))
    if not fits["all_converged"].all():
        print("WARNING: a model did not converge - see the fit table.")
    if args.quick:
        print("NOTE: --quick run. Numbers are for testing only.")
    print(f"\nSaved: calibrated model, posterior_calibrated.nc, tables and charts in {out_dir}")


if __name__ == "__main__":
    main()
