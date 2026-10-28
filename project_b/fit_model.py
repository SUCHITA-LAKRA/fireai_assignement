"""
Project B - Step 3: Fit the hierarchical Bayesian marketing mix model.

What this script does
---------------------
1. Builds the model inputs (step 2) and the priors (config/project_b.yaml,
   each with its reason).
2. PRIOR-PREDICTIVE CHECK: before seeing any revenue, what do our priors
   imply? Weekly revenue, ROI per channel, and share of revenue from paid
   media. If the priors imply absurd revenue, the priors are wrong.
3. Fits the model by MCMC (4 chains). Meridian's model, per region:
     revenue = region baseline + time effect (knots)
             + controls + promo
             + sum over channels of  coefficient x Hill(Adstock(impressions))
   - adstock: geometric decay, up to 8 weeks
   - saturation (Hill): diminishing returns
   - each region gets its own media coefficient, pulled toward a shared
     average (hierarchical), so small regions borrow strength.
4. CONVERGENCE: R-hat and effective sample size for every parameter.
   Results are flagged as untrustworthy if R-hat > 1.1.
5. POSTERIOR-PREDICTIVE CHECK: simulate revenue from the fitted model
   (including noise) and check it covers what really happened, in the
   training weeks AND the 13 hidden weeks.
6. Saves the posterior (project_b/posterior.nc) and the fitted model, which
   steps 5-7 load instead of re-fitting.

The functions here are reused by step 4 (national model) and step 5
(calibrated model).

How to run (from the repo root, after steps 0-2)
------------------------------------------------
    python -m project_b.fit_model            # full fit (~10-30 min on a laptop)
    python -m project_b.fit_model --quick    # 2-3 min smoke test, NOT for results

Outputs
-------
    project_b/posterior.nc                         posterior draws (ArviZ / NetCDF)
    data/processed/mmm_geo.pkl                     fitted model (git-ignored), loaded by later steps
    outputs/project_b/step3_priors.csv             every prior and why
    outputs/project_b/step3_prior_predictive.csv / .png
    outputs/project_b/step3_convergence.csv        R-hat and ESS per parameter
    outputs/project_b/step3_parameters.csv         posterior summary: ROI, adstock, saturation ...
    outputs/project_b/step3_fit_accuracy.csv       R2 / MAPE / wMAPE, train vs hold-out
    outputs/project_b/step3_posterior_predictive.csv / .png
    outputs/project_b/step3_roi_prior_vs_posterior.png
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)  # library API notices, not our code
warnings.filterwarnings("ignore", category=RuntimeWarning, module="arviz")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_b import model_data as md  # noqa: E402
from project_b.config import REPO_ROOT, load_config  # noqa: E402

TEAL, BLUE, ORANGE = "#1E6F6C", "#2a78d6", "#eb6834"
GREY, INK, INK_2 = "#8c8b86", "#0b0b0b", "#52514e"

# Parameters checked for convergence (all that exist in the model are used)
CONVERGENCE_PARAMS = ["roi_m", "alpha_m", "ec_m", "beta_m", "eta_m", "beta_gm",
                      "alpha_om", "ec_om", "contribution_om", "contribution_n",
                      "gamma_c", "xi_c", "knot_values", "tau_g", "sigma"]


# ---------------------------------------------------------------------------
# 1. Priors and model spec
# ---------------------------------------------------------------------------

def make_prior(cfg: dict, overrides: dict | None = None):
    """Meridian PriorDistribution built from the config (plus optional overrides)."""
    from meridian import backend
    from meridian.model import prior_distribution

    tfd = backend.tfd
    f = backend.np_float_dtype
    builders = {
        "LogNormal": lambda p, n: tfd.LogNormal(f(p["loc"]), f(p["scale"]), name=n),
        "Normal": lambda p, n: tfd.Normal(f(p["loc"]), f(p["scale"]), name=n),
        "HalfNormal": lambda p, n: tfd.HalfNormal(f(p["scale"]), name=n),
        "Uniform": lambda p, n: tfd.Uniform(f(p["low"]), f(p["high"]), name=n),
        "Beta": lambda p, n: tfd.Beta(f(p["a"]), f(p["b"]), name=n),
        "Deterministic": lambda p, n: tfd.Deterministic(f(p["value"]), name=n),
        "TruncatedNormal": lambda p, n: tfd.TruncatedNormal(
            f(p["loc"]), f(p["scale"]), f(p["low"]), f(p["high"]), name=n),
    }
    kwargs = {name: builders[p["dist"]](p, name)
              for name, p in cfg["model"]["priors"].items()}
    kwargs.update(overrides or {})
    return prior_distribution.PriorDistribution(**kwargs)


def priors_table(cfg: dict) -> pd.DataFrame:
    rows = []
    for name, p in cfg["model"]["priors"].items():
        params = {k: v for k, v in p.items() if k not in ("dist", "why")}
        rows.append({"parameter": name, "distribution": p["dist"],
                     "settings": ", ".join(f"{k}={v}" for k, v in params.items()),
                     "why": p["why"]})
    return pd.DataFrame(rows)


def make_knots(df: pd.DataFrame, spacing: int) -> list[int]:
    """Knot positions (week index) for the smooth time effect.

    One knot every `spacing` weeks, but never inside the hold-out: a knot
    there would have no revenue to learn from. Knots are added at the last
    training week before the hold-out and the first one after it, so the time
    effect is carried flat into the hidden weeks.
    """
    weeks = np.sort(df[md.TIME].unique())
    hold = np.isin(weeks, df.loc[df["is_holdout"], md.TIME].unique())
    n = len(weeks)
    knots = set(range(0, n, spacing)) | {n - 1}
    knots = {k for k in knots if not hold[k]}
    idx = np.where(hold)[0]
    if len(idx):
        if idx[0] > 0:
            knots.add(int(idx[0] - 1))
        if idx[-1] < n - 1:
            knots.add(int(idx[-1] + 1))
    return sorted(int(k) for k in knots)


def make_spec(cfg: dict, df: pd.DataFrame, level: str, prior=None, **extra):
    from meridian.model import spec

    start, end = md.holdout_window(df)
    knots = make_knots(df, cfg["model"]["knot_spacing_weeks"])
    return spec.ModelSpec(
        prior=prior if prior is not None else make_prior(cfg),
        media_prior_type="roi",
        max_lag=cfg["model"]["max_lag"],
        knots=knots,
        holdout=md.holdout_spec(start, end),
        **extra,
    )


def build_model(cfg: dict, df: pd.DataFrame, level: str = "geo", prior=None, **extra):
    from meridian.model import model

    roles = md.load_roles(cfg)
    if level == "national":
        data = md.build_input_data(md.national_frame(df, roles), roles, "national")
    else:
        data = md.build_input_data(df, roles, "geo")
    return model.Meridian(input_data=data, model_spec=make_spec(cfg, df, level, prior, **extra))


# ---------------------------------------------------------------------------
# 2. Prior-predictive check
# ---------------------------------------------------------------------------

def actual_revenue(mmm) -> np.ndarray:
    """Actual revenue, shape (geos, times)."""
    d = mmm.input_data
    return np.asarray(d.kpi) * np.asarray(d.revenue_per_kpi)


def prior_predictive(mmm, cfg: dict) -> tuple[pd.DataFrame, dict]:
    from meridian.analysis import analyzer

    a = analyzer.Analyzer(mmm)
    weekly = np.asarray(a.expected_outcome(use_posterior=False, aggregate_geos=True,
                                           aggregate_times=False))  # (1, draws, times)
    weekly = weekly.reshape(-1, weekly.shape[-1])
    actual = actual_revenue(mmm).sum(axis=0)
    roi = np.asarray(a.roi(use_posterior=False)).reshape(-1, len(mmm.input_data.media_channel))
    inc = np.asarray(a.incremental_outcome(use_posterior=False, include_non_paid_channels=False))
    share = inc.reshape(-1, inc.shape[-1]).sum(axis=1) / actual.sum()

    q = lambda x: np.quantile(x, [0.05, 0.5, 0.95])  # noqa: E731
    rows = []
    tot = weekly.mean(axis=1)
    rows.append(["average weekly revenue (all regions)", *q(tot), actual.mean()])
    rows.append(["share of revenue from paid media", *q(share), np.nan])
    for i, ch in enumerate(mmm.input_data.media_channel.values):
        rows.append([f"ROI {ch}", *q(roi[:, i]), np.nan])
    table = pd.DataFrame(rows, columns=["quantity", "prior_p05", "prior_median", "prior_p95", "actual"])
    inside = float(np.mean((actual >= np.quantile(weekly, 0.05, axis=0)) &
                           (actual <= np.quantile(weekly, 0.95, axis=0))))
    table.loc[len(table)] = ["share of weeks where actual revenue is inside the prior 90% band",
                             np.nan, inside, np.nan, np.nan]
    by_region = np.asarray(a.expected_outcome(use_posterior=False, aggregate_geos=False,
                                              aggregate_times=False))
    table.loc[len(table)] = ["share of national weeks with NEGATIVE revenue (should be ~0)",
                             np.nan, float(np.mean(weekly < 0)), np.nan, np.nan]
    table.loc[len(table)] = ["share of region-weeks with NEGATIVE revenue (should be small)",
                             np.nan, float(np.mean(by_region < 0)), np.nan, np.nan]
    return table, {"weekly": weekly, "actual": actual}


# ---------------------------------------------------------------------------
# 3. Fit, convergence, posterior-predictive check
# ---------------------------------------------------------------------------

def fit(mmm, cfg: dict, quick: bool = False) -> float:
    s = cfg["model"]["quick_sampling" if quick else "sampling"]
    seed = cfg["model"]["sampling"]["seed"]
    print(f"Sampling: {s['n_chains']} chains x ({s['n_adapt']} adapt + {s['n_burnin']} "
          f"burn-in + {s['n_keep']} kept)" + ("  [QUICK - not for results]" if quick else ""))
    t = time.time()
    mmm.sample_prior(cfg["model"]["prior_draws"], seed=seed)
    mmm.sample_posterior(n_chains=s["n_chains"], n_adapt=s["n_adapt"],
                         n_burnin=s["n_burnin"], n_keep=s["n_keep"], seed=seed)
    minutes = (time.time() - t) / 60
    print(f"Sampling took {minutes:.1f} minutes")
    return minutes


def convergence(mmm, cfg: dict) -> pd.DataFrame:
    import arviz as az

    post = mmm.inference_data.posterior
    names = [p for p in CONVERGENCE_PARAMS if p in post.data_vars]
    rows = []
    for p in names:
        if np.allclose(post[p].std().item(), 0):
            continue  # fixed (deterministic) parameter
        rh = az.rhat(post[p]).to_array().values.ravel()
        ess = az.ess(post[p], method="bulk").to_array().values.ravel()
        rows.append({"parameter": p, "n_values": int(rh.size),
                     "max_rhat": round(float(np.nanmax(rh)), 3),
                     "min_ess_bulk": int(np.nanmin(ess)),
                     "ok": bool(np.nanmax(rh) <= cfg["model"]["max_rhat"])})
    return pd.DataFrame(rows)


def parameter_summary(mmm) -> pd.DataFrame:
    """Posterior mean and 90% interval of the parameters a reader cares about."""
    post = mmm.inference_data.posterior
    prior = mmm.inference_data.prior
    rows = []
    labels = {"roi_m": "ROI, all weeks",
              "alpha_m": "adstock decay per week",
              "ec_m": "half-saturation point",
              "eta_m": "spread of media effect across regions"}
    for p, label in labels.items():
        if p not in post or p not in prior:
            continue
        for ch in post[p].coords["media_channel"].values:
            x = post[p].sel(media_channel=ch).values.ravel()
            pr = prior[p].sel(media_channel=ch).values.ravel()
            rows.append({"parameter": p, "meaning": label, "channel": str(ch),
                         "mean": x.mean(), "p05": np.quantile(x, 0.05), "p95": np.quantile(x, 0.95),
                         "prior_p05": np.quantile(pr, 0.05), "prior_p95": np.quantile(pr, 0.95)})
    for p, label, coord in [("contribution_om", "organic share of revenue", "organic_media_channel"),
                            ("contribution_n", "promo share of revenue", "non_media_channel"),
                            ("gamma_c", "control effect (standardised)", "control_variable")]:
        if p not in post:
            continue
        for c in post[p].coords[coord].values:
            x = post[p].sel({coord: c}).values.ravel()
            pr = prior[p].sel({coord: c}).values.ravel()
            rows.append({"parameter": p, "meaning": label, "channel": str(c),
                         "mean": x.mean(), "p05": np.quantile(x, 0.05), "p95": np.quantile(x, 0.95),
                         "prior_p05": np.quantile(pr, 0.05), "prior_p95": np.quantile(pr, 0.95)})
    out = pd.DataFrame(rows)
    num = ["mean", "p05", "p95", "prior_p05", "prior_p95"]
    out[num] = out[num].round(3)
    return out


def fit_accuracy(mmm) -> pd.DataFrame:
    from meridian.analysis import analyzer

    acc = analyzer.Analyzer(mmm).predictive_accuracy().to_dataframe().reset_index()
    return acc.rename(columns={"value": "value"}).round(4)


def posterior_predictive(mmm, n_draws: int = 400, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Simulate revenue = model mean + noise, compare with what happened.

    Meridian's noise is on the standardised per-person scale; one unit there
    equals (stdev x population) conversions, times revenue per conversion.
    A good model has ~90% of actual values inside its 90% band.
    """
    from meridian.analysis import analyzer

    a = analyzer.Analyzer(mmm)
    mean = np.asarray(a.expected_outcome(use_posterior=True, aggregate_geos=False,
                                         aggregate_times=False))  # (chains, draws, geos, times)
    mean = mean.reshape(-1, *mean.shape[-2:])
    sigma = mmm.inference_data.posterior["sigma"].values.reshape(mean.shape[0], -1)
    rng = np.random.default_rng(seed)
    pick = rng.choice(mean.shape[0], size=min(n_draws, mean.shape[0]), replace=False)
    mean, sigma = mean[pick], sigma[pick]
    stdev = float(np.asarray(mmm.kpi_transformer.population_scaled_stdev))
    pop = np.asarray(mmm.input_data.population)
    if pop.ndim == 0 or pop.size == 1:
        pop = np.ones(mean.shape[1])
    rpk = np.asarray(mmm.input_data.revenue_per_kpi)
    scale = stdev * pop[:, None] * rpk  # (geos, times)
    if sigma.shape[1] == 1:
        noise_sd = sigma[:, 0][:, None, None] * scale[None]
    else:
        noise_sd = sigma[:, :, None] * scale[None]
    sim = mean + rng.normal(size=mean.shape) * noise_sd

    actual = actual_revenue(mmm)
    hold = np.asarray(mmm.holdout_id).astype(bool)
    if hold.ndim == 1:
        hold = np.broadcast_to(hold, actual.shape)
    lo, hi = np.quantile(sim, 0.05, axis=0), np.quantile(sim, 0.95, axis=0)
    inside = (actual >= lo) & (actual <= hi)

    nat_sim = sim.sum(axis=1)
    nat_act = actual.sum(axis=0)
    nat_lo, nat_hi = np.quantile(nat_sim, 0.05, axis=0), np.quantile(nat_sim, 0.95, axis=0)
    nat_in = (nat_act >= nat_lo) & (nat_act <= nat_hi)
    hold_t = hold.any(axis=0)

    rows = [
        ["region-week, training", float(inside[~hold].mean())],
        ["region-week, hold-out", float(inside[hold].mean()) if hold.any() else np.nan],
        ["national week, training", float(nat_in[~hold_t].mean())],
        ["national week, hold-out", float(nat_in[hold_t].mean()) if hold_t.any() else np.nan],
    ]
    if actual.shape[0] == 1:  # national model: region rows would just repeat the national ones
        rows = rows[2:]
    table = pd.DataFrame(rows, columns=["level", "share_of_actuals_inside_90pct_band"]).round(3)
    lines = {"time": pd.to_datetime(mmm.input_data.time.values), "actual": nat_act,
             "mean": mean.sum(axis=1).mean(axis=0), "lo": nat_lo, "hi": nat_hi, "hold": hold_t}
    return table, lines


# ---------------------------------------------------------------------------
# 4. Charts
# ---------------------------------------------------------------------------

def _style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)


def chart_prior_predictive(pp: dict, times, out: Path) -> None:
    w = pp["weekly"] / 1e6
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(times, np.quantile(w, 0.05, 0), np.quantile(w, 0.95, 0), color=TEAL,
                    alpha=0.15, lw=0, label="Priors: 90% of simulated revenue")
    ax.fill_between(times, np.quantile(w, 0.25, 0), np.quantile(w, 0.75, 0), color=TEAL,
                    alpha=0.3, lw=0, label="Priors: middle 50%")
    ax.plot(times, pp["actual"] / 1e6, color=INK, lw=1.6, label="Actual revenue")
    ax.set_title("Prior-predictive check: what the priors expect before seeing revenue "
                 "(weekly, all regions, millions)", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncols=3)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.12)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_posterior_predictive(lines: dict, out: Path) -> None:
    t = lines["time"]
    fig, ax = plt.subplots(figsize=(10, 4))
    if lines["hold"].any():
        h = t[lines["hold"]]
        ax.axvspan(h.min() - pd.Timedelta(days=3), h.max() + pd.Timedelta(days=3),
                   color=GREY, alpha=0.15, lw=0)
        ax.text(h.min(), 0.02, " held-out weeks", transform=ax.get_xaxis_transform(),
                color=INK_2, fontsize=9)
    ax.fill_between(t, lines["lo"] / 1e6, lines["hi"] / 1e6, color=BLUE, alpha=0.2, lw=0,
                    label="Model: 90% band (with noise)")
    ax.plot(t, lines["mean"] / 1e6, color=BLUE, lw=1.6, label="Model: expected")
    ax.plot(t, lines["actual"] / 1e6, color=INK, lw=1.3, label="Actual")
    ax.set_title("Posterior-predictive check: model vs actual weekly revenue (all regions, millions)",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncols=3)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.15)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_roi_prior_posterior(params: pd.DataFrame, out: Path) -> None:
    r = params[params["parameter"] == "roi_m"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    y = np.arange(len(r))[::-1]
    ax.hlines(y + 0.15, r["prior_p05"], r["prior_p95"], color=GREY, lw=5, alpha=0.5,
              label="Prior 90% range")
    ax.hlines(y - 0.15, r["p05"], r["p95"], color=TEAL, lw=5, label="Posterior 90% range")
    ax.plot(r["mean"], y - 0.15, "o", color=TEAL, ms=8, mec="white", mew=1.5)
    for yi, m, lo, hi in zip(y, r["mean"], r["p05"], r["p95"]):
        ax.text(hi + 0.08, yi - 0.15, f"{m:.2f}  ({lo:.2f}–{hi:.2f})", va="center",
                fontsize=9, color=INK)
    ax.axvline(1, color=INK_2, lw=1, ls=(0, (4, 3)))
    ax.text(1.03, y.min() - 0.55, "break-even (ROI 1)", color=INK_2, fontsize=8.5, va="center")
    ax.set_yticks(y, r["channel"])
    ax.set_ylim(y.min() - 0.8, y.max() + 0.5)
    ax.set_xlim(0, max(6, r["p95"].max() + 1.6))
    ax.set_xlabel("Revenue per 1 spent", fontsize=9, color=INK_2)
    ax.set_title("ROI: what we assumed (grey) vs what the data says (teal)", loc="left",
                 fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#e4e3df", lw=0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def save_model(mmm, cfg: dict, name: str) -> Path:
    """Save the fitted model so later steps can load it instead of re-fitting.

    Uses Meridian's pickle saver. Its newer proto format needs the extra
    `google-meridian[schema]` package; not worth another dependency here.
    """
    from meridian.model import model

    path = Path(cfg["paths"]["processed_dir"]) / f"{name}.pkl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        model.save_mmm(mmm, str(path))
    return path


def load_model(cfg: dict, name: str = "mmm_geo"):
    from meridian.model import model

    path = Path(cfg["paths"]["processed_dir"]) / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run step 3 first: python -m project_b.fit_model")
    return model.load_mmm(str(path))


def run_checks_and_save(mmm, cfg: dict, prefix: str, out_dir: Path) -> dict:
    """Convergence, parameters, accuracy and posterior-predictive outputs."""
    conv = convergence(mmm, cfg)
    params = parameter_summary(mmm)
    acc = fit_accuracy(mmm)
    ppc, lines = posterior_predictive(mmm, seed=cfg["model"]["sampling"]["seed"])
    conv.to_csv(out_dir / f"{prefix}_convergence.csv", index=False)
    params.to_csv(out_dir / f"{prefix}_parameters.csv", index=False)
    acc.to_csv(out_dir / f"{prefix}_fit_accuracy.csv", index=False)
    ppc.to_csv(out_dir / f"{prefix}_posterior_predictive.csv", index=False)
    chart_posterior_predictive(lines, out_dir / f"{prefix}_posterior_predictive.png")
    chart_roi_prior_posterior(params, out_dir / f"{prefix}_roi_prior_vs_posterior.png")
    return {"conv": conv, "params": params, "acc": acc, "ppc": ppc}


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="few draws, for testing the pipeline only")
    args = parser.parse_args()

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df = md.load_clean(cfg)
    start, end = md.holdout_window(df)
    print(f"Hold-out: {start.date()} to {end.date()}")

    mmm = build_model(cfg, df, "geo")
    print(f"Time-effect knots at weeks: {make_knots(df, cfg['model']['knot_spacing_weeks'])}")
    priors_table(cfg).to_csv(out_dir / "step3_priors.csv", index=False)

    minutes = fit(mmm, cfg, quick=args.quick)

    pp_table, pp = prior_predictive(mmm, cfg)
    pp_table.round(3).to_csv(out_dir / "step3_prior_predictive.csv", index=False)
    chart_prior_predictive(pp, pd.to_datetime(mmm.input_data.time.values),
                           out_dir / "step3_prior_predictive.png")

    res = run_checks_and_save(mmm, cfg, "step3", out_dir)

    nc = REPO_ROOT / "project_b" / "posterior.nc"
    mmm.inference_data.to_netcdf(str(nc))
    pkl = save_model(mmm, cfg, "mmm_geo")
    run_info = pd.DataFrame([{"quick_mode": args.quick, "sampling_minutes": round(minutes, 1),
                              "holdout_start": start.date(), "holdout_end": end.date(),
                              "all_converged": bool(res["conv"]["ok"].all())}])
    run_info.to_csv(out_dir / "step3_run_info.csv", index=False)

    with pd.option_context("display.width", 170, "display.max_colwidth", 60):
        print("\n--- Prior-predictive check ---")
        print(pp_table.round(3).to_string(index=False))
        print("\n--- Convergence (R-hat should be <= "
              f"{cfg['model']['max_rhat']}, ESS ideally > 400) ---")
        print(res["conv"].to_string(index=False))
        print("\n--- Key parameters (posterior vs prior, 90% ranges) ---")
        print(res["params"].to_string(index=False))
        print("\n--- Fit accuracy (Train vs Test = hold-out) ---")
        print(res["acc"].to_string(index=False))
        print("\n--- Posterior-predictive coverage (target about 0.90) ---")
        print(res["ppc"].to_string(index=False))
    print(f"\nSaved: {nc} ({nc.stat().st_size / 1e6:.1f} MB), {pkl}, tables and charts in {out_dir}")
    if not res["conv"]["ok"].all():
        print("WARNING: some parameters did not converge (R-hat too high). "
              "Increase n_adapt / n_burnin / n_keep in config/project_b.yaml.")
    if args.quick:
        print("NOTE: --quick run. Numbers are for testing only; run without --quick for results.")


if __name__ == "__main__":
    main()
