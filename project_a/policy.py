"""
Project A - Policies: policy(state) -> orders.

    orders = {"po": units per item (read on DC order days only),
              "transfers": units per item x store}

`state` is built by simulate.py each day; state["context"] holds the static
inputs (items, costs, lead times, caps, demand history up to the cutoff).
Policies may keep per-run results in state["context"]["_cache"]; the
simulator clears it before every run.

base_stock_policy - the BASELINE the brief asks us to beat
-----------------------------------------------------------
Textbook order-up-to levels, one per location, from the mean and standard
deviation of daily demand over the last 56 training days:

  store (reviews daily, 2-day transfer):
      S = mu * 3 + z * sigma * sqrt(3)              protection = 2 + 1 days
      transfer = S - (store on hand + in transit)

  DC (orders Mondays, lognormal supplier lead time L):
      S = mu_dc * (E[L] + 7)
          + z * sqrt((E[L] + 7) * sigma_dc^2 + mu_dc^2 * Var(L))
      PO = S - (DC on hand + on order)
      mu_dc = sum of store means, sigma_dc^2 = sum of store variances

  z = 1.645 (95% cycle service at each location)

The improved policy (step 6) will be added below and exported as `policy`.
"""

import numpy as np
from scipy.stats import norm

PLAN_DAYS = 56          # training days used for mean / sd


def _demand_stats(ctx: dict, days: int = PLAN_DAYS):
    """Mean and sd of daily demand per item x store over the last `days`."""
    h = ctx["demand_history"][:, :, -days:]
    n = np.sum(~np.isnan(h), axis=2)
    mu = np.where(n > 0, np.nansum(h, axis=2) / np.maximum(n, 1), 0.0)
    var = np.where(n > 1, np.nansum((h - mu[:, :, None]) ** 2, axis=2) /
                   np.maximum(n - 1, 1), 0.0)
    return mu, np.sqrt(var)


def _base_stock_levels(ctx: dict, z: float):
    mu, sd = _demand_stats(ctx)
    p_store = ctx["dc_to_store_days"] + 1
    s_store = np.ceil(mu * p_store + z * sd * np.sqrt(p_store))
    s_store = np.minimum(s_store, ctx["store_cap"])

    mu_dc, var_dc = mu.sum(axis=1), (sd ** 2).sum(axis=1)
    el = ctx["lead_time_mean"]
    var_l = (ctx["lead_time_cv"] * el) ** 2
    p_dc = el + 7
    s_dc = np.ceil(mu_dc * p_dc + z * np.sqrt(p_dc * var_dc + mu_dc ** 2 * var_l))
    return s_store, s_dc


def base_stock_policy(state: dict) -> dict:
    ctx = state["context"]
    cache = ctx["_cache"]
    if "base_stock" not in cache:
        z = norm.ppf(ctx["target_fill_rate"])
        cache["base_stock"] = _base_stock_levels(ctx, z)
    s_store, s_dc = cache["base_stock"]

    transfers = np.maximum(0, s_store - state["store_on_hand"] - state["store_in_transit"])
    po = None
    if state["is_po_day"]:
        po = np.maximum(0, s_dc - state["dc_on_hand"] - state["dc_on_order"])
    return {"po": po, "transfers": transfers}


# Until the improved policy is written, `policy` is the baseline.
policy = base_stock_policy
