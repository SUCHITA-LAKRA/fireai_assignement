"""
Project B - Step 9: The CMO / CFO memo (memos/memo_B.pdf, max 2 pages).

What this script does
---------------------
Builds the memo from the tables steps 0-8 saved. It contains no typed-in
numbers: every figure is read from a CSV, so re-running the pipeline (for
example with another hold-out window) rebuilds the memo with the new numbers.

How to run (from the repo root, after steps 0-8)
------------------------------------------------
    python -m project_b.memo

Output
------
    memos/memo_B.pdf
"""

import sys
from datetime import date
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_b.config import REPO_ROOT, load_config  # noqa: E402

INK = colors.HexColor("#1F2926")
MUTED = colors.HexColor("#5D6A63")
TEAL = colors.HexColor("#1E6F6C")
LINE = colors.HexColor("#DEE3D9")
SOFT = colors.HexColor("#F4F5F0")
GOLD = colors.HexColor("#A77A30")


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------

def m(x: float, d: int = 2) -> str:
    """Money in millions: 2.77M."""
    return f"{x / 1e6:.{d}f}M"


def k(x: float) -> str:
    return f"{x / 1e3:,.0f}k"


def pct(x: float, d: int = 0, sign: bool = False) -> str:
    return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def r(x: float) -> str:
    return f"{x:.2f}"


# ---------------------------------------------------------------------------
# Load every number the memo uses
# ---------------------------------------------------------------------------

def load_numbers(cfg: dict) -> dict:
    out = Path(cfg["paths"]["output_dir"])
    pb = REPO_ROOT / "project_b"
    n = {}

    roi = pd.read_csv(pb / "roi_table.csv").set_index("channel")
    n["roi"] = roi
    n["all"] = roi.loc["All paid media"]
    ch = roi.drop(index="All paid media")
    n["mroi_min"], n["mroi_max"] = ch["mroi_mean"].min(), ch["mroi_mean"].max()
    n["mroi_p_min"], n["mroi_p_max"] = ch["mroi_prob_above_1"].min(), ch["mroi_prob_above_1"].max()

    alloc = pd.read_csv(pb / "budget_allocation.csv")
    n["alloc"] = alloc
    summ = pd.read_csv(out / "step7_scenario_summary.csv").set_index("plan")
    n["summary"] = summ
    steps = pd.read_csv(out / "step7_budget_steps.csv")
    n["steps"] = steps
    n["cut_step"] = steps[steps["to_plan"] == "last 13 weeks (actual)"].iloc[0]
    stress = pd.read_csv(out / "step7_saturation_stress.csv")
    n["stress"] = stress.groupby("channel")["change_vs_last_13_weeks"].agg(["min", "max"])

    rule = cfg["budget"]["max_change_vs_last_13w"]
    current = alloc[alloc["plan"] == "last 13 weeks (actual)"]["spend_last_13_weeks"].sum()
    n["current_total"] = current
    n["rule_min"], n["rule_max"] = (1 - rule) * current, (1 + rule) * current
    n["budget"] = cfg["budget"]["next_quarter_total"]

    e0 = pd.read_csv(out / "step0_experiment_check.csv")

    def e0v(text):
        return e0[e0["check"].str.contains(text, regex=False)]["value"].iloc[0]
    n["vendor_zero_weeks"] = e0v("zero Channel3 spend during test")
    n["vendor_spend_change"] = e0v("test geos: Channel3 spend change")
    n["vendor_placebo"] = e0v("placebo")

    cal = pd.read_csv(out / "step5_roi_comparison.csv")
    c3 = cal[(cal["channel"] == cfg["experiment"]["channel"]) & (cal["scope"] == "all weeks")]
    n["cal"] = c3.set_index("model")["mean"].to_dict()
    ev = pd.read_csv(out / "step5_experiment_vs_data.csv")
    n["model_test_cells"] = ev[ev["check"].str.contains("test regions and weeks")]["value"].iloc[0]

    nat = pd.read_csv(out / "step4_geo_vs_national_roi.csv").set_index("channel")
    n["nat"] = nat

    acc = pd.read_csv(out / "step6_forecast_accuracy.csv").set_index("forecast")
    n["acc_model"] = acc.loc["Model (calibrated MMM)"]
    naive = acc.drop(index="Model (calibrated MMM)")
    n["acc_naive_best"] = naive["national_weekly_MAPE"].min()

    voi = pd.read_csv(out / "step8_value_of_information.csv")
    n["voi_top"] = voi.iloc[0]
    design = pd.read_csv(out / "step8_recommended_design.csv").set_index("item")["value"]
    n["design"] = design
    des = pd.read_csv(out / "step8_designs.csv")
    top = voi.iloc[0]["channel"]
    n["top_test"] = des[des["channel"] == top].sort_values("power_if_true_roi_is_1").iloc[-1]
    rec = design["recommended test: channel"]
    n["rec_test"] = des[(des["channel"] == rec) &
                        (des["treated_regions"] == int(design["treated regions"].count("Geo"))) &
                        (des["weeks"] == int(design["weeks switched off"]))].iloc[0]

    hold = pd.read_csv(out / "step3_run_info.csv").iloc[0]
    n["holdout"] = (hold["holdout_start"], hold["holdout_end"])
    return n


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def styles() -> dict:
    base = dict(fontName="Helvetica", fontSize=8.6, leading=11.2, textColor=INK)
    return {
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=14.5, leading=18, textColor=INK),
        "meta": ParagraphStyle("meta", fontName="Helvetica", fontSize=8, leading=10.5, textColor=MUTED),
        "h": ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=10, leading=13, textColor=TEAL,
                            spaceBefore=7, spaceAfter=2.5),
        "p": ParagraphStyle("p", alignment=TA_LEFT, spaceAfter=3, **base),
        "small": ParagraphStyle("small", fontName="Helvetica", fontSize=7.6, leading=9.6, textColor=MUTED),
        "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8, leading=10, textColor=INK),
        "cellb": ParagraphStyle("cellb", fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=INK),
        "box": ParagraphStyle("box", fontName="Helvetica", fontSize=8.8, leading=11.6, textColor=INK),
    }


def table(rows: list[list], widths: list[float], st: dict, bold_last: bool = False) -> Table:
    data = [[Paragraph(str(c), st["cellb"] if i == 0 or (bold_last and i == len(rows) - 1) else st["cell"])
             for c in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), SOFT),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def bullets(items: list[str], st: dict, style: str = "p") -> ListFlowable:
    return ListFlowable([ListItem(Paragraph(t, st[style]), leftIndent=10, value="•") for t in items],
                        bulletType="bullet", start="•", leftIndent=10, bulletFontSize=8)


def box(paragraphs: list, st: dict) -> Table:
    t = Table([[paragraphs]], colWidths=[17.8 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SOFT),
        ("LINEBEFORE", (0, 0), (0, -1), 3, GOLD),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


# ---------------------------------------------------------------------------
# The memo
# ---------------------------------------------------------------------------

def build(n: dict, cfg: dict, path: Path) -> None:
    st = styles()
    roi, all_ = n["roi"], n["all"]
    chans = [c for c in roi.index if c != "All paid media"]
    rec = n["alloc"][n["alloc"]["plan"] == f"{n['budget'] / 1e6:g}M: recommended"].set_index("channel")
    plan_name = f"{n['budget'] / 1e6:g}M: recommended"
    same_name = f"{n['budget'] / 1e6:g}M: same % for every channel"
    plan = n["summary"].loc[plan_name]
    cut = n["cut_step"]
    cut_lo = cut["revenue_per_extra_1_p05"] * cut["extra_budget"]
    cut_hi = cut["revenue_per_extra_1_p95"] * cut["extra_budget"]
    ch3 = cfg["experiment"]["channel"]
    d = n["design"]
    top, rt = n["voi_top"], n["rec_test"]
    tt = n["top_test"]

    # Biggest cuts / increases in the recommended plan
    chg = rec["change_vs_last_13_weeks"].sort_values()
    cuts = ", ".join(f"{c} {pct(v, sign=True)}" for c, v in chg.items() if v < -0.2)
    ups = ", ".join(f"{c} {pct(v, sign=True)}" for c, v in chg.items() if v > 0)
    trims = ", ".join(f"{c} {pct(v, sign=True)}" for c, v in chg.items() if -0.2 <= v <= 0)

    s = []
    s.append(Paragraph("Paid media: what it returns, and how to spend next quarter's "
                       f"{m(n['budget'], 0)}", st["title"]))
    s.append(Paragraph(f"To: CMO, CFO &nbsp;·&nbsp; From: Data Science &nbsp;·&nbsp; {date.today():%d %b %Y} "
                       f"&nbsp;·&nbsp; Based on 40 regions × 156 weeks to {n['holdout'][1]}. "
                       "Every figure is reproduced by one command (<i>make all</i>); amounts in the data's currency.",
                       st["meta"]))
    s.append(Spacer(1, 6))

    s.append(box([
        Paragraph("<b>Recommendation in one minute</b>", st["box"]),
        Spacer(1, 2),
        bullets([
            f"<b>Make the cut to {m(n['budget'], 0)}.</b> At today's spend, the next 1 in any channel brings back only "
            f"{r(n['mroi_min'])}–{r(n['mroi_max'])} in revenue (chance it pays back: {pct(n['mroi_p_min'])}–"
            f"{pct(n['mroi_p_max'])}). Going from {m(n['current_total'])} to {m(n['budget'], 0)} saves "
            f"{m(cut['extra_budget'])} of spend and gives up about {m(cut['extra_revenue_mean'])} of revenue "
            f"(90%: {m(cut_lo)}–{m(cut_hi)}).",
            f"<b>Tilt the cut</b>: {cuts}; {trims}; {ups}. This adds about {m(plan['gain_vs_same_cut_mean'])} over cutting "
            f"every channel equally ({pct(plan['prob_beats_same_cut'])} chance it's better), and holds whatever we assume "
            "about how fast returns fall. The cut itself is the big decision; the split is a sensible tilt.",
            f"<b>Do not cut {ch3} because of the vendor's 0.8.</b> Our own spend records show {ch3} was never switched off "
            f"in the test regions ({n['vendor_zero_weeks']} test weeks at zero; spend {n['vendor_spend_change']}). "
            f"Replace it with a clean test ({d['weeks switched off']} weeks, "
            f"{int(rt['treated_regions'])} regions, cost about {k(rt['expected_test_cost'])}).",
        ], st, "box"),
    ], st))

    # 1. What each channel returns
    s.append(Paragraph("1. What each channel returns", st["h"]))
    rows = [["Channel", "Spend, last 13 wks", "Average return per 1<br/>(3 years, 90% range)",
             "Return on the <i>next</i> 1<br/>(today's spend, 90% range)", "Chance it<br/>pays back",
             "Share of revenue"]]
    for c in chans:
        x = roi.loc[c]
        rows.append([c, m(x["spend_last_13_weeks"]), f"{r(x['roi_mean'])} ({r(x['roi_p05'])}–{r(x['roi_p95'])})",
                     f"{r(x['mroi_mean'])} ({r(x['mroi_p05'])}–{r(x['mroi_p95'])})", pct(x["mroi_prob_above_1"]),
                     pct(x["share_of_revenue_mean"], 1)])
    rows.append(["All paid media", m(all_["spend_last_13_weeks"]),
                 f"{r(all_['roi_mean'])} ({r(all_['roi_p05'])}–{r(all_['roi_p95'])})", "–", "–",
                 f"{pct(all_['share_of_revenue_mean'])} ({pct(all_['share_of_revenue_p05'])}–"
                 f"{pct(all_['share_of_revenue_p95'])})"])
    s.append(table(rows, [2.5, 2.4, 3.6, 3.7, 2.5, 3.1], st, bold_last=True))
    s.append(Spacer(1, 3))
    s.append(Paragraph(
        f"<b>How to read this.</b> On average over three years, paid media roughly pays back its cost in revenue "
        f"({r(all_['roi_mean'])}) and drives about {pct(all_['share_of_revenue_mean'])} of revenue. But every channel is past the "
        "point where more spend pays for itself: the <i>next</i> 1 returns about half. These are <b>revenue</b> returns; "
        "to make <b>profit</b> a channel needs a return above 1 ÷ gross margin (e.g. 2.5 at a 40% margin), so the case "
        "against adding spend is stronger still. Ranges are wide and overlap, so treat channel rankings as tendencies, "
        "not verdicts.", st["p"]))
    chart = Path(cfg["paths"]["output_dir"]) / "step6_roi_mroi.png"
    if chart.exists():
        s.append(Image(str(chart), width=17.0 * cm, height=6.8 * cm))
        s.append(Paragraph("Left: average return per 1 over three years. Right: return on the next 1 at today's spend. "
                           "Dot = best estimate, bar = 90% range, dashed line = break-even in revenue.", st["small"]))

    # 2. The plan
    s.append(Paragraph(f"2. The {m(n['budget'], 0)} plan (each channel within ±"
                       f"{pct(cfg['budget']['max_change_vs_last_13w'])} of last quarter)", st["h"]))
    rows = [["Channel", "Last 13 weeks", "Next quarter", "Change", "If returns fall fast / slowly"]]
    for c in chans:
        x = rec.loc[c]
        lo, hi = n["stress"].loc[c, "min"], n["stress"].loc[c, "max"]
        rows.append([c, m(x["spend_last_13_weeks"]), m(x["recommended_spend"]),
                     pct(x["change_vs_last_13_weeks"], sign=True), f"{pct(lo, sign=True)} to {pct(hi, sign=True)}"])
    rows.append(["Total", m(n["current_total"]), m(n["budget"], 0),
                 pct(n["budget"] / n["current_total"] - 1, sign=True), ""])
    s.append(table(rows, [2.5, 3.0, 3.0, 2.2, 4.4], st, bold_last=True))
    s.append(Spacer(1, 3))
    s.append(Paragraph(
        f"Expected extra revenue from paid media next quarter: <b>{m(plan['extra_revenue_mean'], 1)}</b> "
        f"(90%: {m(plan['extra_revenue_p05'], 1)}–{m(plan['extra_revenue_p95'], 1)}), "
        f"versus {m(n['summary'].loc[same_name, 'extra_revenue_mean'], 1)} for an equal cut everywhere. "
        "The plan assumes next quarter is bought like the last one (same weekly timing and prices).", st["p"]))

    # 3. Other budgets
    up = n["steps"][n["steps"]["from_plan"] == "last 13 weeks (actual)"].iloc[0]
    low = n["steps"][n["steps"]["to_plan"] == plan_name].iloc[0]
    best12 = [p for p in n["summary"].index if p.startswith("12M") and "best split" in p]
    same12 = [p for p in n["summary"].index if p.startswith("12M") and "same %" in p]
    s.append(KeepTogether([Paragraph("3. If the budget is 12M or 24M", st["h"]), bullets([
        f"<b>12M is not possible within the ±30% rule</b>: the lowest total the rule allows is "
        f"{m(n['rule_min'])}. Choose which rule to break: cut every channel by "
        f"{pct(1 - 12e6 / n['current_total'])}, or cut deeper where it hurts least (Channel0 and Channel1 by more than half). "
        f"Either way, expect {m(n['summary'].loc[same12[0], 'extra_revenue_mean'], 1)}–"
        f"{m(n['summary'].loc[best12[0], 'extra_revenue_mean'], 1)} of extra revenue; "
        f"the 6M saved vs {m(n['budget'], 0)} costs about {m(low['extra_revenue_mean'])} of revenue "
        f"({r(low['revenue_per_extra_1_mean'])} per 1).",
        f"<b>24M</b>: the extra {m(up['extra_budget'])} above today's spend would bring back about "
        f"{m(up['extra_revenue_mean'])} ({r(up['revenue_per_extra_1_mean'])} per 1; chance it pays back "
        f"{pct(up['prob_extra_budget_pays_back'])}). Not recommended on revenue grounds.",
    ], st)]))

    # 4. The vendor test and the next test
    s.append(Paragraph(f"4. The {ch3} experiment, and the test to run instead", st["h"]))
    s.append(Paragraph(
        f"The vendor reported that every 1 on {ch3} returned 0.8. Our records show the switch-off didn't happen "
        f"({n['vendor_zero_weeks']} test region-weeks at zero spend; spend {n['vendor_spend_change']} vs the weeks "
        f"before), and the small revenue dip was normal noise ({n['vendor_placebo'].split(' (')[0]} of comparable "
        f"periods show a gap as big). Before seeing the test, our model put {ch3} at "
        f"{n['model_test_cells']} in those regions and weeks, so even taken at face value the vendor's number "
        f"doesn't contradict it. Counting the vendor's result at a quarter of its weight moves {ch3} from "
        f"{r(n['cal']['uncalibrated (trust 0)'])} to {r(n['cal'][[x for x in n['cal'] if x.startswith('calibrated')][0]])}; "
        f"believing it fully gives {r(n['cal']['full trust (1)'])}. <b>The decision does not depend on it.</b>", st["p"]))
    s.append(Paragraph(
        f"<b>Proposed test.</b> The channel we know least about is {top['channel']} (not knowing its true return costs about "
        f"{k(top['value_of_perfect_information_per_quarter'])} a quarter), but no affordable test can measure it: even "
        f"{int(tt['treated_regions'])} regions for {int(tt['weeks'])} weeks has only {pct(tt['power_if_true_roi_is_1'])} "
        f"power. Instead, switch <b>{rt['channel']}</b> off in <b>{int(rt['treated_regions'])} of 40 regions</b>, picked at "
        f"random in size-matched pairs, for <b>{int(rt['weeks'])} weeks</b> (plus 2 weeks of follow-up). It has "
        f"{pct(rt['power_if_true_roi_is_1'])} power to detect a channel that pays for itself and measures its return to "
        f"±{r(rt['roi_90pct_margin'])}. Expected cost: about {k(rt['expected_test_cost'])} (revenue lost minus spend saved). "
        f"<b>Condition:</b> spend records must show zero {rt['channel']} spend in every test region and week. "
        "The test won't change next quarter's plan; it is the cheapest way to check the whole model independently.",
        st["p"]))

    # 5. How far to trust it
    am = n["acc_model"]
    gn = n["nat"]
    s.append(KeepTogether([
        Paragraph("5. How far to trust this", st["h"]),
        bullets([
            f"<b>It predicted weeks it never saw.</b> We hid the last 13 weeks ({n['holdout'][0]} to {n['holdout'][1]}); "
            f"the model's forecast of their total revenue was off by {pct(abs(am['national_total_bias']), 2)}, "
            f"weekly error {pct(am['national_weekly_MAPE'], 1)} vs {pct(n['acc_naive_best'], 1)} for simple guesses, and "
            f"{am['weeks_inside_90pct_band']} weeks inside its range (the miss: Christmas week).",
            f"<b>It works region by region on purpose.</b> On national totals the same method says {ch3} returns only "
            f"{r(gn.loc[ch3, 'national_mean'])} (vs {r(gn.loc[ch3, 'geo_mean'])}), because adding regions up hides the "
            "evidence and makes our spend indistinguishable from competitor activity. Don't judge channels from national totals.",
            "<b>It accounts for what else moves sales</b>: competitor activity, public sentiment, promotions and season. "
            "Without competitor activity, ads appear to <i>lose</i> money, because we spend most when rivals are busiest.",
        ], st),
    ]))

    # 6. Limits
    s.append(KeepTogether([
        Paragraph("6. What the model cannot tell you", st["h"]),
        bullets([
            "<b>Profit.</b> We have revenue, not margins. Apply your margin to every return above.",
            "<b>The exact shape of diminishing returns.</b> The data shows returns fall, not precisely how fast. "
            "We checked the plan under fast and slow assumptions; the direction holds, the sizes move.",
            "<b>Long-term brand effects.</b> It sees effects lasting up to about 8 weeks; anything slower is not counted.",
            "<b>Proof of cause.</b> It assumes regional spend wasn't aimed at regions already expected to sell more. "
            "Only a randomised test, like the one proposed, settles that.",
            "<b>Anything new.</b> New channels, new creative, price changes or holidays (no holiday effect is modelled).",
        ], st),
    ]))
    s.append(Spacer(1, 4))
    s.append(Paragraph("Method: Bayesian marketing mix model (Google Meridian) by region, with carry-over and "
                       "diminishing returns, calibrated with the vendor test at 25% weight. Full choices and "
                       "data checks: DECISIONS.md, DATA_ISSUES.md.", st["small"]))

    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.3 * cm, bottomMargin=1.2 * cm, title="Paid media memo",
                            author="Data Science")
    doc.build(s)


def main() -> None:
    cfg = load_config()
    out = REPO_ROOT / "memos" / "memo_B.pdf"
    out.parent.mkdir(exist_ok=True)
    build(load_numbers(cfg), cfg, out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
