# Decision Log

Every major modelling choice, the alternatives considered, and why they were rejected. (Max 3 pages.)

## Project A

| # | Decision | Other options | Why not |
|---|---|---|---|
| A1 | All assumptions in `config/project_a.yaml` | Numbers written inside the code | Assumptions will change live in the interview; one file makes that safe |
| A2 | A stockout is a run of low-sales days that is very unlikely for **that item** | A fixed rule, e.g. "7 zero days in a row" | 3 zero days mean nothing for a slow item but a lot for an item selling 100 a day |
| A3 | Zero runs longer than 56 days = item removed, demand 0 | Fill every run with the normal rate | Months with no sales most likely mean the item left the shelf |
| A4 | Demand correction uses only days up to `--cutoff`; test days are corrected separately, only for the simulator | One correction over all days | Using test days for training would break the cutoff rule |
| A5 | Simulator: stockout days in the test period get a random draw around the normal rate, new for every run | Use test sales as demand | Test sales are capped by Walmart's own stockouts |
| A6 | Baseline = textbook base-stock (order up to a target level) at each store and the DC, 95% service, from the last 56 days | Tune the baseline on the test period | It is the simple baseline the brief asks for; tuning on test days would be cheating |

## Project B

Short version. Full reasoning and numbers: `docs/DECISIONS_PROJECT_B_DETAIL.md` (IDs in the first column).

| Detail | Decision | Other options | Why not |
|---|---|---|---|
| B4 | Check the vendor test against our own data first | Take the 0.8 as given | Spend never dropped in the test geos (B1 in DATA_ISSUES) |
| B5–B9 | Choose model inputs from a causal graph: competitor sales and sentiment as controls; promo and organic as treatments; impressions as the media input | Put every column in; leave them out | Leaving them out makes ads look harmful; spend and impressions are copies |
| B15–B19 | Google Meridian, same ROI prior for every channel (median 1.22) | PyMC-Marketing or hand-written model; per-channel priors | Meridian has geo levels, adstock, saturation, calibration and an optimiser built in; we have no channel-specific evidence before calibration |
| B24–B26 | Trust the geo model over the national one | National model; average the two | National has 40× less data and loses most of the spend variation; it also fits the hold-out worse |
| B27–B29 | Give the vendor test **25% weight** as a prior on Channel3 ROI in the test weeks | Full trust; ignore it | The test likely did not happen as described. The answer barely moves anyway (1.08 / 1.05 / 0.97 at 0 / 25 / 100%) |
| B30–B37 | Budget: use marginal ROI on last-13-week spend, keep every channel within ±30%, and lead with the budget level, not the split | Meridian's default bounds; lead with the split | The default bounds break the ±30% rule. The split adds little (66% chance it beats an equal cut); the budget level matters more |
| B38–B41 | Next test: switch Channel3 off in 20 size-balanced geos for 12 weeks | Test Channel4 (most costly uncertainty) | Channel4's best test has only 41% power; Channel3's has 83% |
| B42–B43 | Memo built by code from saved tables | Write it by hand | One command reproduces every number |
