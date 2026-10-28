# Data Issues Log

Each problem found, how it was detected, and how it was handled.

## Project A — M5 (CA FOODS)

Numbers are for cutoff d_1885 and come from `python -m project_a.demand_recovery`.

| # | Problem | How I found it | What I did | Why it matters |
|---|---|---|---|---|
| A1 | **Many items start selling years after d_1.** Before launch they show 0 sales and no price. | Looked at the first day with a price and the first day with a sale | Days before the first sale count as "not listed", not as zero demand | Otherwise years of fake zeros pull the averages down |
| A2 | **Stockouts look like zero demand.** A fast seller can show weeks of 0 (or only 1–3 units) while it still has a price. | Checked each run of low-sales days against how much the item normally sells around it. If the run is very unlikely by chance (< 1%), it is a stockout | On stockout days, demand = the item's normal sales rate | About 10% of days are stockouts. **Raw sales are about 9–10% below true demand** |
| A3 | **Some zero runs last months or years.** The item is most likely removed from the shelf. | Length of the flagged runs | Runs longer than 56 days count as "removed": demand 0 | Filling them in would invent about 1 million units of demand |
| A4 | **All stores close on Christmas Day.** | Store sales fall to almost zero once a year | Christmas Day: demand 0 | Small, but would otherwise look like a one-day stockout for every item |
| A5 | **CA_2 had an 8-month dip, then grew a lot** (Oct 2014 – May 2015, sales down ~25%; afterwards its share of CA rose from ~16% to ~24%). Other stores show nothing like it. | CA_2's share of California sales over time | During the dip, sales are scaled back up to CA_2's normal share. The later growth is treated as real | Forecasts should rely on recent CA_2 data |
| A6 | **The test period has stockouts too**, so test sales are not test demand. | Same check, run on all days | The simulator uses recorded sales on normal days and a random draw around the normal rate on stockout days | A policy is not rewarded for sales Walmart itself could not make |

**What this bias costs:** the same base-stock policy, planned on raw sales instead of corrected demand, costs **8.1k more (+8.8%) over the 8 test weeks** and reaches **1.5 points lower fill rate** (93.1% vs 94.6%).

**Limits of the stockout check:** on simulated data with no stockouts it still flags about 3% of days, mostly slow sellers. This inflates demand by about 1.6%, so the true gap is about 8–10%. It also misses many short stockouts of very slow items.

## Project B — Meridian sample data

Numbers come from `python -m project_b.data_prep` and `project_b.dag` (`outputs/project_b/step0_*.csv`, `step1_*.csv`).

| # | Problem | How I found it | What I did | Why it matters |
|---|---|---|---|---|
| B1 | **The Channel3 test does not show in the data.** Spend should be 0 in Geo0–Geo7 for 8 weeks from 2023-06-05, but only 2 of 64 test geo-weeks are 0, and spend went **up 18%**. The revenue dip in those geos is normal noise (half of all other 8-week windows show a gap as big). | Compared Channel3 spend and revenue in test vs other geos, before vs during the test | Use the vendor's result only as a weak prior. Recommend running the test again | Main reason not to fully trust the experiment |
| B2 | **The test geos were not random:** 6 of 8 are among the 12 smallest geos. | Ranked geos by population | The next test uses random geos balanced by size | A result from small geos may not hold for big ones |
| B3 | **Spend and impressions are copies of each other.** Each channel has one fixed price per 1,000 impressions. | Spend ÷ impressions | Use impressions in the model and spend only for ROI | Using both would count the same thing twice |
| B4 | **Big geos dominate.** Population vs revenue correlation 0.97; biggest geo is 7.3× the smallest. | Correlations across geos | Model per person (Meridian scales by population) | Without this, size looks like an ad effect |
| B5 | **Spend moves with other drivers** (competitor sales, sentiment, organic, promo). | Correlations within the same week, training weeks only | Put all four in the model as controls or treatments | Without them, ads look like they *lose* money (−0.41 per 1 spent); with them, +0.18 |
| B6 | **Some channels have little data:** Channel2 is 0 in 64% of geo-weeks, and Channel2 and Channel3 move together (0.70). | Share of zero weeks; correlation of spend | Trust the geo model more than the national one; expect wide ranges | Wider, less certain ROIs for these channels |
| B7 | **`conversions` is an index, not people** (2–44 per person per week, not whole numbers). | Conversions ÷ population | Model revenue (conversions × revenue per conversion) | ROI comes out in money |

Otherwise the data is clean: 40 geos × 156 weeks, no missing values, no duplicates, no gaps. Budget note: the last 13 weeks' spend was 20.8M, so 18M is a 13% cut.
