# AI Usage

## Tools used

| Tool | Used for |
|---|---|
| Claude (Anthropic) | Writing and debugging code, checking modelling choices, drafting the documentation |


## Where the AI was wrong, and how I caught it

### Project A — data prep (`project_a/data_prep.py`)

1. **Sales stored as `int16`.** The AI stored daily sales as `int16`. Pandas keeps that type when summing, so any daily total above 32,767 units would silently wrap to a wrong (even negative) number. I caught it while testing the store-closure check: the summed column was still `int16`. Fixed by using `int32`.

2. **Ran out of memory.** The AI's version reshaped the sales table (one column per day → one row per day) while the ID columns were still text. That required lot of memory. My system ran out of memory. I did not dig up much but my system ran out of memory

### Project B — data prep (`project_b/data_prep.py`)

1. **Blank-header column not dropped.** `geo_all_channels.csv` has an extra index column with no name. The AI's loader kept it as an `Unnamed` column. I spotted it and added a step that drops any `Unnamed` column when loading.

2. **Hold-out start date had to match a week exactly.** The AI's code assumed `--holdout-start` is exactly one of the week dates in the data. Any other date (for example a date in the middle of a week) gave a wrong hold-out window. Fixed by moving the date forward to the next week in the data, printing a note when it does, and warning if fewer than 13 weeks are left.
