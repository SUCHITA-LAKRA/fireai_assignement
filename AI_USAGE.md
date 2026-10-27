# AI Usage

## Tools used

| Tool | Used for |
|---|---|

## One place the AI was wrong, and how I caught it

_To be filled in._

**Example (Project A, step 1):** the first version of `data_prep.py`, written with an AI assistant, stored daily sales as `int16`. Pandas keeps that type when summing, so a daily total above 32,767 units would silently wrap to a wrong (even negative) number. Caught while testing the closure check on sample data — the summed column was still `int16`. Fixed by using `int32`.
