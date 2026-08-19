# Independent validation — H1: short-term breadth × MACD

## Status
**Validation sample only. No strategy rule is changed and V0.2-C is not created by this test.**

## Frozen hypothesis from the discovery sample
> The incremental value of MACD should be greater when very few Dividend Aristocrats have a positive 5-session return, and should diminish as the short-term recovery becomes broad.

Primary state variable: `breadth_5d_positive_pct`.

Frozen breadth buckets: `<=25%`, `(25,50]`, `(50,75]`, `>75%`.

## Independent validation windows
- 2018 — late-cycle volatility / Q4 sell-off.
- 2019 — broad bull recovery.
- 2021 — post-COVID bull market.
- 2023–2024 — post-2022 recovery / modern bull market.

These windows were not used to discover H1.

## Coverage

|Window|Universe|Loaded|
|---|---:|---:|
|VALIDATION_2018|53|52|
|VALIDATION_2019|57|55|
|VALIDATION_2021|65|63|
|VALIDATION_2023_2024|67|66|

## Overall MACD lift — validation sample

Negative ΔStop favors MACD_PASS. Positive Δ+3, Δ+5 and ΔRet20 favor MACD_PASS.

|Window|N|Pass N|Fail N|ΔStop pp|Δ+3 pp|Δ+5 pp|ΔRet20 pp|
|---|---:|---:|---:|---:|---:|---:|---:|
|ALL_VALIDATION_POOLED|1001|621|380|-9.62|5.23|5.27|0.65|
|VALIDATION_2018|167|96|71|-16.08|1.94|-0.84|0.96|
|VALIDATION_2019|136|67|69|-19.12|16.09|2.36|1.11|
|VALIDATION_2021|230|150|80|-20.67|15.50|18.42|2.03|
|VALIDATION_2023_2024|468|308|160|0.75|-2.13|0.79|-0.03|

## Primary H1 test — pooled fixed breadth buckets

|Breadth 5d+|N|Pass N|Fail N|ΔStop pp|Δ+3 pp|Δ+5 pp|ΔRet20 pp|
|---|---:|---:|---:|---:|---:|---:|---:|
|<=25|198|98|100|-5.96|-1.00|2.73|0.72|
|(25,50]|290|164|126|-3.78|8.54|2.09|0.57|
|(50,75]|341|227|114|-9.00|5.93|4.54|-0.26|
|>75|172|132|40|-21.89|1.36|17.20|3.29|

## Cross-window consistency — low breadth `<=25%`

A validation window is eligible for this check only if the low-breadth bucket contains at least 5 MACD_PASS and 5 MACD_FAIL candidates.

|Bucket|Eligible windows|MACD lower stop|MACD higher +3|MACD higher +5|MACD higher Ret20|Median ΔRet20|
|---|---:|---:|---:|---:|---:|---:|
|<=25|4|3|3|2|3|0.97|

## Continuous interaction

Model: `outcome ~ MACD + breadth_z + MACD×breadth_z + window fixed effects`, with standard errors clustered by signal date.

|Outcome|Interaction per 1 SD breadth|SE|p-value|N|
|---|---:|---:|---:|---:|
|ret20|0.4894|0.4505|0.2773|1001|
|stop|-0.0752|0.0365|0.0393|1001|
|hit3|0.0223|0.0365|0.5412|1001|
|hit5|0.0556|0.0346|0.1079|1001|

## Cluster-bootstrap 95% intervals

|Breadth bucket|N|ΔRet20 low|ΔRet20 high|Δ+5 low|Δ+5 high|ΔStop low|ΔStop high|
|---|---:|---:|---:|---:|---:|---:|---:|
|<=25|198|-1.30|2.68|-11.97|17.74|-21.35|9.22|
|>75|172|1.04|5.51|-0.24|34.42|-39.63|-2.69|

## Pre-specified directional verdict
Four criteria were frozen before seeing the validation results:
1. MACD Ret20 lift is positive in pooled breadth `<=25%`.
2. MACD Ret20 lift in `<=25%` is greater than in `>75%`.
3. The continuous MACD×breadth interaction for Ret20 is negative.
4. At least 3 eligible windows exist in `<=25%`, and at least 75% of them show positive MACD Ret20 lift.

- `low_breadth_pooled_ret20_positive`: **True**
- `low_breadth_ret20_greater_than_high_breadth`: **False**
- `continuous_interaction_ret20_negative`: **False**
- `low_breadth_cross_window_consistency`: **True**

### Final validation status: **PARTIAL_NOT_ENOUGH_FOR_V02C**

Interpretation rule:
- `VALIDATED_DIRECTIONALLY`: H1 may justify a separately specified V0.2-C experiment, but is not itself a trading rule.
- `PARTIAL_NOT_ENOUGH_FOR_V02C`: evidence is interesting but insufficient; do not create a breadth threshold rule.
- `FAILED_VALIDATION`: treat the discovery result as in-sample and do not promote H1.

## Guardrails
- No threshold search is permitted after this result.
- No alternate breadth horizon is tested here.
- No ROC/volatility combination is added.
- Positive Layer-1 results still do not prove short-put profitability.
