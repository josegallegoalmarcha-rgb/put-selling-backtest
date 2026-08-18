# V0.2-A research — SPY ROC5 > 0 vs frozen V0.1

## Status
**Research candidate only. V0.1 remains the frozen benchmark.**

## Only rule change
`V0.2-A = V0.1 AND SPY_ROC5 > 0`, where `SPY_ROC5 = (SPY Close / SPY Close 5 sessions ago - 1) × 100`.

No MACD, Stochastic, MA50/MA200, regime, swing-low, entry, stop, cooldown or evaluation rule is changed.

## Why two comparisons are reported
1. **Operational:** V0.1 and V0.2-A each apply their own cooldown, which is how the actual rule sets would trade.
2. **Common-candidate diagnostic:** one cooldown is applied to the frozen V0.1 candidate stream first; candidates are then labelled ROC5_PASS/ROC5_FAIL. This isolates ROC5 from cooldown-path effects.

## Operational V0.1 vs V0.2-A

|Window|V0.1 N|V0.2A N|Retention %|Δ Stop pp|Δ +3 pp|Δ +5 pp|Δ MFE pp|Δ MAE pp|Δ Ret20 pp|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|BEAR_2022|38|27|71.05|-1.75|-0.29|-1.95|0.09|-0.39|-0.68|
|BULL_2013_2014|314|208|66.24|5.85|-6.18|-4.20|-0.32|-0.24|-0.56|
|BULL_2017|111|80|72.07|0.80|1.20|-1.68|-0.23|0.00|-0.39|
|CHOP_2015_2016|201|145|72.14|3.86|-4.74|-3.43|-0.33|0.01|0.01|
|COVID_2020|96|72|75.00|-8.68|6.94|11.81|1.09|1.55|3.09|
|GFC_2007_2009|112|72|64.29|1.29|-5.36|-5.56|-1.17|0.41|-0.34|

## Common-candidate ROC5 lift

A negative `Δ Stop` favors ROC5_PASS. Positive `Δ +3`, `Δ +5`, `Δ MFE`, `Δ MAE` (less-negative MAE), and `Δ Ret20` favor ROC5_PASS.

|Window|Pass N|Fail N|Pass %|Δ Stop pp|Δ +3 pp|Δ +5 pp|Δ MFE pp|Δ MAE pp|Δ Ret20 pp|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|ALL_WINDOWS_POOLED|571|301|65.48|10.76|-12.00|-9.93|-1.22|0.31|-0.60|
|BEAR_2022|26|12|68.42|-9.62|4.49|-2.56|0.71|-1.63|-2.52|
|BULL_2013_2014|194|120|61.78|19.53|-19.21|-14.16|-1.05|-0.68|-1.74|
|BULL_2017|77|34|69.37|9.13|5.58|-2.14|-0.74|0.06|-1.16|
|CHOP_2015_2016|139|62|69.15|7.21|-14.92|-11.42|-1.06|0.15|-0.08|
|COVID_2020|66|30|68.75|-13.64|13.33|24.85|0.90|5.15|7.65|
|GFC_2007_2009|69|43|61.61|7.45|-18.20|-20.76|-3.81|0.94|-1.56|

## Cluster-bootstrap 95% intervals — common candidates

Signal dates are resampled as clusters so that same-day market signal bursts are not treated as independent observations.

|Window|ΔStop low|ΔStop high|Δ+3 low|Δ+3 high|Δ+5 low|Δ+5 high|ΔRet20 low|ΔRet20 high|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|BEAR_2022|-42.43|28.08|-30.88|44.29|-41.18|28.98|-7.98|4.73|
|BULL_2013_2014|4.90|33.33|-34.61|-4.10|-29.63|1.32|-3.51|-0.07|
|BULL_2017|-14.06|33.75|-14.85|26.42|-18.16|13.53|-3.00|0.72|
|CHOP_2015_2016|-10.47|25.09|-31.26|2.32|-25.34|0.72|-1.82|1.59|
|COVID_2020|-46.35|42.69|-26.92|38.02|-18.82|50.02|-8.07|16.28|
|GFC_2007_2009|-16.67|28.80|-38.15|2.83|-41.33|0.75|-4.68|2.42|
|ALL_WINDOWS_POOLED|-0.19|21.05|-21.74|-2.42|-19.22|-0.40|-2.48|1.70|

## Frozen-benchmark integrity

The script re-runs V0.1 beside V0.2-A and compares the result with the already-saved frozen V0.1 summary files. This detects data-source drift, especially in Yahoo-adjusted historical data.

|Window|Frozen found|N exact|Max metric diff|Close match|
|---|---|---|---:|---|
|GFC_2007_2009|Yes|True|0.0000|True|
|BULL_2013_2014|Yes|True|0.0000|True|
|CHOP_2015_2016|Yes|True|0.0000|True|
|BULL_2017|Yes|True|0.0000|True|
|COVID_2020|Yes|False|0.5906|False|
|BEAR_2022|Yes|True|0.0000|True|

## Pre-specified interpretation guardrails
- Do **not** accept V0.2-A because the pooled average improves alone.
- The candidate should reduce transition/chop damage in 2015–16, COVID and 2022 without materially destroying GFC, 2013–14 or 2017.
- Signal retention matters: a filter that deletes most opportunities may improve averages while making the system economically irrelevant.
- The rejected-candidate file must be inspected for good rebounds destroyed by the filter, especially +3%/+5% before stop.
- A single crisis cluster must not determine acceptance.
- Positive Layer-1 performance still does not prove PUT-option profitability.
- MACD remains frozen in this experiment. Its role belongs to a later V0.2-B experiment only.
