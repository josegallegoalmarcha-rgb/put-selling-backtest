# MACD common-candidate diagnostic — four frozen windows

## Purpose
This is a **diagnostic only**. It does not change V0.1 and is not a new strategy version.

The prior V0.1-vs-control comparison applied the 20-session cooldown separately to each signal stream. That can make V0.1 and control samples differ for reasons unrelated to MACD. Here we remove that distortion.

## Design
1. Build the same base candidate pool: stock MA50>MA200, allowed SPY regime, Slow Stochastic 14,3,3 cross above 20.
2. Apply **one common 20-session cooldown to the base candidate stream**.
3. Only after candidate selection, label each episode `MACD_PASS` or `MACD_FAIL` using the frozen V0.1 MACD rule.
4. Compare outcomes inside that common candidate pool.
5. Bootstrap confidence intervals resample **signal dates as clusters**, not individual stocks, to preserve same-day market clustering.

A negative `Δ Stop` favors MACD_PASS. Positive `Δ +3`, `Δ +5`, MFE and `Δ Ret20` favor MACD_PASS. Less-negative MAE also favors MACD_PASS.

## Summary by window and MACD label

|Window|Group|N|Stop %|+3%|+5%|MFE %|MAE %|Ret20 %|
|---|---|---|---|---|---|---|---|---|
|2015_2016|MACD_FAIL|132|45.45|46.21|30.30|3.98|-2.67|0.90|
|2015_2016|MACD_PASS|180|59.44|40.00|15.56|2.77|-3.12|-0.33|
|2022|MACD_FAIL|25|40.00|68.00|60.00|7.16|-4.26|0.89|
|2022|MACD_PASS|33|63.64|42.42|27.27|3.86|-5.45|-2.33|
|COVID_2020|MACD_FAIL|60|78.33|50.00|25.00|4.27|-7.93|-7.85|
|COVID_2020|MACD_PASS|70|65.71|61.43|35.71|5.61|-5.36|-4.69|
|GFC_2007_2009|MACD_FAIL|61|45.90|80.33|52.46|6.97|-3.32|1.34|
|GFC_2007_2009|MACD_PASS|99|35.35|66.67|49.49|6.53|-3.59|2.49|
|ALL_WINDOWS_POOLED|MACD_FAIL|278|52.16|56.47|36.69|4.98|-4.09|-0.90|
|ALL_WINDOWS_POOLED|MACD_PASS|382|54.71|51.05|29.06|4.36|-3.85|-0.57|

## Incremental MACD lift inside common candidates

|Window|Pass N|Fail N|Pass %|Δ Stop pp|Δ +3 pp|Δ +5 pp|Δ MFE pp|Δ MAE pp|Δ Ret20 pp|
|---|---|---|---|---|---|---|---|---|---|
|2015_2016|180|132|57.69|13.99|-6.21|-14.75|-1.20|-0.45|-1.23|
|2022|33|25|56.90|23.64|-25.58|-32.73|-3.30|-1.18|-3.22|
|COVID_2020|70|60|53.85|-12.62|11.43|10.71|1.35|2.57|3.16|
|GFC_2007_2009|99|61|61.88|-10.55|-13.66|-2.96|-0.44|-0.27|1.15|
|ALL_WINDOWS_POOLED|382|278|57.88|2.55|-5.43|-7.63|-0.62|0.24|0.32|

## Cluster-bootstrap 95% intervals for key lifts

Intervals are descriptive robustness checks, not formal causal proof.

|Window|ΔStop low|ΔStop high|Δ+3 low|Δ+3 high|ΔRet20 low|ΔRet20 high|
|---|---|---|---|---|---|---|
|2015_2016|-0.31|27.31|-19.35|7.60|-2.55|0.07|
|2022|-2.02|51.12|-52.29|-4.24|-8.67|2.28|
|COVID_2020|-31.03|7.83|-4.05|31.95|-2.72|7.88|
|GFC_2007_2009|-25.84|5.82|-28.53|0.03|-1.57|4.07|
|ALL_WINDOWS_POOLED|-8.38|12.67|-13.88|3.21|-1.50|2.27|

## Guardrails
- Do not use this diagnostic to rewrite any historical trade.
- Do not promote or delete MACD based on one window alone.
- If MACD lift changes sign materially across windows, treat it as regime-dependent until a pre-specified real-time discriminator is validated.
- The next step after interpreting this report is the pre-planned bull-control windows 2013–2014 and 2017, still with frozen V0.1.
