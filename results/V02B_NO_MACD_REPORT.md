# V0.2-B research — remove mandatory MACD vs frozen V0.1

## Status
**Research candidate only. V0.1 remains frozen.**

## Only rule change
`V0.2-B = stock MA50>MA200 + SPY GREEN/YELLOW + Slow Stochastic 14,3,3 cross above 20`.

The frozen V0.1 MACD requirement is **not** used as an entry gate in V0.2-B. MACD is still calculated and recorded as a diagnostic variable. No other rule changes.

Operationally, V0.2-B is the former `STOCH_ONLY_CONTROL` promoted to a research candidate; the old control files remain the frozen benchmark.

## Operational V0.1 vs V0.2-B

|Window|V0.1 N|V0.2B N|Expansion %|Extra N|Δ Stop pp|Δ +3 pp|Δ +5 pp|Δ MFE pp|Δ MAE pp|Δ Ret20 pp|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|BEAR_2022|38|58|152.63|20|-14.97|8.71|9.80|0.71|0.48|1.70|
|BULL_2013_2014|314|447|142.36|133|3.27|-3.97|-5.44|-0.39|-0.14|-0.37|
|BULL_2017|111|172|154.95|61|-9.17|8.44|6.96|0.80|0.05|0.81|
|CHOP_2015_2016|201|312|155.22|111|-4.68|1.33|3.88|0.32|0.12|0.63|
|COVID_2020|96|129|134.38|33|14.80|-6.69|-12.74|-2.30|-1.17|-4.03|
|GFC_2007_2009|112|160|142.86|48|4.55|4.02|0.62|0.06|0.25|-0.45|

## Common-candidate MACD lift

A negative ΔStop favors MACD_PASS. Positive Δ+3, Δ+5, ΔMFE, ΔMAE (less-negative MAE), and ΔRet20 favor MACD_PASS.

|Window|Pass N|Fail N|Pass %|Δ Stop pp|Δ +3 pp|Δ +5 pp|Δ MFE pp|Δ MAE pp|Δ Ret20 pp|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|ALL_WINDOWS_POOLED|730|548|57.12|1.96|-2.19|-3.16|-0.33|0.11|0.08|
|BEAR_2022|33|25|56.90|23.64|-25.58|-32.73|-3.30|-1.18|-3.22|
|BULL_2013_2014|248|199|55.48|-7.32|9.43|8.97|0.66|0.12|0.52|
|BULL_2017|101|71|58.72|22.33|-19.24|-17.35|-1.79|-0.21|-1.59|
|CHOP_2015_2016|180|132|57.69|13.99|-6.21|-14.75|-1.20|-0.45|-1.23|
|COVID_2020|69|60|53.49|-11.67|10.87|11.23|1.37|2.51|3.12|
|GFC_2007_2009|99|61|61.88|-10.55|-13.66|-2.96|-0.44|-0.27|1.15|

## Cluster-bootstrap 95% intervals — common candidates

|Window|ΔStop low|ΔStop high|Δ+3 low|Δ+3 high|Δ+5 low|Δ+5 high|ΔRet20 low|ΔRet20 high|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|BEAR_2022|-3.30|48.69|-51.03|-5.24|-55.24|-5.68|-8.68|2.51|
|BULL_2013_2014|-17.72|2.56|-0.90|20.48|-0.18|18.56|-0.49|1.57|
|BULL_2017|8.08|37.11|-34.29|-3.72|-30.39|-2.51|-2.97|-0.13|
|CHOP_2015_2016|-0.31|27.31|-19.35|7.60|-25.00|-3.52|-2.55|0.07|
|COVID_2020|-30.70|9.04|-5.70|32.17|-6.59|28.85|-2.94|7.82|
|GFC_2007_2009|-26.40|6.47|-28.03|1.17|-18.72|12.68|-1.46|4.14|
|ALL_WINDOWS_POOLED|-5.40|8.68|-8.46|4.18|-8.69|2.44|-0.97|1.34|

## Clustering / capacity diagnostic

Removing MACD can increase signal frequency and correlated assignment exposure. This table is therefore part of the acceptance test, not an afterthought.

|Window|Version|Signal days|Max same day|Days ≥3|Days ≥5|Days ≥10|Share signals on ≥5-signal days %|
|---|---|---:|---:|---:|---:|---:|---:|
|BEAR_2022|V0.1|18|8|5|1|0|21.05|
|BEAR_2022|V0.2B_NO_MACD|21|12|7|4|1|50.00|
|BULL_2013_2014|V0.1|132|11|39|19|1|39.49|
|BULL_2013_2014|V0.2B_NO_MACD|184|18|62|24|2|37.81|
|BULL_2017|V0.1|76|4|8|0|0|0.00|
|BULL_2017|V0.2B_NO_MACD|105|5|16|1|0|2.91|
|CHOP_2015_2016|V0.1|118|9|16|7|0|22.39|
|CHOP_2015_2016|V0.2B_NO_MACD|147|11|39|13|2|29.17|
|COVID_2020|V0.1|44|14|9|5|2|44.79|
|COVID_2020|V0.2B_NO_MACD|45|20|13|4|3|44.96|
|GFC_2007_2009|V0.1|61|7|16|4|0|20.54|
|GFC_2007_2009|V0.2B_NO_MACD|81|8|20|8|0|29.38|

## Frozen benchmark integrity

Both the re-run V0.1 and re-run V0.2-B are checked against the already-saved frozen `V0.1` and `STOCH_ONLY_CONTROL` summaries.

|Window|Rerun|Frozen row|Frozen found|N exact|Max metric diff|Close match|
|---|---|---|---|---|---:|---|
|GFC_2007_2009|V0.1|V0.1|Yes|True|0.0000|True|
|GFC_2007_2009|V0.2B_NO_MACD|STOCH_ONLY_CONTROL|Yes|True|0.0000|True|
|BULL_2013_2014|V0.1|V0.1|Yes|True|0.0000|True|
|BULL_2013_2014|V0.2B_NO_MACD|STOCH_ONLY_CONTROL|Yes|True|0.0000|True|
|CHOP_2015_2016|V0.1|V0.1|Yes|True|0.0000|True|
|CHOP_2015_2016|V0.2B_NO_MACD|STOCH_ONLY_CONTROL|Yes|True|0.0000|True|
|BULL_2017|V0.1|V0.1|Yes|True|0.0000|True|
|BULL_2017|V0.2B_NO_MACD|STOCH_ONLY_CONTROL|Yes|True|0.0000|True|
|COVID_2020|V0.1|V0.1|Yes|False|0.5906|False|
|COVID_2020|V0.2B_NO_MACD|STOCH_ONLY_CONTROL|Yes|False|0.5546|False|
|BEAR_2022|V0.1|V0.1|Yes|True|0.0000|True|
|BEAR_2022|V0.2B_NO_MACD|STOCH_ONLY_CONTROL|Yes|True|0.0000|True|

## Pre-specified interpretation guardrails
- Do not accept V0.2-B because pooled averages improve alone.
- Removing MACD should improve or stabilize performance across several windows, not merely one crisis.
- GFC, 2013–14 and the COVID rebound should not be materially damaged without compensating robustness elsewhere.
- Signal expansion and clustering count as costs: a higher-frequency candidate that creates larger correlated clusters may be worse at portfolio level even if mean directional metrics improve.
- The incremental `MACD_FAIL` candidates must be inspected for both saved winners and newly admitted losers.
- Positive Layer-1 performance still does not prove short-put profitability.
- No MACD replacement, ROC filter, threshold optimization or additional indicator is allowed inside this experiment.
