# MASTER Layer-1 diagnostic — 2,279 common candidates

## Status
**Diagnostic only. No V0.1/V0.2 rule is changed.**

This consolidates the frozen **1,278 discovery candidates** and the **1,001 independent validation candidates**. The purpose is to decide whether Layer 1 is mature enough to justify testing PUT monetization in Layer 2, while preserving the observed regime heterogeneity.

## Integrity
- Discovery: **1,278** candidates.
- Validation: **1,001** candidates.
- Total: **2,279** candidates.
- Windows: **10**.
- Distinct signal dates/clusters: **1035**.

## 1. Absolute base-candidate behavior

The base stream is MA50>MA200 + SPY GREEN/YELLOW + Stochastic cross with common per-stock cooldown. These absolute results are **not an unconditional-market alpha test**.

|N|Stop %|+3 %|+5 %|MFE %|MAE %|Ret20 avg %|Ret20 median %|Ret20 >0 %|Ret20 q05 %|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|2279|48.84|54.10|35.54|4.55|-3.09|0.94|1.20|58.49|-9.95|

## 2. MACD incremental effect — pooled and replication

Negative ΔStop favors MACD_PASS. Positive Δ+3, Δ+5, ΔMFE, ΔMAE (less-negative MAE) and ΔRet20 favor MACD_PASS.

|Sample|N|Pass N|Fail N|ΔStop pp|Δ+3 pp|Δ+5 pp|ΔMFE|ΔMAE|ΔRet20 pp|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|ALL_2279|2279|1351|928|-2.87|0.92|0.51|0.01|0.07|0.34|
|DISCOVERY|1278|730|548|1.96|-2.19|-3.16|-0.33|0.11|0.08|
|VALIDATION|1001|621|380|-9.62|5.23|5.27|0.47|0.01|0.65|

## 3. MACD effect by window

|Window|N|Pass %|ΔStop pp|Δ+3 pp|Δ+5 pp|ΔRet20 pp|
|---|---:|---:|---:|---:|---:|---:|
|GFC_2007_2009|160|61.88|-10.55|-13.66|-2.96|1.15|
|BULL_2013_2014|447|55.48|-7.32|9.43|8.97|0.52|
|CHOP_2015_2016|312|57.69|13.99|-6.21|-14.75|-1.23|
|BULL_2017|172|58.72|22.33|-19.24|-17.35|-1.59|
|COVID_2020|129|53.49|-11.67|10.87|11.23|3.12|
|BEAR_2022|58|56.90|23.64|-25.58|-32.73|-3.22|
|VALIDATION_2018|167|57.49|-16.08|1.94|-0.84|0.96|
|VALIDATION_2019|136|49.26|-19.12|16.09|2.36|1.11|
|VALIDATION_2021|230|65.22|-20.67|15.50|18.42|2.03|
|VALIDATION_2023_2024|468|65.81|0.75|-2.13|0.79|-0.03|

## 4. SPY regime and descriptive window family

### By SPY regime

|Regime|N|ΔStop pp|Δ+3 pp|Δ+5 pp|ΔRet20 pp|
|---|---:|---:|---:|---:|---:|
|GREEN|2129|-3.72|1.09|1.03|0.26|
|YELLOW|150|6.20|3.85|-0.11|0.84|

### By descriptive window family

|Family|N|ΔStop pp|Δ+3 pp|Δ+5 pp|ΔRet20 pp|
|---|---:|---:|---:|---:|---:|
|BULL|1453|-3.68|3.85|4.24|0.21|
|CORRECTION_BEAR|537|5.68|-5.82|-12.43|-0.76|
|CRISIS|289|-13.77|-1.00|5.20|2.74|

## 5. Date-cluster bootstrap 95% intervals

|Group|Clusters|ΔStop low|high|Δ+3 low|high|Δ+5 low|high|ΔRet20 low|high|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|ALL_2279|1035|-7.83|2.03|-3.62|5.75|-3.64|4.69|-0.38|1.17|
|DISCOVERY|583|-5.29|8.80|-8.15|3.97|-8.84|2.60|-0.90|1.31|
|VALIDATION|452|-16.24|-2.77|-1.68|11.84|-0.96|11.66|-0.30|1.60|

## 6. Window-balanced robustness

This prevents large windows from dominating the signal-weighted pooled result.

|Window set|Metric|Windows|Favorable|Mean effect|Median|Min|Max|Sign-test p|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|ALL_WINDOWS|delta_stop_pp|10|6|-2.47|-8.94|-20.67|23.64|0.3770|
|ALL_WINDOWS|delta_hit3_pp|10|5|-1.30|-0.10|-25.58|16.09|0.6230|
|ALL_WINDOWS|delta_hit5_pp|10|5|-2.69|-0.02|-32.73|18.42|0.6230|
|ALL_WINDOWS|delta_ret20_pp|10|6|0.28|0.74|-3.22|3.12|0.3770|
|DISCOVERY_WINDOWS|delta_stop_pp|6|3|5.07|3.33|-11.67|23.64|0.6562|
|DISCOVERY_WINDOWS|delta_hit3_pp|6|2|-7.40|-9.94|-25.58|10.87|0.8906|
|DISCOVERY_WINDOWS|delta_hit5_pp|6|2|-7.93|-8.86|-32.73|11.23|0.8906|
|DISCOVERY_WINDOWS|delta_ret20_pp|6|3|-0.21|-0.36|-3.22|3.12|0.6562|
|VALIDATION_WINDOWS|delta_stop_pp|4|3|-13.78|-17.60|-20.67|0.75|0.3125|
|VALIDATION_WINDOWS|delta_hit3_pp|4|3|7.85|8.72|-2.13|16.09|0.3125|
|VALIDATION_WINDOWS|delta_hit5_pp|4|3|5.18|1.57|-0.84|18.42|0.3125|
|VALIDATION_WINDOWS|delta_ret20_pp|4|3|1.02|1.04|-0.03|2.03|0.3125|

## 7. Formal heterogeneity diagnostics

Linear-probability/OLS interaction models use standard errors clustered by signal date. The window test asks whether the MACD effect differs across windows; the discovery-vs-validation test asks whether the effect changed materially out of sample.

|Outcome|Dimension|Interaction terms|p-value|N|
|---|---|---:|---:|---:|
|stop|WINDOW|9|0.0000|2279|
|stop|DISCOVERY_VS_VALIDATION|1|0.0104|2279|
|stop|WINDOW_FAMILY|2|0.1038|2279|
|hit3|WINDOW|9|0.0008|2279|
|hit3|DISCOVERY_VS_VALIDATION|1|0.1069|2279|
|hit3|WINDOW_FAMILY|2|0.2014|2279|
|hit5|WINDOW|9|0.0001|2279|
|hit5|DISCOVERY_VS_VALIDATION|1|0.0654|2279|
|hit5|WINDOW_FAMILY|2|0.0038|2279|
|ret20|WINDOW|9|0.0339|2279|
|ret20|DISCOVERY_VS_VALIDATION|1|0.2230|2279|
|ret20|WINDOW_FAMILY|2|0.1007|2279|

## 8. Clustering

|Window|Group|Signals|Signal days|Max same day|Share on ≥5-signal days %|Share on ≥10-signal days %|
|---|---|---:|---:|---:|---:|---:|
|GFC_2007_2009|ALL_BASE|160|81|8|29.38|0.00|
|GFC_2007_2009|MACD_PASS|99|56|7|22.22|0.00|
|BULL_2013_2014|ALL_BASE|447|184|18|37.81|8.05|
|BULL_2013_2014|MACD_PASS|248|125|9|29.03|0.00|
|CHOP_2015_2016|ALL_BASE|312|147|11|29.17|7.05|
|CHOP_2015_2016|MACD_PASS|180|110|9|17.78|0.00|
|BULL_2017|ALL_BASE|172|105|5|2.91|0.00|
|BULL_2017|MACD_PASS|101|72|4|0.00|0.00|
|COVID_2020|ALL_BASE|129|45|20|44.96|41.09|
|COVID_2020|MACD_PASS|69|34|13|31.88|18.84|
|BEAR_2022|ALL_BASE|58|21|12|50.00|20.69|
|BEAR_2022|MACD_PASS|33|17|5|15.15|0.00|
|VALIDATION_2018|ALL_BASE|167|78|12|30.54|13.17|
|VALIDATION_2018|MACD_PASS|96|56|10|15.62|10.42|
|VALIDATION_2019|ALL_BASE|136|67|7|14.71|0.00|
|VALIDATION_2019|MACD_PASS|67|43|4|0.00|0.00|
|VALIDATION_2021|ALL_BASE|230|98|13|29.13|5.65|
|VALIDATION_2021|MACD_PASS|150|71|12|33.33|8.00|
|VALIDATION_2023_2024|ALL_BASE|468|209|15|30.34|7.91|
|VALIDATION_2023_2024|MACD_PASS|308|152|11|25.65|3.57|

## 9. Largest same-day candidate clusters

|Window|Date|Signals|MACD pass %|Stop %|+3 %|+5 %|Ret20 %|Regime|
|---|---|---:|---:|---:|---:|---:|---:|---|
|COVID_2020|2020-03-03|20|65.00|100.00|35.00|5.00|-20.56|YELLOW|
|COVID_2020|2020-03-02|19|5.26|100.00|57.89|15.79|-14.72|GREEN|
|BULL_2013_2014|2013-06-07|18|38.89|88.89|5.56|0.00|0.84|GREEN|
|BULL_2013_2014|2013-10-10|18|27.78|5.56|88.89|55.56|4.33|GREEN|
|VALIDATION_2023_2024|2024-05-31|15|60.00|40.00|46.67|33.33|-0.64|GREEN|
|COVID_2020|2020-09-28|14|64.29|28.57|92.86|78.57|1.65|GREEN|
|VALIDATION_2021|2021-02-03|13|92.31|7.69|92.31|76.92|10.50|GREEN|
|VALIDATION_2018|2018-02-07|12|8.33|33.33|66.67|33.33|3.33|GREEN|
|BEAR_2022|2022-02-25|12|25.00|25.00|66.67|50.00|3.56|YELLOW|
|CHOP_2015_2016|2015-03-13|11|81.82|54.55|27.27|9.09|0.57|GREEN|
|CHOP_2015_2016|2015-08-26|11|0.00|9.09|54.55|36.36|-0.49|YELLOW|
|VALIDATION_2023_2024|2023-06-02|11|100.00|9.09|90.91|72.73|4.82|GREEN|
|VALIDATION_2023_2024|2024-12-24|11|81.82|81.82|18.18|18.18|1.92|GREEN|
|VALIDATION_2018|2018-02-13|10|100.00|10.00|80.00|60.00|4.41|GREEN|
|BULL_2013_2014|2014-10-16|9|66.67|0.00|100.00|100.00|11.83|YELLOW|
|BULL_2013_2014|2014-12-18|9|100.00|55.56|0.00|0.00|-2.46|GREEN|
|CHOP_2015_2016|2016-06-29|9|11.11|11.11|100.00|88.89|5.66|GREEN|
|VALIDATION_2021|2021-06-22|9|88.89|22.22|55.56|44.44|1.10|GREEN|
|GFC_2007_2009|2009-10-06|8|62.50|12.50|100.00|87.50|3.03|GREEN|
|BULL_2013_2014|2014-12-17|8|0.00|12.50|62.50|12.50|0.62|GREEN|

## Interpretation guardrails
- No rule is promoted or rejected automatically by this diagnostic.
- Positive base-candidate Ret20 does not prove alpha versus an unconditional benchmark.
- A significant pooled MACD effect does not erase window heterogeneity.
- A non-significant discovery-vs-validation interaction is evidence of replication, not proof of universality.
- Same-day clusters are correlated macro events and must not be treated as independent trades.
- Layer 2, if opened, must still test option-chain economics, IV/theta, DTE/delta, earnings/ex-dividend exclusions, contract granularity and portfolio capacity.
- This report is designed to support the human decision: continue Layer-1 signal research or freeze a provisional architecture and proceed to Layer 2.
