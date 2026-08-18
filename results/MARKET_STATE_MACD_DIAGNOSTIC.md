# Market-state diagnostic — MACD_PASS vs MACD_FAIL

## Status
**Diagnostic only. No V0.1/V0.2 rule is changed and no V0.2-C is created.**

Sample: **1,278 common candidates** from the frozen V0.2-B common-candidate stream.

## Pre-specified market-state variables
- SPY ROC5: five-session speed of decline/rebound.
- SPY distance to MA50.
- SPY distance to MA200.
- SPY MA50 10-session slope.
- SPY 20-session annualized realized volatility.
- Breadth: percentage of the point-in-time universe above MA50.
- Breadth: percentage of the point-in-time universe with positive 5-session return.
- Same-day candidate cluster size and cluster share of the loaded universe.

All state variables are measured at the **signal close**. No future information is used.

## What this diagnostic is allowed to answer
We are looking for a state variable whose interaction with MACD is directionally stable across multiple windows. Fixed coarse buckets and one-variable-at-a-time interaction regressions are descriptive tools only; they do not define a trading threshold.

## MACD selection by market state — pooled means

|Group|N|ROC5 %|Dist MA50 %|Dist MA200 %|MA50 slope10 %|RV20 %|Breadth >MA50 %|Breadth 5d+ %|Cluster size|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|MACD_FAIL|548|-0.67|0.40|5.97|0.65|13.39|54.40|43.45|4.89|
|MACD_PASS|730|0.38|0.73|6.42|0.59|13.03|55.14|56.39|4.01|

## Continuous interaction regressions

Each row estimates whether a **1 standard-deviation change in the state variable changes the observed MACD_PASS minus MACD_FAIL relationship**, controlling for window fixed effects and clustering standard errors by signal date. These are descriptive associations, not causal estimates.

|Variable|Outcome|Interaction / 1 SD|SE|p-value|N|
|---|---|---:|---:|---:|---:|
|spy_dist_ma50_pct|hit3|-0.0291|0.0264|0.2703|1278|
|cluster_share_pct|hit3|-0.0274|0.0279|0.3261|1278|
|breadth_above_ma50_pct|hit3|-0.0275|0.0288|0.3400|1278|
|cluster_size|hit3|-0.0242|0.0268|0.3676|1278|
|spy_rv20_ann_pct|hit3|0.0086|0.0278|0.7577|1278|
|breadth_5d_positive_pct|hit3|0.0089|0.0309|0.7729|1278|
|spy_dist_ma200_pct|hit3|-0.0070|0.0259|0.7884|1278|
|spy_ma50_slope10_pct|hit3|-0.0066|0.0274|0.8087|1278|
|spy_roc5_pct|hit3|0.0003|0.0288|0.9909|1278|
|spy_ma50_slope10_pct|hit5|0.0392|0.0295|0.1837|1278|
|spy_roc5_pct|hit5|-0.0321|0.0291|0.2689|1278|
|spy_dist_ma200_pct|hit5|0.0196|0.0276|0.4764|1278|
|spy_rv20_ann_pct|hit5|0.0182|0.0276|0.5088|1278|
|spy_dist_ma50_pct|hit5|-0.0120|0.0276|0.6646|1278|
|breadth_5d_positive_pct|hit5|-0.0080|0.0281|0.7752|1278|
|breadth_above_ma50_pct|hit5|0.0057|0.0274|0.8339|1278|
|cluster_share_pct|hit5|-0.0060|0.0295|0.8400|1278|
|cluster_size|hit5|-0.0019|0.0308|0.9516|1278|
|breadth_5d_positive_pct|ret20|-1.0444|0.4533|0.0212|1278|
|spy_roc5_pct|ret20|-1.0571|0.5873|0.0719|1278|
|breadth_above_ma50_pct|ret20|-0.4696|0.5073|0.3547|1278|
|spy_dist_ma50_pct|ret20|-0.6208|0.6803|0.3615|1278|
|spy_rv20_ann_pct|ret20|0.6666|0.7507|0.3746|1278|
|cluster_size|ret20|0.3095|0.4832|0.5219|1278|
|cluster_share_pct|ret20|0.2718|0.4772|0.5690|1278|
|spy_dist_ma200_pct|ret20|-0.2230|0.5958|0.7082|1278|
|spy_ma50_slope10_pct|ret20|0.1301|0.6729|0.8467|1278|
|spy_dist_ma200_pct|stop|-0.0489|0.0319|0.1252|1278|
|spy_rv20_ann_pct|stop|-0.0361|0.0335|0.2817|1278|
|spy_ma50_slope10_pct|stop|-0.0287|0.0311|0.3558|1278|
|cluster_size|stop|-0.0152|0.0382|0.6908|1278|
|spy_roc5_pct|stop|-0.0156|0.0394|0.6924|1278|
|cluster_share_pct|stop|-0.0106|0.0361|0.7690|1278|
|spy_dist_ma50_pct|stop|-0.0094|0.0345|0.7850|1278|
|breadth_5d_positive_pct|stop|0.0026|0.0375|0.9443|1278|
|breadth_above_ma50_pct|stop|-0.0005|0.0362|0.9899|1278|

## Cross-window consistency guardrail

The fixed-bin file records, for every state bucket, how many eligible windows show MACD with fewer stops, more +3, more +5 and higher Ret20. A future V0.2-C should only be considered if a relationship is repeated across several windows with adequate PASS and FAIL sample sizes.

The full tables are written to CSV because the interaction matrix is too large to render safely in this report.

## Largest signal clusters

|Window|Date|Signals|MACD pass %|Stop %|Ret20 %|ROC5 %|RV20 %|Breadth >MA50 %|Breadth 5d+ %|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|COVID_2020|2020-03-03|20|65.0|100.0|-20.56|-3.97|30.97|4.8|6.5|
|COVID_2020|2020-03-02|19|5.3|100.0|-14.72|-4.13|29.71|11.3|4.8|
|BULL_2013_2014|2013-06-07|18|38.9|88.9|0.84|0.81|12.15|61.5|78.8|
|BULL_2013_2014|2013-10-10|18|27.8|5.6|4.33|0.92|12.72|63.5|78.8|
|COVID_2020|2020-09-28|14|64.3|28.6|1.65|2.21|24.97|54.8|72.6|
|BEAR_2022|2022-02-25|12|25.0|25.0|3.56|0.16|24.60|29.0|59.7|
|CHOP_2015_2016|2015-03-13|11|81.8|54.5|0.57|0.26|10.57|43.1|39.2|
|CHOP_2015_2016|2015-08-26|11|0.0|9.1|-0.49|-6.66|25.43|7.8|0.0|
|BULL_2013_2014|2014-10-16|9|66.7|0.0|11.83|-3.35|15.96|15.4|11.5|
|BULL_2013_2014|2014-12-18|9|100.0|55.6|-2.46|1.28|15.67|76.9|90.4|
|CHOP_2015_2016|2016-06-29|9|11.1|11.1|5.66|-0.70|19.04|58.8|49.0|
|GFC_2007_2009|2009-10-06|8|62.5|12.5|3.03|-0.46|16.60|66.0|32.0|
|BULL_2013_2014|2014-12-17|8|0.0|12.5|0.62|-0.67|12.81|71.2|59.6|
|CHOP_2015_2016|2016-09-21|8|87.5|100.0|-5.09|1.76|13.29|25.5|88.2|
|GFC_2007_2009|2009-11-09|7|100.0|14.3|0.57|5.02|21.50|78.0|100.0|
|BULL_2013_2014|2013-06-10|7|100.0|42.9|3.65|0.27|12.11|59.6|59.6|
|BULL_2013_2014|2013-08-23|7|100.0|57.1|2.78|0.49|9.20|46.2|44.2|
|BULL_2013_2014|2013-12-18|7|28.6|0.0|2.66|1.67|10.02|71.2|71.2|
|CHOP_2015_2016|2015-06-10|7|28.6|42.9|0.38|-0.46|9.21|31.4|27.5|
|CHOP_2015_2016|2016-06-16|7|14.3|57.1|5.94|-1.75|8.16|72.5|33.3|

## SPY data-state drift check

|Window|N|Max abs Close drift %|Max abs MA50 drift %|Max abs MA200 drift %|
|---|---:|---:|---:|---:|
|BEAR_2022|58|0.00004|0.00000|0.00000|
|BULL_2013_2014|447|0.00000|0.00000|0.00000|
|BULL_2017|172|0.00004|0.00001|0.00000|
|CHOP_2015_2016|312|0.00000|0.00000|0.00000|
|COVID_2020|129|0.00004|0.00000|0.00000|
|GFC_2007_2009|160|0.00000|0.00000|0.00000|

## Guardrails
- Do not create a rule from the best-looking single bucket.
- Do not scan additional thresholds after seeing the output.
- A candidate explanatory variable should show the same directional interaction in several distinct windows and retain adequate sample size.
- Breadth and clustering are treated as separate concepts: breadth measures the cross-section of the point-in-time universe; clustering measures the density of actual candidate signals.
- This remains Layer 1 underlying-price research, not short-put P/L.
