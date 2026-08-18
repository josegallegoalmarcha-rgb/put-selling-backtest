# Layer 1 — V0.1 — BULL_2013_2014

**Bull-control window. This is not an options P/L backtest.**

## Frozen-rule integrity
**No V0.1 trading rule has changed.** MA50/MA200, MACD, Slow Stochastic 14,3,3, SPY regime, swing-low, next-open entry, daily-close stop and 20-session cooldown are inherited unchanged.

## Data coverage
- Loaded **52/54** tickers.
- Missing: FDO, SIAL.
- SPY data source: legacy.

## Summary

|Signal|N|Stocks|Stop %|+2% before stop|+3% before stop|+5% before stop|Avg MFE %|Avg MAE %|Avg 10d %|Avg 20d %|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|STOCH_ONLY_CONTROL|447|52|41.16|66.67|57.49|39.15|4.62|-2.24|1.25|2.62|
|V0.1|314|52|37.90|71.97|61.46|44.59|5.01|-2.10|1.78|2.99|

## Guardrails
1. The bull control is evaluated with exactly the frozen V0.1 logic.
2. The fixed annual universe is not reconstituted retrospectively during the test.
3. Positive underlying results do not prove option profitability.
