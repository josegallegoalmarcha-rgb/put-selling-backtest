# Layer 1 — V0.1 — BULL_2017

**Bull-control window. This is not an options P/L backtest.**

## Frozen-rule integrity
**No V0.1 trading rule has changed.** MA50/MA200, MACD, Slow Stochastic 14,3,3, SPY regime, swing-low, next-open entry, daily-close stop and 20-session cooldown are inherited unchanged.

## Data coverage
- Loaded **51/51** tickers.
- Missing: None.
- SPY data source: yfinance.

## Summary

|Signal|N|Stocks|Stop %|+2% before stop|+3% before stop|+5% before stop|Avg MFE %|Avg MAE %|Avg 10d %|Avg 20d %|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|STOCH_ONLY_CONTROL|172|50|41.28|66.28|53.49|34.88|4.28|-2.15|1.08|1.82|
|V0.1|111|47|50.45|62.16|45.05|27.93|3.48|-2.20|0.49|1.02|

## Guardrails
1. The bull control is evaluated with exactly the frozen V0.1 logic.
2. The fixed annual universe is not reconstituted retrospectively during the test.
3. Positive underlying results do not prove option profitability.
