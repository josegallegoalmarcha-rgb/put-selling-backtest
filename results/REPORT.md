# Layer 1 — V0.1 underlying signal backtest

**This is not yet an options P/L backtest.** It tests whether the underlying setup detects rebounds.

## Frozen rules
- Universe: contemporaneous 31-Jul-2007 Dividend Aristocrats list; later failures retained.
- Stock trend: MA50 > MA200.
- MACD: histogram < 0 and less negative for two consecutive closes.
- Stochastic: slow 14,3,3 crosses 20 upward.
- Market filter: no new signal when SPY MA50 <= MA200.
- Swing low: minimum Low of the current stochastic <=20 episode.
- Entry proxy: next-session Open.
- Stop: daily Close below swing low.
- Evaluation: +2%, +3%, +5% rebound before stop over 20 trading days.
- Control: same setup without the MACD-improvement condition.
- Per-stock cooldown: 20 trading days.

## Data coverage
Loaded **51/59** tickers. Missing: ROH, CBSS, WWY, FDO, MHP, SIAL, STR, WAG.

## Summary

|Signal|N|Stocks|Stop %|+2% before stop|+3% before stop|+5% before stop|Avg MFE %|Avg MAE %|Avg 10d %|Avg 20d %|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|STOCH_ONLY_CONTROL|160|46|39.38|77.50|71.88|50.62|6.70|-3.49|1.11|2.05|
|V0.1|112|45|34.82|75.89|67.86|50.00|6.64|-3.74|1.18|2.50|

## Guardrails
1. Positive Layer-1 results validate only the rebound signal, not option profitability.
2. Earnings/ex-dividend filters enter the option implementation layer, not this directional test.
3. Public OHLC corporate-action treatment must be audited before the synthetic/real options layer.
4. V0.1 is frozen for this test; a bad episode is not repaired retrospectively.
