# Layer 1 — V0.1 — COVID 2020

**This is not yet an options P/L backtest.** It tests whether the underlying setup detects rebounds.

## Frozen-rule integrity
**No V0.1 trading rule has been changed from the GFC or 2015–2016 tests.** Only the contemporaneous universe, test dates and required modern data source differ.

## Test window
- 2-Jan-2020 through 30-Sep-2020.
- This captures the pre-crash period, the February–March collapse, the March rebound and the early recovery.

## Frozen rules
- Stock trend: MA50 > MA200.
- MACD: histogram < 0 and less negative for two consecutive closes.
- Stochastic: slow 14,3,3 crosses 20 upward.
- Market filter: no new signal when SPY MA50 <= MA200.
- Swing low: minimum Low of current stochastic <=20 episode.
- Entry proxy: next-session Open.
- Stop: daily Close below swing low.
- Evaluation: +2%, +3%, +5% rebound before stop over 20 trading days.
- Control: same setup without MACD-improvement condition.
- Per-stock cooldown: 20 trading days.

## Point-in-time universe
- 64 Dividend Aristocrats from the contemporaneous 25-Jan-2020 list.
- Later dividend cuts/removals are deliberately not filtered out.

## Data plumbing
- Yahoo Finance via yfinance, auto-adjusted OHLC.
- UTX is read through RTX for historical continuity after the 2020 corporate rename/merger.
- These are data-source decisions only; they are not strategy rules.

## Data coverage
Loaded **62/64** tickers. Missing: PBCT, WBA.

## Summary

|Signal|N|Stocks|Stop %|+2% before stop|+3% before stop|+5% before stop|Avg MFE %|Avg MAE %|Avg 10d %|Avg 20d %|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|STOCH_ONLY_CONTROL|130|58|71.54|73.08|56.15|30.77|4.99|-6.55|-5.16|-6.15|
|V0.1|97|53|56.70|72.16|62.89|43.30|7.26|-5.37|-1.62|-2.15|

## Guardrails
1. Positive Layer-1 results validate only the rebound signal, not option profitability.
2. Earnings/ex-dividend filters remain outside this directional layer.
3. Cross-period comparison must note that 2020 requires a modern adjusted-OHLC data source, unlike the legacy dataset used earlier.
4. V0.1 remains frozen: COVID results are not allowed to alter the inherited rules.
