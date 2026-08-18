# Layer 1 — V0.1 — 2022 Bear Market

**This is not yet an options P/L backtest.** It tests whether the underlying setup detects rebounds.

## Frozen-rule integrity
**No V0.1 trading rule has been changed from the GFC, 2015–2016 or COVID tests.** Only the point-in-time universe and test dates differ.

## Test window
- 3-Jan-2022 through 30-Dec-2022.
- Full calendar-year bear-market test.

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
- Fixed 1-Jan-2022 snapshot: 64 Dividend Aristocrats.
- T and PBCT retained because they were members at test start.
- BRO and CHD excluded because their addition was effective 1-Feb-2022.
- LEG excluded because it had already left the S&P 500 in Dec-2021.
- No later membership change is used retrospectively.

## Data plumbing
- Yahoo Finance via yfinance, auto-adjusted OHLC, same modern data approach used in the COVID test.

## Data coverage
Loaded **62/64** tickers. Missing: PBCT, WBA.

## Summary

|Signal|N|Stocks|Stop %|+2% before stop|+3% before stop|+5% before stop|Avg MFE %|Avg MAE %|Avg 10d %|Avg 20d %|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|STOCH_ONLY_CONTROL|58|41|53.45|67.24|53.45|41.38|5.28|-4.93|-0.89|-0.94|
|V0.1|38|35|68.42|55.26|44.74|31.58|4.57|-5.42|-1.69|-2.64|

## Guardrails
1. Positive Layer-1 results validate only the rebound signal, not option profitability.
2. Earnings/ex-dividend filters remain outside this directional layer.
3. V0.1 remains frozen: no 2022 result is allowed to alter the inherited rules before cross-window review.
4. Clustering files are diagnostics only and never affect entries.
