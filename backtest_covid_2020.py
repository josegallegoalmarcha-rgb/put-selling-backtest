from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import pandas as pd
import yfinance as yf

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

# Contemporaneous Dividend Aristocrats list published 25-Jan-2020.
# 64 constituents. No later survivor filtering is applied.
UNIVERSE_2020_01 = """
ABBV ABT ADM ADP AFL ALB AMCR AOS APD ATO BDX BEN BF-B CAH CAT CB CINF CL CLX
CTAS CVX DOV ECL ED EMR ESS EXPD FRT GD GPC GWW HRL ITW JNJ KMB KO LEG LIN LOW
MCD MDT MKC MMM NUE O PBCT PEP PG PNR PPG ROP ROST SHW SPGI SWK SYY T TGT TROW
UTX VFC WBA WMT XOM
""".split()

# Data-continuity aliases only. They do NOT alter V0.1 rules.
# UTX was renamed RTX after the 2020 Raytheon merger; Yahoo's RTX history
# contains the predecessor's historical series.
DATA_TICKER_ALIAS = {
    "BF-B": "BF-B",
    "UTX": "RTX",
}

# COVID stress window used for this validation test.
# Includes the pre-crash setup, crash, rebound and early recovery.
START = pd.Timestamp("2020-01-02")
END = pd.Timestamp("2020-09-30")

COOLDOWN = 20
HORIZONS = (5, 10, 15, 20)

# Indicator warm-up and forward-look buffer only; not part of the signal.
DOWNLOAD_START = "2018-01-01"
DOWNLOAD_END = "2020-11-15"

def load_symbol(ticker: str) -> pd.DataFrame | None:
    data_ticker = DATA_TICKER_ALIAS.get(ticker, ticker)
    try:
        d = yf.download(
            data_ticker,
            start=DOWNLOAD_START,
            end=DOWNLOAD_END,
            auto_adjust=True,
            progress=False,
            actions=False,
            threads=False,
        )
    except Exception:
        return None

    if d is None or d.empty:
        return None

    # yfinance may return MultiIndex columns even for one ticker.
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)

    need = ["Open", "High", "Low", "Close"]
    if any(c not in d.columns for c in need):
        return None

    d = d[need].copy()
    d.index = pd.to_datetime(d.index).tz_localize(None)
    d = d.sort_index().drop_duplicates()
    for c in need:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=need)
    return d if not d.empty else None

def add_indicators(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    c = x["Close"]

    x["MA50"] = c.rolling(50).mean()
    x["MA200"] = c.rolling(200).mean()

    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    x["MACD"] = e12 - e26
    x["MACD_SIGNAL"] = x["MACD"].ewm(span=9, adjust=False).mean()
    x["MACD_HIST"] = x["MACD"] - x["MACD_SIGNAL"]

    ll = x["Low"].rolling(14).min()
    hh = x["High"].rolling(14).max()
    fast_k = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    x["STOCH_K"] = fast_k.rolling(3).mean()       # Slow Stochastic 14,3,3
    x["STOCH_D"] = x["STOCH_K"].rolling(3).mean()
    return x

def load_spy() -> pd.DataFrame:
    s = load_symbol("SPY")
    if s is None:
        raise RuntimeError("SPY could not be downloaded from Yahoo Finance.")
    s = add_indicators(s)

    # IDENTICAL V0.1 regime definition.
    s["REGIME"] = np.select(
        [
            s["MA50"] <= s["MA200"],
            (s["MA50"] > s["MA200"]) & (s["Close"] <= s["MA200"]),
            (s["MA50"] > s["MA200"]) & (s["Close"] > s["MA200"]),
        ],
        ["RED", "YELLOW", "GREEN"],
        default="UNKNOWN",
    )
    return s

def attach_spy(x: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    r = spy[["Close", "MA50", "MA200", "REGIME"]].rename(
        columns={
            "Close": "SPY_CLOSE",
            "MA50": "SPY_MA50",
            "MA200": "SPY_MA200",
        }
    )
    return x.join(r, how="left").ffill()

def masks(x: pd.DataFrame):
    # IDENTICAL V0.1 signal logic.
    trend = x["MA50"] > x["MA200"]
    market = x["REGIME"].isin(["GREEN", "YELLOW"])
    stoch_cross = (x["STOCH_K"].shift(1) <= 20) & (x["STOCH_K"] > 20)

    h = x["MACD_HIST"]
    macd_improving = (h < 0) & (h > h.shift(1)) & (h.shift(1) > h.shift(2))

    control = (trend & market & stoch_cross).fillna(False)
    v01 = (control & macd_improving).fillna(False)
    return v01, control

def spaced(mask: pd.Series):
    # IDENTICAL 20-session per-stock cooldown.
    raw = np.flatnonzero(mask.to_numpy())
    out, last = [], -10**9
    for i in raw:
        if i - last >= COOLDOWN:
            out.append(int(i))
            last = int(i)
    return out

def swing_low(x: pd.DataFrame, i: int):
    # IDENTICAL V0.1 swing-low definition.
    j = i - 1
    while (
        j > 0
        and pd.notna(x["STOCH_K"].iloc[j])
        and x["STOCH_K"].iloc[j] <= 20
    ):
        j -= 1

    a = min(j + 1, i)
    seg = x.iloc[a:i+1]
    return float(seg["Low"].min()), seg.index[0]

def evaluate(x: pd.DataFrame, i: int, ticker: str, kind: str):
    # Need next-day entry + 20 forward trading days.
    if i + 21 >= len(x):
        return None

    entry_i = i + 1
    entry = float(x["Open"].iloc[entry_i])
    if not np.isfinite(entry) or entry <= 0:
        return None

    sl, swing_start = swing_low(x, i)
    path = x.iloc[entry_i:entry_i+21]

    # IDENTICAL V0.1 stop: first DAILY CLOSE below swing low.
    stop_pos = np.flatnonzero((path["Close"] < sl).to_numpy())
    stop_rel = int(stop_pos[0]) if len(stop_pos) else None
    effective = path.iloc[:stop_rel+1] if stop_rel is not None else path

    z = {
        "ticker": ticker,
        "data_ticker": DATA_TICKER_ALIAS.get(ticker, ticker),
        "signal_type": kind,
        "signal_date": x.index[i],
        "entry_date": x.index[entry_i],
        "entry_open": entry,
        "regime": x["REGIME"].iloc[i],
        "spy_close": x["SPY_CLOSE"].iloc[i],
        "spy_ma50": x["SPY_MA50"].iloc[i],
        "spy_ma200": x["SPY_MA200"].iloc[i],
        "swing_start": swing_start,
        "swing_low": sl,
        "swing_distance_pct": (entry/sl - 1)*100 if sl > 0 else np.nan,
        "stop_hit_20d": stop_rel is not None,
        "days_to_stop": stop_rel if stop_rel is not None else np.nan,
        "mfe_20d_pct": (path["High"].max()/entry - 1)*100,
        "mae_20d_pct": (path["Low"].min()/entry - 1)*100,
        "mfe_before_stop_pct": (effective["High"].max()/entry - 1)*100,
        "mae_before_stop_pct": (effective["Low"].min()/entry - 1)*100,
    }

    for h in HORIZONS:
        z[f"ret_{h}d_pct"] = (x["Close"].iloc[entry_i+h]/entry - 1)*100

    for target in (2, 3, 5):
        hits = np.flatnonzero(
            (effective["High"] >= entry*(1+target/100)).to_numpy()
        )
        z[f"hit_{target}pct_before_stop"] = bool(len(hits))
        z[f"days_to_{target}pct"] = int(hits[0]) if len(hits) else np.nan

    return z

def summarize(g: pd.DataFrame):
    if len(g) == 0:
        return {}
    return {
        "n": len(g),
        "stocks": g["ticker"].nunique(),
        "stop_rate_pct": 100*g["stop_hit_20d"].mean(),
        "hit2_before_stop_pct": 100*g["hit_2pct_before_stop"].mean(),
        "hit3_before_stop_pct": 100*g["hit_3pct_before_stop"].mean(),
        "hit5_before_stop_pct": 100*g["hit_5pct_before_stop"].mean(),
        "avg_mfe_before_stop_pct": g["mfe_before_stop_pct"].mean(),
        "avg_mae_before_stop_pct": g["mae_before_stop_pct"].mean(),
        "avg_ret_5d_pct": g["ret_5d_pct"].mean(),
        "avg_ret_10d_pct": g["ret_10d_pct"].mean(),
        "avg_ret_15d_pct": g["ret_15d_pct"].mean(),
        "avg_ret_20d_pct": g["ret_20d_pct"].mean(),
        "median_swing_distance_pct": g["swing_distance_pct"].median(),
    }

def main():
    spy = load_spy()
    events = []
    coverage = []

    for ticker in UNIVERSE_2020_01:
        d = load_symbol(ticker)

        if d is None:
            coverage.append({
                "ticker": ticker,
                "data_ticker": DATA_TICKER_ALIAS.get(ticker, ticker),
                "loaded": False,
            })
            continue

        coverage.append({
            "ticker": ticker,
            "data_ticker": DATA_TICKER_ALIAS.get(ticker, ticker),
            "loaded": True,
            "first": d.index.min(),
            "last": d.index.max(),
            "rows": len(d),
        })

        x = attach_spy(add_indicators(d), spy)
        v01, control = masks(x)

        for kind, mask in [
            ("V0.1", v01),
            ("STOCH_ONLY_CONTROL", control),
        ]:
            for i in spaced(mask):
                if START <= x.index[i] <= END:
                    e = evaluate(x, i, ticker, kind)
                    if e is not None:
                        events.append(e)

        # Be polite to Yahoo's endpoint.
        time.sleep(0.05)

    cov = pd.DataFrame(coverage)
    cov.to_csv(RESULTS/"coverage_covid_2020.csv", index=False)

    if not events:
        raise RuntimeError(
            "No events were generated. Check Yahoo download coverage/logs."
        )

    ev = pd.DataFrame(events).sort_values(
        ["signal_date", "ticker", "signal_type"]
    )
    ev.to_csv(RESULTS/"events_covid_2020.csv", index=False)

    rows = []
    for kind, g in ev.groupby("signal_type"):
        r = {"signal_type": kind}
        r.update(summarize(g))
        rows.append(r)

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS/"summary_covid_2020.csv", index=False)

    stock_rows = []
    for (kind, ticker), g in ev.groupby(["signal_type", "ticker"]):
        r = {"signal_type": kind, "ticker": ticker}
        r.update(summarize(g))
        stock_rows.append(r)
    pd.DataFrame(stock_rows).to_csv(
        RESULTS/"summary_by_stock_covid_2020.csv", index=False
    )

    regime_rows = []
    for (kind, regime), g in ev.groupby(["signal_type", "regime"]):
        r = {"signal_type": kind, "regime": regime}
        r.update(summarize(g))
        regime_rows.append(r)
    pd.DataFrame(regime_rows).to_csv(
        RESULTS/"summary_by_regime_covid_2020.csv", index=False
    )

    # Temporal clustering diagnostic, without changing trade rules.
    v = ev[ev["signal_type"] == "V0.1"].copy()
    day = (
        v.groupby("signal_date")
        .agg(
            signals=("ticker", "size"),
            stocks=("ticker", "nunique"),
            stop_rate=("stop_hit_20d", "mean"),
            avg_ret_20d_pct=("ret_20d_pct", "mean"),
        )
        .reset_index()
    )
    day["stop_rate_pct"] = 100 * day.pop("stop_rate")
    day.sort_values(
        ["signals", "signal_date"], ascending=[False, True]
    ).to_csv(RESULTS/"clustering_by_day_covid_2020.csv", index=False)

    loaded = int(cov["loaded"].sum())
    missing = ", ".join(
        cov.loc[~cov["loaded"], "ticker"].tolist()
    ) or "None"

    report = [
        "# Layer 1 — V0.1 — COVID 2020\n\n",
        "**This is not yet an options P/L backtest.** "
        "It tests whether the underlying setup detects rebounds.\n\n",

        "## Frozen-rule integrity\n",
        "**No V0.1 trading rule has been changed from the GFC or "
        "2015–2016 tests.** Only the contemporaneous universe, "
        "test dates and required modern data source differ.\n\n",

        "## Test window\n",
        "- 2-Jan-2020 through 30-Sep-2020.\n",
        "- This captures the pre-crash period, the February–March "
        "collapse, the March rebound and the early recovery.\n\n",

        "## Frozen rules\n",
        "- Stock trend: MA50 > MA200.\n",
        "- MACD: histogram < 0 and less negative for two consecutive closes.\n",
        "- Stochastic: slow 14,3,3 crosses 20 upward.\n",
        "- Market filter: no new signal when SPY MA50 <= MA200.\n",
        "- Swing low: minimum Low of current stochastic <=20 episode.\n",
        "- Entry proxy: next-session Open.\n",
        "- Stop: daily Close below swing low.\n",
        "- Evaluation: +2%, +3%, +5% rebound before stop over 20 trading days.\n",
        "- Control: same setup without MACD-improvement condition.\n",
        f"- Per-stock cooldown: {COOLDOWN} trading days.\n\n",

        "## Point-in-time universe\n",
        "- 64 Dividend Aristocrats from the contemporaneous "
        "25-Jan-2020 list.\n",
        "- Later dividend cuts/removals are deliberately not filtered out.\n\n",

        "## Data plumbing\n",
        "- Yahoo Finance via yfinance, auto-adjusted OHLC.\n",
        "- UTX is read through RTX for historical continuity after "
        "the 2020 corporate rename/merger.\n",
        "- These are data-source decisions only; they are not strategy rules.\n\n",

        "## Data coverage\n",
        f"Loaded **{loaded}/{len(UNIVERSE_2020_01)}** tickers. "
        f"Missing: {missing}.\n\n",

        "## Summary\n\n",
    ]

    if len(summary):
        report.append(
            "|Signal|N|Stocks|Stop %|+2% before stop|+3% before stop|"
            "+5% before stop|Avg MFE %|Avg MAE %|Avg 10d %|Avg 20d %|\n"
        )
        report.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        for _, r in summary.iterrows():
            report.append(
                f"|{r.signal_type}|{int(r.n)}|{int(r.stocks)}|"
                f"{r.stop_rate_pct:.2f}|{r.hit2_before_stop_pct:.2f}|"
                f"{r.hit3_before_stop_pct:.2f}|{r.hit5_before_stop_pct:.2f}|"
                f"{r.avg_mfe_before_stop_pct:.2f}|"
                f"{r.avg_mae_before_stop_pct:.2f}|"
                f"{r.avg_ret_10d_pct:.2f}|{r.avg_ret_20d_pct:.2f}|\n"
            )

    report += [
        "\n## Guardrails\n",
        "1. Positive Layer-1 results validate only the rebound signal, "
        "not option profitability.\n",
        "2. Earnings/ex-dividend filters remain outside this directional layer.\n",
        "3. Cross-period comparison must note that 2020 requires a modern "
        "adjusted-OHLC data source, unlike the legacy dataset used earlier.\n",
        "4. V0.1 remains frozen: COVID results are not allowed to alter "
        "the inherited rules.\n",
    ]

    (RESULTS/"REPORT_COVID_2020.md").write_text(
        "".join(report), encoding="utf-8"
    )
    print("".join(report))

if __name__ == "__main__":
    main()
