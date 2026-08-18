from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import pandas as pd
import requests

DATA_REPO = "neo-zhao/CMSC320_Final_Tutorial_Huge_Stock_Market_Dataset"
RAW_BASE = f"https://raw.githubusercontent.com/{DATA_REPO}/main"
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

# Point-in-time universe: 53 S&P 500 Dividend Aristocrats at 1-Jan-2015.
# Bemis (BMS) is excluded because it left the S&P 500 on 4-Dec-2014.
# FDO and SIAL remain because they were still constituents at the start of 2015.
UNIVERSE_2015_01 = """
ABBV ABT ADM ADP AFL APD BCR BDX BEN BF-B CAH CB CINF CL CLX CTAS CVX DOV ECL
ED EMR FDO GPC GWW HCP HRL ITW JNJ KMB KO LEG LOW MCD MDT MHFI MKC MMM NUE PEP
PG PNR PPG SHW SIAL SWK SYY T TGT TROW VFC WBA WMT XOM
""".split()

# Purely a data-source ticker normalization. This does NOT change the strategy.
DATA_TICKER_ALIAS = {
    "MHFI": "SPGI",   # McGraw-Hill Financial renamed S&P Global in 2016.
}

START = pd.Timestamp("2015-01-01")
END = pd.Timestamp("2016-12-31")
COOLDOWN = 20
HORIZONS = (5, 10, 15, 20)

def get(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text

def load_symbol(ticker, etf=False):
    folder = "ETFs" if etf else "Stocks"
    data_ticker = DATA_TICKER_ALIAS.get(ticker, ticker)
    url = f"{RAW_BASE}/{folder}/{data_ticker.lower()}.us.txt"
    try:
        txt = get(url)
    except Exception:
        return None
    try:
        d = pd.read_csv(io.StringIO(txt))
    except Exception:
        return None
    if "Date" not in d.columns:
        return None
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.sort_values("Date").drop_duplicates("Date").set_index("Date")
    need = ["Open", "High", "Low", "Close"]
    if any(c not in d.columns for c in need):
        return None
    for c in need:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=need)

def add_indicators(d):
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
    x["STOCH_K"] = fast_k.rolling(3).mean()
    x["STOCH_D"] = x["STOCH_K"].rolling(3).mean()
    return x

def load_spy():
    s = load_symbol("SPY", etf=True)
    if s is None:
        raise RuntimeError("SPY could not be loaded")
    s = add_indicators(s)
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

def attach_spy(x, spy):
    r = spy[["Close", "MA50", "MA200", "REGIME"]].rename(
        columns={"Close":"SPY_CLOSE","MA50":"SPY_MA50","MA200":"SPY_MA200"}
    )
    return x.join(r, how="left").ffill()

def masks(x):
    trend = x["MA50"] > x["MA200"]
    market = x["REGIME"].isin(["GREEN", "YELLOW"])
    stoch_cross = (x["STOCH_K"].shift(1) <= 20) & (x["STOCH_K"] > 20)

    h = x["MACD_HIST"]
    macd_improving = (h < 0) & (h > h.shift(1)) & (h.shift(1) > h.shift(2))

    control = (trend & market & stoch_cross).fillna(False)
    v01 = (control & macd_improving).fillna(False)
    return v01, control

def spaced(mask):
    raw = np.flatnonzero(mask.to_numpy())
    out, last = [], -10**9
    for i in raw:
        if i - last >= COOLDOWN:
            out.append(int(i))
            last = int(i)
    return out

def swing_low(x, i):
    j = i - 1
    while j > 0 and pd.notna(x["STOCH_K"].iloc[j]) and x["STOCH_K"].iloc[j] <= 20:
        j -= 1
    a = min(j + 1, i)
    seg = x.iloc[a:i+1]
    return float(seg["Low"].min()), seg.index[0]

def evaluate(x, i, ticker, kind):
    if i + 21 >= len(x):
        return None

    entry_i = i + 1
    entry = float(x["Open"].iloc[entry_i])
    if not np.isfinite(entry) or entry <= 0:
        return None

    sl, swing_start = swing_low(x, i)
    path = x.iloc[entry_i:entry_i+21]

    stop_pos = np.flatnonzero((path["Close"] < sl).to_numpy())
    stop_rel = int(stop_pos[0]) if len(stop_pos) else None
    effective = path.iloc[:stop_rel+1] if stop_rel is not None else path

    z = {
        "ticker": ticker,
        "signal_type": kind,
        "signal_date": x.index[i],
        "entry_date": x.index[entry_i],
        "entry_open": entry,
        "regime": x["REGIME"].iloc[i],
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

    for target in (2,3,5):
        hits = np.flatnonzero((effective["High"] >= entry*(1+target/100)).to_numpy())
        z[f"hit_{target}pct_before_stop"] = bool(len(hits))
        z[f"days_to_{target}pct"] = int(hits[0]) if len(hits) else np.nan
    return z

def summarize(g):
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

    for ticker in UNIVERSE_2015_01:
        d = load_symbol(ticker)
        if d is None:
            coverage.append({
                "ticker":ticker,
                "data_ticker":DATA_TICKER_ALIAS.get(ticker,ticker),
                "loaded":False
            })
            continue

        coverage.append({
            "ticker":ticker,
            "data_ticker":DATA_TICKER_ALIAS.get(ticker,ticker),
            "loaded":True,
            "first":d.index.min(),
            "last":d.index.max(),
            "rows":len(d)
        })

        x = attach_spy(add_indicators(d), spy)
        v01, control = masks(x)

        for kind, mask in [("V0.1", v01), ("STOCH_ONLY_CONTROL", control)]:
            for i in spaced(mask):
                if START <= x.index[i] <= END:
                    e = evaluate(x, i, ticker, kind)
                    if e is not None:
                        events.append(e)

    cov = pd.DataFrame(coverage)
    cov.to_csv(RESULTS/"coverage_2015.csv", index=False)

    ev = pd.DataFrame(events).sort_values(["signal_date","ticker","signal_type"])
    ev.to_csv(RESULTS/"events_2015_2016.csv", index=False)

    rows = []
    for kind, g in ev.groupby("signal_type"):
        r = {"signal_type":kind}
        r.update(summarize(g))
        rows.append(r)
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS/"summary_2015_2016.csv", index=False)

    stock_rows = []
    for (kind,ticker), g in ev.groupby(["signal_type","ticker"]):
        r = {"signal_type":kind,"ticker":ticker}
        r.update(summarize(g))
        stock_rows.append(r)
    pd.DataFrame(stock_rows).to_csv(
        RESULTS/"summary_by_stock_2015_2016.csv", index=False
    )

    regime_rows = []
    for (kind,regime), g in ev.groupby(["signal_type","regime"]):
        r = {"signal_type":kind,"regime":regime}
        r.update(summarize(g))
        regime_rows.append(r)
    pd.DataFrame(regime_rows).to_csv(
        RESULTS/"summary_by_regime_2015_2016.csv", index=False
    )

    loaded = int(cov["loaded"].sum())
    missing = ", ".join(cov.loc[~cov["loaded"],"ticker"].tolist()) or "None"

    report = [
        "# Layer 1 — V0.1 — 2015–2016\n\n",
        "**This is not yet an options P/L backtest.** It tests whether the underlying setup detects rebounds.\n\n",
        "## Frozen-rule integrity\n",
        "**No V0.1 trading rule has been changed from the GFC test.** Only the point-in-time universe and test dates differ.\n\n",
        "## Frozen rules\n",
        "- Universe: 53 S&P 500 Dividend Aristocrats at 1-Jan-2015.\n",
        "- Stock trend: MA50 > MA200.\n",
        "- MACD: histogram < 0 and less negative for two consecutive closes.\n",
        "- Stochastic: slow 14,3,3 crosses 20 upward.\n",
        "- Market filter: no new signal when SPY MA50 <= MA200.\n",
        "- Swing low: minimum Low of the current stochastic <=20 episode.\n",
        "- Entry proxy: next-session Open.\n",
        "- Stop: daily Close below swing low.\n",
        "- Evaluation: +2%, +3%, +5% rebound before stop over 20 trading days.\n",
        "- Control: same setup without the MACD-improvement condition.\n",
        f"- Per-stock cooldown: {COOLDOWN} trading days.\n\n",
        "## Data plumbing\n",
        "- MHFI is read from the SPGI history because McGraw-Hill Financial changed its ticker/name to S&P Global in 2016.\n",
        "- This alias is data continuity only; it is not a strategy rule.\n\n",
        "## Data coverage\n",
        f"Loaded **{loaded}/{len(UNIVERSE_2015_01)}** tickers. Missing: {missing}.\n\n",
        "## Summary\n\n",
    ]

    if len(summary):
        report.append("|Signal|N|Stocks|Stop %|+2% before stop|+3% before stop|+5% before stop|Avg MFE %|Avg MAE %|Avg 10d %|Avg 20d %|\n")
        report.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for _,r in summary.iterrows():
            report.append(
                f"|{r.signal_type}|{int(r.n)}|{int(r.stocks)}|{r.stop_rate_pct:.2f}|"
                f"{r.hit2_before_stop_pct:.2f}|{r.hit3_before_stop_pct:.2f}|{r.hit5_before_stop_pct:.2f}|"
                f"{r.avg_mfe_before_stop_pct:.2f}|{r.avg_mae_before_stop_pct:.2f}|"
                f"{r.avg_ret_10d_pct:.2f}|{r.avg_ret_20d_pct:.2f}|\n"
            )

    report += [
        "\n## Guardrails\n",
        "1. Positive Layer-1 results validate only the rebound signal, not option profitability.\n",
        "2. Earnings/ex-dividend filters enter the option implementation layer, not this directional test.\n",
        "3. Public OHLC corporate-action treatment must be audited before the synthetic/real options layer.\n",
        "4. V0.1 is frozen: 2015–2016 is not allowed to modify the rules inherited from the GFC test.\n",
    ]

    (RESULTS/"REPORT_2015_2016.md").write_text("".join(report), encoding="utf-8")
    print("".join(report))

if __name__ == "__main__":
    main()
