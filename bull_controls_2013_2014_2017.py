from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

# Legacy OHLC source already used for GFC and 2015-16.
DATA_REPO = "neo-zhao/CMSC320_Final_Tutorial_Huge_Stock_Market_Dataset"
RAW_BASE = f"https://raw.githubusercontent.com/{DATA_REPO}/main"

# ---------------------------------------------------------------------
# POINT-IN-TIME / CONTEMPORANEOUS UNIVERSES
# ---------------------------------------------------------------------

# 2013 annual Dividend Aristocrats update: 54 names.
# Fixed through the full 2013-2014 bull-control window.
# Later removals are deliberately retained to avoid retrospective filtering.
UNIVERSE_2013 = """
MMM ABT ABBV AFL APD ADM T ADP BDX BMS BF-B BCR CAH CVX CB CINF CTAS CLX KO CL
ED DOV ECL EMR XOM FDO BEN GPC HCP HRL ITW JNJ KMB LEG LOW MKC MCD MHFI MDT NUE
PNR PEP PPG PG SHW SIAL SWK SYY TROW TGT VFC GWW WAG WMT
""".split()

# 2017 annual Dividend Aristocrats list: 51 names.
# Relative to the 2015 snapshot: FDO/SIAL had been acquired, old Chubb (CB)
# was removed after acquisition, HCP left the index in 2017, while GD and FRT
# were added. BCR remains because its acquisition closed only at end-Dec-2017.
UNIVERSE_2017 = """
ABBV ABT ADM ADP AFL APD BCR BDX BEN BF-B CAH CINF CL CLX CTAS CVX DOV ECL ED EMR
GPC GWW HRL ITW JNJ KMB KO LEG LOW MCD MDT MHFI MKC MMM NUE PEP PG PNR PPG SHW
SWK SYY T TGT TROW VFC WBA WMT XOM GD FRT
""".split()

assert len(UNIVERSE_2013) == 54
assert len(UNIVERSE_2017) == 51

ALIASES = {
    "MHFI": "SPGI",
    "WAG": "WBA",
}

WINDOWS = {
    "BULL_2013_2014": {
        "universe": UNIVERSE_2013,
        "start": pd.Timestamp("2013-01-22"),
        "end": pd.Timestamp("2014-12-31"),
        "preferred_source": "legacy",
    },
    "BULL_2017": {
        "universe": UNIVERSE_2017,
        "start": pd.Timestamp("2017-01-03"),
        "end": pd.Timestamp("2017-12-29"),
        "preferred_source": "yfinance",
    },
}

COOLDOWN = 20
HORIZONS = (5, 10, 15, 20)

# Yahoo download range includes warm-up and forward-evaluation buffer.
YF_START = "2011-01-01"
YF_END = "2018-03-01"


# ---------------------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------------------

def get_text(url: str) -> str:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text


def normalize_ohlc(d: pd.DataFrame) -> pd.DataFrame | None:
    if d is None or d.empty:
        return None

    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)

    need = ["Open", "High", "Low", "Close"]
    if any(c not in d.columns for c in need):
        return None

    d = d[need].copy()
    d.index = pd.to_datetime(d.index).tz_localize(None)
    d = d.sort_index().loc[~d.index.duplicated(keep="first")]

    for c in need:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna(subset=need)
    return d if not d.empty else None


def load_legacy(ticker: str, etf: bool = False) -> pd.DataFrame | None:
    folder = "ETFs" if etf else "Stocks"
    data_ticker = ALIASES.get(ticker, ticker)
    url = f"{RAW_BASE}/{folder}/{data_ticker.lower()}.us.txt"

    try:
        txt = get_text(url)
        d = pd.read_csv(io.StringIO(txt))
    except Exception:
        return None

    if "Date" not in d.columns:
        return None

    d["Date"] = pd.to_datetime(d["Date"])
    d = d.sort_values("Date").drop_duplicates("Date").set_index("Date")
    return normalize_ohlc(d)


def load_yfinance(ticker: str) -> pd.DataFrame | None:
    data_ticker = ALIASES.get(ticker, ticker)
    try:
        d = yf.download(
            data_ticker,
            start=YF_START,
            end=YF_END,
            auto_adjust=True,
            progress=False,
            actions=False,
            threads=False,
        )
    except Exception:
        return None

    return normalize_ohlc(d)


def load_symbol(ticker: str, preferred_source: str, etf: bool = False):
    # Keep each window on one preferred source where possible.
    # Fallback is data plumbing only and is explicitly recorded in coverage.
    if preferred_source == "legacy":
        d = load_legacy(ticker, etf=etf)
        if d is not None:
            return d, "legacy"
        if not etf:
            d = load_yfinance(ticker)
            if d is not None:
                return d, "yfinance_fallback"
    else:
        if not etf:
            d = load_yfinance(ticker)
            if d is not None:
                return d, "yfinance"
        else:
            d = load_yfinance(ticker)
            if d is not None:
                return d, "yfinance"
        d = load_legacy(ticker, etf=etf)
        if d is not None:
            return d, "legacy_fallback"

    return None, "missing"


# ---------------------------------------------------------------------
# FROZEN V0.1
# ---------------------------------------------------------------------

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
    x["STOCH_K"] = fast_k.rolling(3).mean()   # slow K
    x["STOCH_D"] = x["STOCH_K"].rolling(3).mean()
    return x


def load_spy(preferred_source: str) -> pd.DataFrame:
    s, source = load_symbol("SPY", preferred_source, etf=True)
    if s is None:
        raise RuntimeError(f"SPY could not be loaded for {preferred_source}")

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
    s.attrs["source"] = source
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


def base_and_macd_masks(x: pd.DataFrame):
    trend = x["MA50"] > x["MA200"]
    market = x["REGIME"].isin(["GREEN", "YELLOW"])
    stoch_cross = (x["STOCH_K"].shift(1) <= 20) & (x["STOCH_K"] > 20)

    h = x["MACD_HIST"]
    macd_improving = (h < 0) & (h > h.shift(1)) & (h.shift(1) > h.shift(2))

    base = (trend & market & stoch_cross).fillna(False)
    v01 = (base & macd_improving).fillna(False)
    return base, v01, macd_improving.fillna(False)


def spaced(mask: pd.Series):
    raw = np.flatnonzero(mask.to_numpy())
    out, last = [], -10**9
    for i in raw:
        if i - last >= COOLDOWN:
            out.append(int(i))
            last = int(i)
    return out


def swing_low(x: pd.DataFrame, i: int):
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


def evaluate(x: pd.DataFrame, i: int, ticker: str, kind: str, source: str):
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
        "data_source": source,
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


def run_window(window_name: str, cfg: dict):
    start, end = cfg["start"], cfg["end"]
    preferred = cfg["preferred_source"]
    universe = cfg["universe"]

    spy = load_spy(preferred)

    coverage = []
    standard_events = []
    common_events = []

    for ticker in universe:
        d, source = load_symbol(ticker, preferred, etf=False)
        if d is None:
            coverage.append({
                "window": window_name,
                "ticker": ticker,
                "loaded": False,
                "source": source,
            })
            continue

        coverage.append({
            "window": window_name,
            "ticker": ticker,
            "loaded": True,
            "source": source,
            "first": d.index.min(),
            "last": d.index.max(),
            "rows": len(d),
        })

        x = attach_spy(add_indicators(d), spy)
        base, v01, macd_improving = base_and_macd_masks(x)

        # Standard historical V0.1 and STOCH_ONLY_CONTROL:
        # cooldown remains separately applied exactly as in the earlier tests.
        for kind, mask in [
            ("V0.1", v01),
            ("STOCH_ONLY_CONTROL", base),
        ]:
            for i in spaced(mask):
                if start <= x.index[i] <= end:
                    e = evaluate(x, i, ticker, kind, source)
                    if e is not None:
                        e["window"] = window_name
                        standard_events.append(e)

        # Secondary diagnostic only: common-candidate MACD split.
        # One cooldown on the BASE stream, then label pass/fail.
        for i in spaced(base):
            if start <= x.index[i] <= end:
                label = "MACD_PASS" if bool(macd_improving.iloc[i]) else "MACD_FAIL"
                e = evaluate(x, i, ticker, label, source)
                if e is not None:
                    e["window"] = window_name
                    common_events.append(e)

        time.sleep(0.03)

    cov = pd.DataFrame(coverage)
    ev = pd.DataFrame(standard_events)
    cc = pd.DataFrame(common_events)

    if ev.empty:
        raise RuntimeError(f"No standard events generated for {window_name}")

    return cov, ev, cc, spy.attrs.get("source", "unknown")


def write_standard_outputs(window_name: str, cov: pd.DataFrame, ev: pd.DataFrame, spy_source: str):
    slug = window_name.lower()

    cov.to_csv(RESULTS/f"coverage_{slug}.csv", index=False)
    ev.sort_values(["signal_date", "ticker", "signal_type"]).to_csv(
        RESULTS/f"events_{slug}.csv", index=False
    )

    rows = []
    for kind, g in ev.groupby("signal_type"):
        r = {"signal_type": kind}
        r.update(summarize(g))
        rows.append(r)
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS/f"summary_{slug}.csv", index=False)

    regime_rows = []
    for (kind, regime), g in ev.groupby(["signal_type", "regime"]):
        r = {"signal_type": kind, "regime": regime}
        r.update(summarize(g))
        regime_rows.append(r)
    pd.DataFrame(regime_rows).to_csv(
        RESULTS/f"summary_by_regime_{slug}.csv", index=False
    )

    loaded = int(cov["loaded"].sum())
    missing = ", ".join(cov.loc[~cov["loaded"], "ticker"].tolist()) or "None"

    report = [
        f"# Layer 1 — V0.1 — {window_name}\n\n",
        "**Bull-control window. This is not an options P/L backtest.**\n\n",
        "## Frozen-rule integrity\n",
        "**No V0.1 trading rule has changed.** MA50/MA200, MACD, "
        "Slow Stochastic 14,3,3, SPY regime, swing-low, next-open entry, "
        "daily-close stop and 20-session cooldown are inherited unchanged.\n\n",
        "## Data coverage\n",
        f"- Loaded **{loaded}/{len(cov)}** tickers.\n",
        f"- Missing: {missing}.\n",
        f"- SPY data source: {spy_source}.\n\n",
        "## Summary\n\n",
        "|Signal|N|Stocks|Stop %|+2% before stop|+3% before stop|"
        "+5% before stop|Avg MFE %|Avg MAE %|Avg 10d %|Avg 20d %|\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]

    for _, r in summary.iterrows():
        report.append(
            f"|{r.signal_type}|{int(r.n)}|{int(r.stocks)}|"
            f"{r.stop_rate_pct:.2f}|{r.hit2_before_stop_pct:.2f}|"
            f"{r.hit3_before_stop_pct:.2f}|{r.hit5_before_stop_pct:.2f}|"
            f"{r.avg_mfe_before_stop_pct:.2f}|{r.avg_mae_before_stop_pct:.2f}|"
            f"{r.avg_ret_10d_pct:.2f}|{r.avg_ret_20d_pct:.2f}|\n"
        )

    report += [
        "\n## Guardrails\n",
        "1. The bull control is evaluated with exactly the frozen V0.1 logic.\n",
        "2. The fixed annual universe is not reconstituted retrospectively during the test.\n",
        "3. Positive underlying results do not prove option profitability.\n",
    ]

    (RESULTS/f"REPORT_{window_name}.md").write_text(
        "".join(report), encoding="utf-8"
    )


def main():
    all_cc = []

    for window_name, cfg in WINDOWS.items():
        cov, ev, cc, spy_source = run_window(window_name, cfg)
        write_standard_outputs(window_name, cov, ev, spy_source)

        if not cc.empty:
            all_cc.append(cc)

    # Common-candidate MACD diagnostic for bull controls.
    if all_cc:
        cc = pd.concat(all_cc, ignore_index=True)
        cc.to_csv(
            RESULTS/"macd_common_candidate_bull_controls_events.csv",
            index=False
        )

        rows = []
        for (window, label), g in cc.groupby(["window", "signal_type"]):
            r = {"window": window, "macd_group": label}
            r.update(summarize(g))
            rows.append(r)

        pooled = []
        for label, g in cc.groupby("signal_type"):
            r = {"window": "ALL_BULL_CONTROLS", "macd_group": label}
            r.update(summarize(g))
            pooled.append(r)

        cc_summary = pd.DataFrame(rows + pooled)
        cc_summary.to_csv(
            RESULTS/"macd_common_candidate_bull_controls_summary.csv",
            index=False
        )

        report = [
            "# MACD common-candidate diagnostic — bull controls\n\n",
            "Diagnostic only. It does not change V0.1.\n\n",
            "|Window|Group|N|Stop %|+3%|+5%|MFE %|MAE %|Ret20 %|\n",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|\n",
        ]
        for _, r in cc_summary.iterrows():
            report.append(
                f"|{r.window}|{r.macd_group}|{int(r.n)}|"
                f"{r.stop_rate_pct:.2f}|{r.hit3_before_stop_pct:.2f}|"
                f"{r.hit5_before_stop_pct:.2f}|"
                f"{r.avg_mfe_before_stop_pct:.2f}|"
                f"{r.avg_mae_before_stop_pct:.2f}|"
                f"{r.avg_ret_20d_pct:.2f}|\n"
            )

        (RESULTS/"MACD_COMMON_CANDIDATE_BULL_CONTROLS.md").write_text(
            "".join(report), encoding="utf-8"
        )

    print("Bull-control backtests completed.")


if __name__ == "__main__":
    main()
