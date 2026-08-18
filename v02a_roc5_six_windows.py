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

# ---------------------------------------------------------------------
# V0.2-A RESEARCH QUESTION
# ---------------------------------------------------------------------
#
# V0.1 is the frozen benchmark.
#
# V0.2-A changes ONE AND ONLY ONE trading condition:
#
#     SPY_ROC5 > 0
#
# where:
#
#     SPY_ROC5 = (SPY_Close / SPY_Close.shift(5) - 1) * 100
#
# measured on the signal close.
#
# Everything else remains identical:
# - point-in-time universes
# - stock MA50 > MA200
# - SPY GREEN/YELLOW regime gate
# - Slow Stochastic 14,3,3 cross above 20
# - frozen MACD rule
# - swing-low definition
# - next-session-open entry proxy
# - daily Close < swing-low stop
# - 20-session cooldown
# - 5/10/15/20d evaluation
#
# Two analyses are produced:
#
# A) OPERATIONAL VERSION COMPARISON
#    Each complete rule set gets its own cooldown:
#       V0.1  -> spaced(V0.1 mask)
#       V0.2A -> spaced(V0.1 mask & SPY_ROC5 > 0)
#    This is how the two strategies would actually behave.
#
# B) COMMON-CANDIDATE ROC5 DIAGNOSTIC
#    Apply ONE common cooldown to the frozen V0.1 stream first, then label
#    each V0.1 candidate ROC5_PASS / ROC5_FAIL. This isolates the marginal
#    information in ROC5 without cooldown-path distortion.
#
# Neither analysis changes historical V0.1.
# ---------------------------------------------------------------------

DATA_REPO = "neo-zhao/CMSC320_Final_Tutorial_Huge_Stock_Market_Dataset"
RAW_BASE = f"https://raw.githubusercontent.com/{DATA_REPO}/main"

COOLDOWN = 20
HORIZONS = (5, 10, 15, 20)
ROC_DAYS = 5
ROC_THRESHOLD = 0.0

# ---------------------------------------------------------------------
# FROZEN POINT-IN-TIME UNIVERSES
# ---------------------------------------------------------------------

UNIVERSE_GFC = """
FHN BAC ED USB PFE BBT CMA RF FITB MO KEY CINF LEG GCI LLY KMB SNV GE JNJ BUD
PPG CLX MTB KO ROH ABT AVY VFC CBSS CB PEP PG SWK EMR MMM MCD SLM WWY ADP WMT
SHW GWW SVU FDO DOV MHP ADM STT BDX JCI LOW SIAL STR TGT WAG NUE BCR CTL PGR
""".split()

UNIVERSE_2013 = """
MMM ABT ABBV AFL APD ADM T ADP BDX BMS BF-B BCR CAH CVX CB CINF CTAS CLX KO CL
ED DOV ECL EMR XOM FDO BEN GPC HCP HRL ITW JNJ KMB LEG LOW MKC MCD MHFI MDT NUE
PNR PEP PPG PG SHW SIAL SWK SYY TROW TGT VFC GWW WAG WMT
""".split()

UNIVERSE_2015 = """
ABBV ABT ADM ADP AFL APD BCR BDX BEN BF-B CAH CB CINF CL CLX CTAS CVX DOV ECL
ED EMR FDO GPC GWW HCP HRL ITW JNJ KMB KO LEG LOW MCD MDT MHFI MKC MMM NUE PEP
PG PNR PPG SHW SIAL SWK SYY T TGT TROW VFC WBA WMT XOM
""".split()

UNIVERSE_2017 = """
ABBV ABT ADM ADP AFL APD BCR BDX BEN BF-B CAH CINF CL CLX CTAS CVX DOV ECL ED EMR
GPC GWW HRL ITW JNJ KMB KO LEG LOW MCD MDT MHFI MKC MMM NUE PEP PG PNR PPG SHW
SWK SYY T TGT TROW VFC WBA WMT XOM GD FRT
""".split()

UNIVERSE_2020 = """
ABBV ABT ADM ADP AFL ALB AMCR AOS APD ATO BDX BEN BF-B CAH CAT CB CINF CL CLX
CTAS CVX DOV ECL ED EMR ESS EXPD FRT GD GPC GWW HRL ITW JNJ KMB KO LEG LIN LOW
MCD MDT MKC MMM NUE O PBCT PEP PG PNR PPG ROP ROST SHW SPGI SWK SYY T TGT TROW
UTX VFC WBA WMT XOM
""".split()

UNIVERSE_2022 = """
ABBV ABT ADM ADP AFL ALB AMCR AOS APD ATO BDX BEN BF-B CAH CAT CB CINF CL CLX
CTAS CVX DOV ECL ED EMR ESS EXPD FRT GD GPC GWW HRL IBM ITW JNJ KMB KO LIN LOW
MCD MDT MKC MMM NEE NUE O PBCT PEP PG PNR PPG ROP SHW SPGI SWK SYY T TGT TROW
VFC WBA WMT WST XOM
""".split()

assert len(UNIVERSE_GFC) == 59
assert len(UNIVERSE_2013) == 54
assert len(UNIVERSE_2015) == 53
assert len(UNIVERSE_2017) == 51
assert len(UNIVERSE_2020) == 64
assert len(UNIVERSE_2022) == 64

WINDOWS = {
    "GFC_2007_2009": {
        "universe": UNIVERSE_GFC,
        "start": pd.Timestamp("2007-08-01"),
        "end": pd.Timestamp("2009-12-31"),
        "preferred_source": "legacy",
        "aliases": {},
        "frozen_summary": "results/summary_GFC_2007_2009.csv",
    },
    "BULL_2013_2014": {
        "universe": UNIVERSE_2013,
        "start": pd.Timestamp("2013-01-22"),
        "end": pd.Timestamp("2014-12-31"),
        "preferred_source": "legacy",
        "aliases": {"MHFI": "SPGI", "WAG": "WBA"},
        "frozen_summary": "results/summary_bull_2013_2014.csv",
    },
    "CHOP_2015_2016": {
        "universe": UNIVERSE_2015,
        "start": pd.Timestamp("2015-01-01"),
        "end": pd.Timestamp("2016-12-31"),
        "preferred_source": "legacy",
        "aliases": {"MHFI": "SPGI"},
        "frozen_summary": "results/summary_2015_2016.csv",
    },
    "BULL_2017": {
        "universe": UNIVERSE_2017,
        "start": pd.Timestamp("2017-01-03"),
        "end": pd.Timestamp("2017-12-29"),
        "preferred_source": "yfinance",
        "aliases": {"MHFI": "SPGI"},
        "frozen_summary": "results/summary_bull_2017.csv",
    },
    "COVID_2020": {
        "universe": UNIVERSE_2020,
        "start": pd.Timestamp("2020-01-02"),
        "end": pd.Timestamp("2020-09-30"),
        "preferred_source": "yfinance",
        "aliases": {"UTX": "RTX"},
        "frozen_summary": "results/summary_covid_2020.csv",
    },
    "BEAR_2022": {
        "universe": UNIVERSE_2022,
        "start": pd.Timestamp("2022-01-03"),
        "end": pd.Timestamp("2022-12-30"),
        "preferred_source": "yfinance",
        "aliases": {},
        "frozen_summary": "results/summary_2022.csv",
    },
}

# Enough history for MA200 and enough forward buffer for 20d evaluation.
YF_START = "2005-01-01"
YF_END = "2023-03-01"


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
    d.index = pd.to_datetime(d.index)
    try:
        d.index = d.index.tz_localize(None)
    except TypeError:
        pass

    d = d.sort_index()
    d = d.loc[~d.index.duplicated(keep="first")]

    for c in need:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna(subset=need)
    return d if not d.empty else None


def load_legacy(ticker: str, aliases: dict, etf: bool = False):
    folder = "ETFs" if etf else "Stocks"
    data_ticker = aliases.get(ticker, ticker)
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


def load_yfinance(ticker: str, aliases: dict):
    data_ticker = aliases.get(ticker, ticker)

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


def load_symbol(ticker: str, cfg: dict, etf: bool = False):
    preferred = cfg["preferred_source"]
    aliases = cfg["aliases"]

    if preferred == "legacy":
        d = load_legacy(ticker, aliases, etf=etf)
        if d is not None:
            return d, "legacy"

        # Only a fallback for data availability. It is recorded.
        if not etf:
            d = load_yfinance(ticker, aliases)
            if d is not None:
                return d, "yfinance_fallback"
    else:
        d = load_yfinance(ticker, aliases)
        if d is not None:
            return d, "yfinance"

        # Useful mainly for 2017 if Yahoo no longer serves an acquired name.
        d = load_legacy(ticker, aliases, etf=etf)
        if d is not None:
            return d, "legacy_fallback"

    return None, "missing"


# ---------------------------------------------------------------------
# FROZEN INDICATORS + NEW ROC5 FIELD
# ---------------------------------------------------------------------

def add_stock_indicators(d: pd.DataFrame) -> pd.DataFrame:
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


def load_spy(cfg: dict):
    s, source = load_symbol("SPY", cfg, etf=True)
    if s is None:
        raise RuntimeError("SPY could not be loaded")

    s = add_stock_indicators(s)

    s["REGIME"] = np.select(
        [
            s["MA50"] <= s["MA200"],
            (s["MA50"] > s["MA200"]) & (s["Close"] <= s["MA200"]),
            (s["MA50"] > s["MA200"]) & (s["Close"] > s["MA200"]),
        ],
        ["RED", "YELLOW", "GREEN"],
        default="UNKNOWN",
    )

    # ONLY NEW V0.2-A VARIABLE.
    s["ROC5_PCT"] = (s["Close"] / s["Close"].shift(ROC_DAYS) - 1.0) * 100.0
    s["ROC5_PASS"] = s["ROC5_PCT"] > ROC_THRESHOLD

    s.attrs["source"] = source
    return s


def attach_spy(x: pd.DataFrame, spy: pd.DataFrame):
    r = spy[
        ["Close", "MA50", "MA200", "REGIME", "ROC5_PCT", "ROC5_PASS"]
    ].rename(
        columns={
            "Close": "SPY_CLOSE",
            "MA50": "SPY_MA50",
            "MA200": "SPY_MA200",
            "ROC5_PCT": "SPY_ROC5_PCT",
            "ROC5_PASS": "SPY_ROC5_PASS",
        }
    )

    return x.join(r, how="left").ffill()


def masks(x: pd.DataFrame):
    trend = x["MA50"] > x["MA200"]
    market = x["REGIME"].isin(["GREEN", "YELLOW"])
    stoch_cross = (x["STOCH_K"].shift(1) <= 20) & (x["STOCH_K"] > 20)

    h = x["MACD_HIST"]
    macd_improving = (h < 0) & (h > h.shift(1)) & (h.shift(1) > h.shift(2))

    # Frozen V0.1.
    v01 = (trend & market & stoch_cross & macd_improving).fillna(False)

    # V0.2-A: ONE extra condition only.
    v02a = (v01 & (x["SPY_ROC5_PCT"] > ROC_THRESHOLD)).fillna(False)

    return v01, v02a


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


def evaluate(
    x: pd.DataFrame,
    i: int,
    ticker: str,
    signal_type: str,
    source: str,
    window: str,
):
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
        "window": window,
        "ticker": ticker,
        "data_source": source,
        "signal_type": signal_type,
        "signal_date": x.index[i],
        "entry_date": x.index[entry_i],
        "entry_open": entry,
        "regime": x["REGIME"].iloc[i],
        "spy_close": x["SPY_CLOSE"].iloc[i],
        "spy_ma50": x["SPY_MA50"].iloc[i],
        "spy_ma200": x["SPY_MA200"].iloc[i],
        "spy_roc5_pct": x["SPY_ROC5_PCT"].iloc[i],
        "spy_roc5_pass": bool(x["SPY_ROC5_PCT"].iloc[i] > ROC_THRESHOLD),
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
        z[f"ret_{h}d_pct"] = (
            x["Close"].iloc[entry_i+h] / entry - 1
        ) * 100

    for target in (2, 3, 5):
        hits = np.flatnonzero(
            (effective["High"] >= entry*(1+target/100)).to_numpy()
        )
        z[f"hit_{target}pct_before_stop"] = bool(len(hits))
        z[f"days_to_{target}pct"] = (
            int(hits[0]) if len(hits) else np.nan
        )

    return z


# ---------------------------------------------------------------------
# SUMMARIES
# ---------------------------------------------------------------------

def summarize(g: pd.DataFrame):
    if len(g) == 0:
        return {}

    return {
        "n": len(g),
        "stocks": g["ticker"].nunique(),
        "stop_rate_pct": 100 * g["stop_hit_20d"].mean(),
        "hit2_before_stop_pct": 100 * g["hit_2pct_before_stop"].mean(),
        "hit3_before_stop_pct": 100 * g["hit_3pct_before_stop"].mean(),
        "hit5_before_stop_pct": 100 * g["hit_5pct_before_stop"].mean(),
        "avg_mfe_before_stop_pct": g["mfe_before_stop_pct"].mean(),
        "avg_mae_before_stop_pct": g["mae_before_stop_pct"].mean(),
        "avg_ret_5d_pct": g["ret_5d_pct"].mean(),
        "avg_ret_10d_pct": g["ret_10d_pct"].mean(),
        "avg_ret_15d_pct": g["ret_15d_pct"].mean(),
        "avg_ret_20d_pct": g["ret_20d_pct"].mean(),
        "median_swing_distance_pct": g["swing_distance_pct"].median(),
    }


def summarize_frame(ev: pd.DataFrame, group_cols: list[str]):
    rows = []

    for keys, g in ev.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        r = dict(zip(group_cols, keys))
        r.update(summarize(g))
        rows.append(r)

    return pd.DataFrame(rows)


def version_delta_rows(summary_by_window: pd.DataFrame):
    rows = []

    for window, g in summary_by_window.groupby("window"):
        idx = g.set_index("signal_type")

        if "V0.1" not in idx.index or "V0.2A_ROC5" not in idx.index:
            continue

        a = idx.loc["V0.1"]
        b = idx.loc["V0.2A_ROC5"]

        rows.append({
            "window": window,
            "v01_n": int(a["n"]),
            "v02a_n": int(b["n"]),
            "signal_retention_pct": 100 * b["n"] / a["n"] if a["n"] else np.nan,
            "delta_stop_pp": b["stop_rate_pct"] - a["stop_rate_pct"],
            "delta_hit3_pp": b["hit3_before_stop_pct"] - a["hit3_before_stop_pct"],
            "delta_hit5_pp": b["hit5_before_stop_pct"] - a["hit5_before_stop_pct"],
            "delta_mfe_pp": b["avg_mfe_before_stop_pct"] - a["avg_mfe_before_stop_pct"],
            "delta_mae_pp": b["avg_mae_before_stop_pct"] - a["avg_mae_before_stop_pct"],
            "delta_ret20_pp": b["avg_ret_20d_pct"] - a["avg_ret_20d_pct"],
        })

    return pd.DataFrame(rows)


def common_lift_rows(common_summary: pd.DataFrame):
    rows = []

    for window, g in common_summary.groupby("window"):
        idx = g.set_index("roc5_group")

        if "ROC5_PASS" not in idx.index or "ROC5_FAIL" not in idx.index:
            continue

        p = idx.loc["ROC5_PASS"]
        f = idx.loc["ROC5_FAIL"]
        total = p["n"] + f["n"]

        rows.append({
            "window": window,
            "pass_n": int(p["n"]),
            "fail_n": int(f["n"]),
            "pass_share_pct": 100 * p["n"] / total if total else np.nan,
            "delta_stop_pp": p["stop_rate_pct"] - f["stop_rate_pct"],
            "delta_hit3_pp": p["hit3_before_stop_pct"] - f["hit3_before_stop_pct"],
            "delta_hit5_pp": p["hit5_before_stop_pct"] - f["hit5_before_stop_pct"],
            "delta_mfe_pp": p["avg_mfe_before_stop_pct"] - f["avg_mfe_before_stop_pct"],
            "delta_mae_pp": p["avg_mae_before_stop_pct"] - f["avg_mae_before_stop_pct"],
            "delta_ret20_pp": p["avg_ret_20d_pct"] - f["avg_ret_20d_pct"],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# CLUSTER BOOTSTRAP
# ---------------------------------------------------------------------

def metric_delta(g: pd.DataFrame, metric: str):
    p = g[g["roc5_group"] == "ROC5_PASS"]
    f = g[g["roc5_group"] == "ROC5_FAIL"]

    if len(p) == 0 or len(f) == 0:
        return np.nan

    if metric == "stop":
        return 100*p["stop_hit_20d"].mean() - 100*f["stop_hit_20d"].mean()
    if metric == "hit3":
        return 100*p["hit_3pct_before_stop"].mean() - 100*f["hit_3pct_before_stop"].mean()
    if metric == "hit5":
        return 100*p["hit_5pct_before_stop"].mean() - 100*f["hit_5pct_before_stop"].mean()
    if metric == "ret20":
        return p["ret_20d_pct"].mean() - f["ret_20d_pct"].mean()

    raise ValueError(metric)


def cluster_bootstrap(g: pd.DataFrame, reps: int = 2000, seed: int = 20260818):
    if g.empty:
        return {}

    x = g.copy()
    x["cluster_key"] = (
        x["window"].astype(str)
        + "__"
        + pd.to_datetime(x["signal_date"]).dt.strftime("%Y-%m-%d")
    )

    clusters = x["cluster_key"].drop_duplicates().to_numpy()

    if len(clusters) < 2:
        return {}

    rng = np.random.default_rng(seed)
    store = {m: [] for m in ("stop", "hit3", "hit5", "ret20")}

    grouped = {k: v.copy() for k, v in x.groupby("cluster_key")}

    for _ in range(reps):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)

        pieces = []
        for j, key in enumerate(sampled):
            piece = grouped[key].copy()
            # Preserve duplicate sampled clusters as independent bootstrap draws.
            piece["bootstrap_draw"] = j
            pieces.append(piece)

        b = pd.concat(pieces, ignore_index=True)

        for m in store:
            d = metric_delta(b, m)
            if np.isfinite(d):
                store[m].append(d)

    out = {}

    for m, values in store.items():
        if values:
            arr = np.asarray(values)
            out[f"delta_{m}_low95"] = np.quantile(arr, 0.025)
            out[f"delta_{m}_high95"] = np.quantile(arr, 0.975)
        else:
            out[f"delta_{m}_low95"] = np.nan
            out[f"delta_{m}_high95"] = np.nan

    return out


# ---------------------------------------------------------------------
# FROZEN BENCHMARK INTEGRITY CHECK
# ---------------------------------------------------------------------

def read_frozen_v01(path_str: str):
    p = Path(path_str)

    if not p.exists():
        return None

    try:
        d = pd.read_csv(p)
    except Exception:
        return None

    if "signal_type" not in d.columns:
        return None

    row = d[d["signal_type"] == "V0.1"]

    if row.empty:
        return None

    return row.iloc[0]


def benchmark_check(window: str, cfg: dict, rerun_summary: pd.DataFrame):
    frozen = read_frozen_v01(cfg["frozen_summary"])

    r = rerun_summary[
        (rerun_summary["window"] == window)
        & (rerun_summary["signal_type"] == "V0.1")
    ]

    if frozen is None or r.empty:
        return {
            "window": window,
            "frozen_summary_found": False,
        }

    cur = r.iloc[0]

    metrics = [
        ("n", "n"),
        ("stop_rate_pct", "stop_rate_pct"),
        ("hit3_before_stop_pct", "hit3_before_stop_pct"),
        ("hit5_before_stop_pct", "hit5_before_stop_pct"),
        ("avg_mfe_before_stop_pct", "avg_mfe_before_stop_pct"),
        ("avg_mae_before_stop_pct", "avg_mae_before_stop_pct"),
        ("avg_ret_20d_pct", "avg_ret_20d_pct"),
    ]

    out = {
        "window": window,
        "frozen_summary_found": True,
    }

    max_abs_diff = 0.0

    for cur_col, frozen_col in metrics:
        if frozen_col not in frozen.index:
            continue

        a = float(cur[cur_col])
        b = float(frozen[frozen_col])
        diff = a - b

        out[f"rerun_{cur_col}"] = a
        out[f"frozen_{cur_col}"] = b
        out[f"diff_{cur_col}"] = diff

        if cur_col != "n":
            max_abs_diff = max(max_abs_diff, abs(diff))

    out["exact_n_match"] = (
        int(round(float(cur["n"]))) == int(round(float(frozen["n"])))
    )
    out["max_abs_metric_diff"] = max_abs_diff

    # Descriptive flag only.
    out["benchmark_close_match"] = bool(
        out["exact_n_match"] and max_abs_diff <= 0.05
    )

    return out


# ---------------------------------------------------------------------
# WINDOW RUNNER
# ---------------------------------------------------------------------

def run_window(window: str, cfg: dict):
    spy = load_spy(cfg)
    coverage = []
    operational = []
    common = []

    for ticker in cfg["universe"]:
        d, source = load_symbol(ticker, cfg, etf=False)

        if d is None:
            coverage.append({
                "window": window,
                "ticker": ticker,
                "loaded": False,
                "source": source,
            })
            continue

        coverage.append({
            "window": window,
            "ticker": ticker,
            "loaded": True,
            "source": source,
            "first": d.index.min(),
            "last": d.index.max(),
            "rows": len(d),
        })

        x = attach_spy(add_stock_indicators(d), spy)
        v01, v02a = masks(x)

        # A) Operational comparison: each full strategy has its own cooldown.
        for signal_type, mask in [
            ("V0.1", v01),
            ("V0.2A_ROC5", v02a),
        ]:
            for i in spaced(mask):
                if cfg["start"] <= x.index[i] <= cfg["end"]:
                    e = evaluate(
                        x, i, ticker, signal_type, source, window
                    )
                    if e is not None:
                        operational.append(e)

        # B) Common-candidate diagnostic:
        # one common cooldown on frozen V0.1 first, then ROC5 label.
        for i in spaced(v01):
            if cfg["start"] <= x.index[i] <= cfg["end"]:
                label = (
                    "ROC5_PASS"
                    if x["SPY_ROC5_PCT"].iloc[i] > ROC_THRESHOLD
                    else "ROC5_FAIL"
                )

                e = evaluate(
                    x, i, ticker, label, source, window
                )

                if e is not None:
                    e["roc5_group"] = label
                    common.append(e)

        time.sleep(0.03)

    return (
        pd.DataFrame(coverage),
        pd.DataFrame(operational),
        pd.DataFrame(common),
        spy.attrs.get("source", "unknown"),
    )


# ---------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------

def main():
    all_coverage = []
    all_operational = []
    all_common = []
    spy_sources = {}

    for window, cfg in WINDOWS.items():
        cov, op, cc, spy_source = run_window(window, cfg)

        if op.empty:
            raise RuntimeError(f"No operational events for {window}")

        all_coverage.append(cov)
        all_operational.append(op)
        all_common.append(cc)
        spy_sources[window] = spy_source

    coverage = pd.concat(all_coverage, ignore_index=True)
    operational = pd.concat(all_operational, ignore_index=True)
    common = pd.concat(all_common, ignore_index=True)

    operational = operational.sort_values(
        ["window", "signal_date", "ticker", "signal_type"]
    )
    common = common.sort_values(
        ["window", "signal_date", "ticker", "roc5_group"]
    )

    coverage.to_csv(
        RESULTS/"v02a_roc5_coverage_all_windows.csv", index=False
    )
    operational.to_csv(
        RESULTS/"v02a_roc5_operational_events_all_windows.csv", index=False
    )
    common.to_csv(
        RESULTS/"v02a_roc5_common_candidate_events.csv", index=False
    )

    # Operational summaries.
    op_summary = summarize_frame(
        operational, ["window", "signal_type"]
    )
    op_summary.to_csv(
        RESULTS/"v02a_roc5_operational_summary.csv", index=False
    )

    op_regime = summarize_frame(
        operational, ["window", "signal_type", "regime"]
    )
    op_regime.to_csv(
        RESULTS/"v02a_roc5_operational_summary_by_regime.csv",
        index=False,
    )

    version_delta = version_delta_rows(op_summary)
    version_delta.to_csv(
        RESULTS/"v02a_roc5_operational_delta.csv", index=False
    )

    # Add pooled operational summary.
    pooled_op = summarize_frame(operational, ["signal_type"])
    pooled_op.insert(0, "window", "ALL_WINDOWS_POOLED")
    pooled_op.to_csv(
        RESULTS/"v02a_roc5_operational_pooled.csv", index=False
    )

    # Common-candidate summaries.
    cc_summary = summarize_frame(
        common, ["window", "roc5_group"]
    )

    pooled_cc = summarize_frame(common, ["roc5_group"])
    pooled_cc.insert(0, "window", "ALL_WINDOWS_POOLED")

    cc_full = pd.concat(
        [cc_summary, pooled_cc],
        ignore_index=True,
    )

    cc_full.to_csv(
        RESULTS/"v02a_roc5_common_candidate_summary.csv", index=False
    )

    lift = common_lift_rows(cc_full)
    lift.to_csv(
        RESULTS/"v02a_roc5_common_candidate_lift.csv", index=False
    )

    # Cluster-bootstrap intervals.
    boot_rows = []

    for window, g in common.groupby("window"):
        row = {"window": window}
        row.update(cluster_bootstrap(g))
        boot_rows.append(row)

    pooled = common.copy()
    row = {"window": "ALL_WINDOWS_POOLED"}
    row.update(cluster_bootstrap(pooled))
    boot_rows.append(row)

    boot = pd.DataFrame(boot_rows)
    boot.to_csv(
        RESULTS/"v02a_roc5_cluster_bootstrap.csv", index=False
    )

    # Benchmark integrity checks.
    checks = []

    for window, cfg in WINDOWS.items():
        checks.append(
            benchmark_check(window, cfg, op_summary)
        )

    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(
        RESULTS/"v02a_roc5_frozen_benchmark_check.csv", index=False
    )

    # Candidate-accounting output:
    # What V0.2-A filters out from the common V0.1 candidate stream.
    rejected = common[common["roc5_group"] == "ROC5_FAIL"].copy()
    accepted = common[common["roc5_group"] == "ROC5_PASS"].copy()

    rejected.to_csv(
        RESULTS/"v02a_roc5_rejected_v01_candidates.csv", index=False
    )
    accepted.to_csv(
        RESULTS/"v02a_roc5_accepted_v01_candidates.csv", index=False
    )

    # Markdown report.
    report = [
        "# V0.2-A research — SPY ROC5 > 0 vs frozen V0.1\n\n",
        "## Status\n",
        "**Research candidate only. V0.1 remains the frozen benchmark.**\n\n",
        "## Only rule change\n",
        "`V0.2-A = V0.1 AND SPY_ROC5 > 0`, where "
        "`SPY_ROC5 = (SPY Close / SPY Close 5 sessions ago - 1) × 100`.\n\n",
        "No MACD, Stochastic, MA50/MA200, regime, swing-low, entry, stop, "
        "cooldown or evaluation rule is changed.\n\n",
        "## Why two comparisons are reported\n",
        "1. **Operational:** V0.1 and V0.2-A each apply their own cooldown, "
        "which is how the actual rule sets would trade.\n",
        "2. **Common-candidate diagnostic:** one cooldown is applied to the "
        "frozen V0.1 candidate stream first; candidates are then labelled "
        "ROC5_PASS/ROC5_FAIL. This isolates ROC5 from cooldown-path effects.\n\n",
        "## Operational V0.1 vs V0.2-A\n\n",
        "|Window|V0.1 N|V0.2A N|Retention %|Δ Stop pp|Δ +3 pp|"
        "Δ +5 pp|Δ MFE pp|Δ MAE pp|Δ Ret20 pp|\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]

    for _, r in version_delta.iterrows():
        report.append(
            f"|{r.window}|{int(r.v01_n)}|{int(r.v02a_n)}|"
            f"{r.signal_retention_pct:.2f}|{r.delta_stop_pp:.2f}|"
            f"{r.delta_hit3_pp:.2f}|{r.delta_hit5_pp:.2f}|"
            f"{r.delta_mfe_pp:.2f}|{r.delta_mae_pp:.2f}|"
            f"{r.delta_ret20_pp:.2f}|\n"
        )

    report += [
        "\n## Common-candidate ROC5 lift\n\n",
        "A negative `Δ Stop` favors ROC5_PASS. Positive `Δ +3`, `Δ +5`, "
        "`Δ MFE`, `Δ MAE` (less-negative MAE), and `Δ Ret20` favor ROC5_PASS.\n\n",
        "|Window|Pass N|Fail N|Pass %|Δ Stop pp|Δ +3 pp|Δ +5 pp|"
        "Δ MFE pp|Δ MAE pp|Δ Ret20 pp|\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]

    for _, r in lift.iterrows():
        report.append(
            f"|{r.window}|{int(r.pass_n)}|{int(r.fail_n)}|"
            f"{r.pass_share_pct:.2f}|{r.delta_stop_pp:.2f}|"
            f"{r.delta_hit3_pp:.2f}|{r.delta_hit5_pp:.2f}|"
            f"{r.delta_mfe_pp:.2f}|{r.delta_mae_pp:.2f}|"
            f"{r.delta_ret20_pp:.2f}|\n"
        )

    report += [
        "\n## Cluster-bootstrap 95% intervals — common candidates\n\n",
        "Signal dates are resampled as clusters so that same-day market "
        "signal bursts are not treated as independent observations.\n\n",
        "|Window|ΔStop low|ΔStop high|Δ+3 low|Δ+3 high|"
        "Δ+5 low|Δ+5 high|ΔRet20 low|ΔRet20 high|\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]

    for _, r in boot.iterrows():
        report.append(
            f"|{r.window}|"
            f"{r.get('delta_stop_low95', np.nan):.2f}|"
            f"{r.get('delta_stop_high95', np.nan):.2f}|"
            f"{r.get('delta_hit3_low95', np.nan):.2f}|"
            f"{r.get('delta_hit3_high95', np.nan):.2f}|"
            f"{r.get('delta_hit5_low95', np.nan):.2f}|"
            f"{r.get('delta_hit5_high95', np.nan):.2f}|"
            f"{r.get('delta_ret20_low95', np.nan):.2f}|"
            f"{r.get('delta_ret20_high95', np.nan):.2f}|\n"
        )

    report += [
        "\n## Frozen-benchmark integrity\n\n",
        "The script re-runs V0.1 beside V0.2-A and compares the result with "
        "the already-saved frozen V0.1 summary files. This detects data-source "
        "drift, especially in Yahoo-adjusted historical data.\n\n",
        "|Window|Frozen found|N exact|Max metric diff|Close match|\n",
        "|---|---|---|---:|---|\n",
    ]

    for _, r in checks_df.iterrows():
        if not bool(r.get("frozen_summary_found", False)):
            report.append(
                f"|{r.window}|No|—|—|—|\n"
            )
        else:
            report.append(
                f"|{r.window}|Yes|{bool(r.get('exact_n_match', False))}|"
                f"{r.get('max_abs_metric_diff', np.nan):.4f}|"
                f"{bool(r.get('benchmark_close_match', False))}|\n"
            )

    report += [
        "\n## Pre-specified interpretation guardrails\n",
        "- Do **not** accept V0.2-A because the pooled average improves alone.\n",
        "- The candidate should reduce transition/chop damage in 2015–16, "
        "COVID and 2022 without materially destroying GFC, 2013–14 or 2017.\n",
        "- Signal retention matters: a filter that deletes most opportunities "
        "may improve averages while making the system economically irrelevant.\n",
        "- The rejected-candidate file must be inspected for good rebounds "
        "destroyed by the filter, especially +3%/+5% before stop.\n",
        "- A single crisis cluster must not determine acceptance.\n",
        "- Positive Layer-1 performance still does not prove PUT-option profitability.\n",
        "- MACD remains frozen in this experiment. Its role belongs to a later "
        "V0.2-B experiment only.\n",
    ]

    (RESULTS/"V02A_ROC5_REPORT.md").write_text(
        "".join(report), encoding="utf-8"
    )

    print("".join(report))


if __name__ == "__main__":
    main()
