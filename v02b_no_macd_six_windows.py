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
# V0.2-B RESEARCH QUESTION
# ---------------------------------------------------------------------
#
# V0.1 is the frozen benchmark.
#
# V0.2-B changes ONE AND ONLY ONE trading condition:
#     MACD is no longer a mandatory entry filter.
#
# V0.2-B candidate stream therefore is exactly:
#     stock MA50 > MA200
#     SPY regime in GREEN/YELLOW
#     Slow Stochastic 14,3,3 crosses upward through 20
#
# Everything else remains identical:
# - point-in-time universes
# - swing-low definition
# - next-session-open entry proxy
# - daily Close < swing-low stop
# - 20-session cooldown
# - 5/10/15/20d evaluation
#
# MACD remains calculated and recorded as a DIAGNOSTIC variable only.
#
# A) OPERATIONAL VERSION COMPARISON
#    Each complete rule set gets its own cooldown:
#       V0.1  -> spaced(base & MACD_PASS)
#       V0.2B -> spaced(base)
#
# B) COMMON-CANDIDATE MACD DIAGNOSTIC
#    Apply ONE common cooldown to V0.2-B/base first, then label each
#    candidate MACD_PASS / MACD_FAIL. This isolates the marginal
#    information in MACD without cooldown-path distortion.
# ---------------------------------------------------------------------

DATA_REPO = "neo-zhao/CMSC320_Final_Tutorial_Huge_Stock_Market_Dataset"
RAW_BASE = f"https://raw.githubusercontent.com/{DATA_REPO}/main"

COOLDOWN = 20
HORIZONS = (5, 10, 15, 20)

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

YF_START = "2005-01-01"
YF_END = "2023-03-01"


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
        if not etf:
            d = load_yfinance(ticker, aliases)
            if d is not None:
                return d, "yfinance_fallback"
    else:
        d = load_yfinance(ticker, aliases)
        if d is not None:
            return d, "yfinance"
        d = load_legacy(ticker, aliases, etf=etf)
        if d is not None:
            return d, "legacy_fallback"
    return None, "missing"


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
    x["STOCH_K"] = fast_k.rolling(3).mean()
    x["STOCH_D"] = x["STOCH_K"].rolling(3).mean()
    return x


def load_spy(cfg: dict):
    s, source = load_symbol("SPY", cfg, etf=True)
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
    s.attrs["source"] = source
    return s


def attach_spy(x: pd.DataFrame, spy: pd.DataFrame):
    r = spy[["Close", "MA50", "MA200", "REGIME"]].rename(
        columns={
            "Close": "SPY_CLOSE",
            "MA50": "SPY_MA50",
            "MA200": "SPY_MA200",
        }
    )
    return x.join(r, how="left").ffill()


def masks(x: pd.DataFrame):
    trend = x["MA50"] > x["MA200"]
    market = x["REGIME"].isin(["GREEN", "YELLOW"])
    stoch_cross = (x["STOCH_K"].shift(1) <= 20) & (x["STOCH_K"] > 20)

    h = x["MACD_HIST"]
    macd_pass = (h < 0) & (h > h.shift(1)) & (h.shift(1) > h.shift(2))

    base = (trend & market & stoch_cross).fillna(False)
    v01 = (base & macd_pass).fillna(False)
    v02b = base.copy()
    return v01, v02b, macd_pass.fillna(False)


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


def evaluate(x, i, ticker, signal_type, source, window):
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

    h = float(x["MACD_HIST"].iloc[i]) if pd.notna(x["MACD_HIST"].iloc[i]) else np.nan
    h1 = float(x["MACD_HIST"].iloc[i-1]) if i >= 1 and pd.notna(x["MACD_HIST"].iloc[i-1]) else np.nan
    h2 = float(x["MACD_HIST"].iloc[i-2]) if i >= 2 and pd.notna(x["MACD_HIST"].iloc[i-2]) else np.nan
    macd_pass = bool(np.isfinite(h) and np.isfinite(h1) and np.isfinite(h2) and h < 0 and h > h1 > h2)

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
        "macd_hist": h,
        "macd_hist_1": h1,
        "macd_hist_2": h2,
        "macd_pass": macd_pass,
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
    for horizon in HORIZONS:
        z[f"ret_{horizon}d_pct"] = (
            x["Close"].iloc[entry_i+horizon] / entry - 1
        ) * 100
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


def summarize_frame(ev: pd.DataFrame, group_cols: list[str]):
    rows = []
    for keys, g in ev.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        r = dict(zip(group_cols, keys))
        r.update(summarize(g))
        rows.append(r)
    return pd.DataFrame(rows)


def operational_delta_rows(summary_by_window: pd.DataFrame):
    rows = []
    for window, g in summary_by_window.groupby("window"):
        idx = g.set_index("signal_type")
        if "V0.1" not in idx.index or "V0.2B_NO_MACD" not in idx.index:
            continue
        a = idx.loc["V0.1"]
        b = idx.loc["V0.2B_NO_MACD"]
        rows.append({
            "window": window,
            "v01_n": int(a["n"]),
            "v02b_n": int(b["n"]),
            "signal_expansion_pct": 100*b["n"]/a["n"] if a["n"] else np.nan,
            "extra_signals_n": int(b["n"] - a["n"]),
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
        idx = g.set_index("macd_group")
        if "MACD_PASS" not in idx.index or "MACD_FAIL" not in idx.index:
            continue
        p = idx.loc["MACD_PASS"]
        f = idx.loc["MACD_FAIL"]
        total = p["n"] + f["n"]
        rows.append({
            "window": window,
            "pass_n": int(p["n"]),
            "fail_n": int(f["n"]),
            "pass_share_pct": 100*p["n"]/total if total else np.nan,
            "delta_stop_pp": p["stop_rate_pct"] - f["stop_rate_pct"],
            "delta_hit3_pp": p["hit3_before_stop_pct"] - f["hit3_before_stop_pct"],
            "delta_hit5_pp": p["hit5_before_stop_pct"] - f["hit5_before_stop_pct"],
            "delta_mfe_pp": p["avg_mfe_before_stop_pct"] - f["avg_mfe_before_stop_pct"],
            "delta_mae_pp": p["avg_mae_before_stop_pct"] - f["avg_mae_before_stop_pct"],
            "delta_ret20_pp": p["avg_ret_20d_pct"] - f["avg_ret_20d_pct"],
        })
    return pd.DataFrame(rows)


def metric_delta(g: pd.DataFrame, metric: str):
    p = g[g["macd_group"] == "MACD_PASS"]
    f = g[g["macd_group"] == "MACD_FAIL"]
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
    grouped = {k: v.copy() for k, v in x.groupby("cluster_key")}
    store = {m: [] for m in ("stop", "hit3", "hit5", "ret20")}
    for _ in range(reps):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        pieces = []
        for j, key in enumerate(sampled):
            piece = grouped[key].copy()
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


def read_frozen_rows(path_str: str):
    p = Path(path_str)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def benchmark_check(window: str, cfg: dict, rerun_summary: pd.DataFrame):
    frozen = read_frozen_rows(cfg["frozen_summary"])
    out_rows = []
    for rerun_type, frozen_type in [
        ("V0.1", "V0.1"),
        ("V0.2B_NO_MACD", "STOCH_ONLY_CONTROL"),
    ]:
        cur = rerun_summary[
            (rerun_summary["window"] == window)
            & (rerun_summary["signal_type"] == rerun_type)
        ]
        if frozen is None or cur.empty or "signal_type" not in frozen.columns:
            out_rows.append({
                "window": window,
                "rerun_signal_type": rerun_type,
                "frozen_signal_type": frozen_type,
                "frozen_summary_found": False,
            })
            continue
        fr = frozen[frozen["signal_type"] == frozen_type]
        if fr.empty:
            out_rows.append({
                "window": window,
                "rerun_signal_type": rerun_type,
                "frozen_signal_type": frozen_type,
                "frozen_summary_found": False,
            })
            continue
        cur = cur.iloc[0]
        fr = fr.iloc[0]
        metrics = [
            "n", "stop_rate_pct", "hit3_before_stop_pct", "hit5_before_stop_pct",
            "avg_mfe_before_stop_pct", "avg_mae_before_stop_pct", "avg_ret_20d_pct",
        ]
        row = {
            "window": window,
            "rerun_signal_type": rerun_type,
            "frozen_signal_type": frozen_type,
            "frozen_summary_found": True,
        }
        max_abs_diff = 0.0
        for col in metrics:
            if col not in fr.index:
                continue
            a = float(cur[col])
            b = float(fr[col])
            row[f"rerun_{col}"] = a
            row[f"frozen_{col}"] = b
            row[f"diff_{col}"] = a - b
            if col != "n":
                max_abs_diff = max(max_abs_diff, abs(a-b))
        row["exact_n_match"] = int(round(float(cur["n"]))) == int(round(float(fr["n"])))
        row["max_abs_metric_diff"] = max_abs_diff
        row["benchmark_close_match"] = bool(row["exact_n_match"] and max_abs_diff <= 0.05)
        out_rows.append(row)
    return out_rows


def run_window(window: str, cfg: dict):
    spy = load_spy(cfg)
    coverage, operational, common = [], [], []
    for ticker in cfg["universe"]:
        d, source = load_symbol(ticker, cfg, etf=False)
        if d is None:
            coverage.append({"window": window, "ticker": ticker, "loaded": False, "source": source})
            continue
        coverage.append({
            "window": window, "ticker": ticker, "loaded": True, "source": source,
            "first": d.index.min(), "last": d.index.max(), "rows": len(d),
        })
        x = attach_spy(add_indicators(d), spy)
        v01, v02b, macd_pass = masks(x)

        # Operational comparison: each complete strategy gets its own cooldown.
        for signal_type, mask in [("V0.1", v01), ("V0.2B_NO_MACD", v02b)]:
            for i in spaced(mask):
                if cfg["start"] <= x.index[i] <= cfg["end"]:
                    e = evaluate(x, i, ticker, signal_type, source, window)
                    if e is not None:
                        operational.append(e)

        # Common-candidate diagnostic: cooldown once on no-MACD/base stream.
        for i in spaced(v02b):
            if cfg["start"] <= x.index[i] <= cfg["end"]:
                label = "MACD_PASS" if bool(macd_pass.iloc[i]) else "MACD_FAIL"
                e = evaluate(x, i, ticker, label, source, window)
                if e is not None:
                    e["macd_group"] = label
                    common.append(e)
        time.sleep(0.03)

    return pd.DataFrame(coverage), pd.DataFrame(operational), pd.DataFrame(common), spy.attrs.get("source", "unknown")


def clustering_by_day(operational: pd.DataFrame):
    rows = []
    for (window, signal_type, signal_date), g in operational.groupby(["window", "signal_type", "signal_date"]):
        rows.append({
            "window": window,
            "signal_type": signal_type,
            "signal_date": signal_date,
            "signals": len(g),
            "stocks": g["ticker"].nunique(),
            "stop_rate_pct": 100*g["stop_hit_20d"].mean(),
            "avg_ret_20d_pct": g["ret_20d_pct"].mean(),
        })
    return pd.DataFrame(rows)


def clustering_summary(day: pd.DataFrame):
    rows = []
    for (window, signal_type), g in day.groupby(["window", "signal_type"]):
        rows.append({
            "window": window,
            "signal_type": signal_type,
            "signal_days": len(g),
            "max_same_day_signals": int(g["signals"].max()),
            "days_ge_3_signals": int((g["signals"] >= 3).sum()),
            "days_ge_5_signals": int((g["signals"] >= 5).sum()),
            "days_ge_10_signals": int((g["signals"] >= 10).sum()),
            "share_signals_on_ge5_days_pct": 100*g.loc[g["signals"] >= 5, "signals"].sum()/g["signals"].sum() if g["signals"].sum() else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    all_cov, all_op, all_common = [], [], []
    for window, cfg in WINDOWS.items():
        cov, op, cc, _ = run_window(window, cfg)
        if op.empty:
            raise RuntimeError(f"No operational events for {window}")
        all_cov.append(cov)
        all_op.append(op)
        all_common.append(cc)

    coverage = pd.concat(all_cov, ignore_index=True)
    operational = pd.concat(all_op, ignore_index=True).sort_values(["window", "signal_date", "ticker", "signal_type"])
    common = pd.concat(all_common, ignore_index=True).sort_values(["window", "signal_date", "ticker", "macd_group"])

    coverage.to_csv(RESULTS/"v02b_no_macd_coverage_all_windows.csv", index=False)
    operational.to_csv(RESULTS/"v02b_no_macd_operational_events_all_windows.csv", index=False)
    common.to_csv(RESULTS/"v02b_no_macd_common_candidate_events.csv", index=False)

    op_summary = summarize_frame(operational, ["window", "signal_type"])
    op_summary.to_csv(RESULTS/"v02b_no_macd_operational_summary.csv", index=False)
    op_regime = summarize_frame(operational, ["window", "signal_type", "regime"])
    op_regime.to_csv(RESULTS/"v02b_no_macd_operational_summary_by_regime.csv", index=False)

    op_delta = operational_delta_rows(op_summary)
    op_delta.to_csv(RESULTS/"v02b_no_macd_operational_delta.csv", index=False)

    pooled_op = summarize_frame(operational, ["signal_type"])
    pooled_op.insert(0, "window", "ALL_WINDOWS_POOLED")
    pooled_op.to_csv(RESULTS/"v02b_no_macd_operational_pooled.csv", index=False)

    cc_summary = summarize_frame(common, ["window", "macd_group"])
    pooled_cc = summarize_frame(common, ["macd_group"])
    pooled_cc.insert(0, "window", "ALL_WINDOWS_POOLED")
    cc_full = pd.concat([cc_summary, pooled_cc], ignore_index=True)
    cc_full.to_csv(RESULTS/"v02b_no_macd_common_candidate_summary.csv", index=False)

    lift = common_lift_rows(cc_full)
    lift.to_csv(RESULTS/"v02b_no_macd_common_candidate_lift.csv", index=False)

    boot_rows = []
    for window, g in common.groupby("window"):
        row = {"window": window}
        row.update(cluster_bootstrap(g))
        boot_rows.append(row)
    pooled = {"window": "ALL_WINDOWS_POOLED"}
    pooled.update(cluster_bootstrap(common))
    boot_rows.append(pooled)
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(RESULTS/"v02b_no_macd_cluster_bootstrap.csv", index=False)

    # Signals admitted by V0.2-B that fail the frozen MACD condition.
    incremental = common[common["macd_group"] == "MACD_FAIL"].copy()
    incremental.to_csv(RESULTS/"v02b_no_macd_incremental_candidates.csv", index=False)

    # Clustering / capacity diagnostics.
    day = clustering_by_day(operational)
    day.to_csv(RESULTS/"v02b_no_macd_clustering_by_day.csv", index=False)
    cluster_summary = clustering_summary(day)
    cluster_summary.to_csv(RESULTS/"v02b_no_macd_clustering_summary.csv", index=False)

    # Frozen benchmark integrity: V0.1 and old stochastic-only control.
    check_rows = []
    for window, cfg in WINDOWS.items():
        check_rows.extend(benchmark_check(window, cfg, op_summary))
    checks = pd.DataFrame(check_rows)
    checks.to_csv(RESULTS/"v02b_no_macd_frozen_benchmark_check.csv", index=False)

    report = [
        "# V0.2-B research — remove mandatory MACD vs frozen V0.1\n\n",
        "## Status\n",
        "**Research candidate only. V0.1 remains frozen.**\n\n",
        "## Only rule change\n",
        "`V0.2-B = stock MA50>MA200 + SPY GREEN/YELLOW + Slow Stochastic 14,3,3 cross above 20`.\n\n",
        "The frozen V0.1 MACD requirement is **not** used as an entry gate in V0.2-B. "
        "MACD is still calculated and recorded as a diagnostic variable. No other rule changes.\n\n",
        "Operationally, V0.2-B is the former `STOCH_ONLY_CONTROL` promoted to a research candidate; "
        "the old control files remain the frozen benchmark.\n\n",
        "## Operational V0.1 vs V0.2-B\n\n",
        "|Window|V0.1 N|V0.2B N|Expansion %|Extra N|Δ Stop pp|Δ +3 pp|Δ +5 pp|Δ MFE pp|Δ MAE pp|Δ Ret20 pp|\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for _, r in op_delta.iterrows():
        report.append(
            f"|{r.window}|{int(r.v01_n)}|{int(r.v02b_n)}|{r.signal_expansion_pct:.2f}|{int(r.extra_signals_n)}|"
            f"{r.delta_stop_pp:.2f}|{r.delta_hit3_pp:.2f}|{r.delta_hit5_pp:.2f}|{r.delta_mfe_pp:.2f}|"
            f"{r.delta_mae_pp:.2f}|{r.delta_ret20_pp:.2f}|\n"
        )

    report += [
        "\n## Common-candidate MACD lift\n\n",
        "A negative ΔStop favors MACD_PASS. Positive Δ+3, Δ+5, ΔMFE, ΔMAE (less-negative MAE), and ΔRet20 favor MACD_PASS.\n\n",
        "|Window|Pass N|Fail N|Pass %|Δ Stop pp|Δ +3 pp|Δ +5 pp|Δ MFE pp|Δ MAE pp|Δ Ret20 pp|\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for _, r in lift.iterrows():
        report.append(
            f"|{r.window}|{int(r.pass_n)}|{int(r.fail_n)}|{r.pass_share_pct:.2f}|{r.delta_stop_pp:.2f}|"
            f"{r.delta_hit3_pp:.2f}|{r.delta_hit5_pp:.2f}|{r.delta_mfe_pp:.2f}|{r.delta_mae_pp:.2f}|{r.delta_ret20_pp:.2f}|\n"
        )

    report += [
        "\n## Cluster-bootstrap 95% intervals — common candidates\n\n",
        "|Window|ΔStop low|ΔStop high|Δ+3 low|Δ+3 high|Δ+5 low|Δ+5 high|ΔRet20 low|ΔRet20 high|\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for _, r in boot.iterrows():
        report.append(
            f"|{r.window}|{r.get('delta_stop_low95', np.nan):.2f}|{r.get('delta_stop_high95', np.nan):.2f}|"
            f"{r.get('delta_hit3_low95', np.nan):.2f}|{r.get('delta_hit3_high95', np.nan):.2f}|"
            f"{r.get('delta_hit5_low95', np.nan):.2f}|{r.get('delta_hit5_high95', np.nan):.2f}|"
            f"{r.get('delta_ret20_low95', np.nan):.2f}|{r.get('delta_ret20_high95', np.nan):.2f}|\n"
        )

    report += [
        "\n## Clustering / capacity diagnostic\n\n",
        "Removing MACD can increase signal frequency and correlated assignment exposure. This table is therefore part of the acceptance test, not an afterthought.\n\n",
        "|Window|Version|Signal days|Max same day|Days ≥3|Days ≥5|Days ≥10|Share signals on ≥5-signal days %|\n",
        "|---|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for _, r in cluster_summary.iterrows():
        report.append(
            f"|{r.window}|{r.signal_type}|{int(r.signal_days)}|{int(r.max_same_day_signals)}|{int(r.days_ge_3_signals)}|"
            f"{int(r.days_ge_5_signals)}|{int(r.days_ge_10_signals)}|{r.share_signals_on_ge5_days_pct:.2f}|\n"
        )

    report += [
        "\n## Frozen benchmark integrity\n\n",
        "Both the re-run V0.1 and re-run V0.2-B are checked against the already-saved frozen `V0.1` and `STOCH_ONLY_CONTROL` summaries.\n\n",
        "|Window|Rerun|Frozen row|Frozen found|N exact|Max metric diff|Close match|\n",
        "|---|---|---|---|---|---:|---|\n",
    ]
    for _, r in checks.iterrows():
        if not bool(r.get("frozen_summary_found", False)):
            report.append(f"|{r.window}|{r.rerun_signal_type}|{r.frozen_signal_type}|No|—|—|—|\n")
        else:
            report.append(
                f"|{r.window}|{r.rerun_signal_type}|{r.frozen_signal_type}|Yes|{bool(r.get('exact_n_match', False))}|"
                f"{r.get('max_abs_metric_diff', np.nan):.4f}|{bool(r.get('benchmark_close_match', False))}|\n"
            )

    report += [
        "\n## Pre-specified interpretation guardrails\n",
        "- Do not accept V0.2-B because pooled averages improve alone.\n",
        "- Removing MACD should improve or stabilize performance across several windows, not merely one crisis.\n",
        "- GFC, 2013–14 and the COVID rebound should not be materially damaged without compensating robustness elsewhere.\n",
        "- Signal expansion and clustering count as costs: a higher-frequency candidate that creates larger correlated clusters may be worse at portfolio level even if mean directional metrics improve.\n",
        "- The incremental `MACD_FAIL` candidates must be inspected for both saved winners and newly admitted losers.\n",
        "- Positive Layer-1 performance still does not prove short-put profitability.\n",
        "- No MACD replacement, ROC filter, threshold optimization or additional indicator is allowed inside this experiment.\n",
    ]

    (RESULTS/"V02B_NO_MACD_REPORT.md").write_text("".join(report), encoding="utf-8")
    print("".join(report))


if __name__ == "__main__":
    main()
