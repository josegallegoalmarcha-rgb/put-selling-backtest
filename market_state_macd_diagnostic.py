from __future__ import annotations

import io
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import statsmodels.formula.api as smf

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

SOURCE_EVENTS = RESULTS / "v02b_no_macd_common_candidate_events.csv"

DATA_REPO = "neo-zhao/CMSC320_Final_Tutorial_Huge_Stock_Market_Dataset"
RAW_BASE = f"https://raw.githubusercontent.com/{DATA_REPO}/main"

YF_START = "2005-01-01"
YF_END = "2023-03-01"

# ---------------------------------------------------------------------
# DIAGNOSTIC ONLY — NO STRATEGY RULE IS CHANGED
# ---------------------------------------------------------------------
#
# Sample:
#   the existing 1,278 V0.2-B common candidates already frozen in
#   results/v02b_no_macd_common_candidate_events.csv
#
# MACD label:
#   MACD_PASS / MACD_FAIL already stored in the frozen candidate file.
#
# State variables measured ONLY at the signal close:
#   1) SPY fall/rebound speed: ROC5
#   2) SPY distance to MA50
#   3) SPY distance to MA200
#   4) SPY MA50 10-session slope
#   5) SPY 20-session annualized realized volatility
#   6) universe breadth above MA50
#   7) universe breadth with positive 5-session return
#   8) same-day candidate cluster size
#   9) same-day cluster share of loaded universe
#
# This script DOES NOT optimize thresholds and DOES NOT create V0.2-C.
# Fixed coarse bins are descriptive only.
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

WINDOWS = {
    "GFC_2007_2009": {
        "universe": UNIVERSE_GFC,
        "preferred_source": "legacy",
        "aliases": {},
    },
    "BULL_2013_2014": {
        "universe": UNIVERSE_2013,
        "preferred_source": "legacy",
        "aliases": {"MHFI": "SPGI", "WAG": "WBA"},
    },
    "CHOP_2015_2016": {
        "universe": UNIVERSE_2015,
        "preferred_source": "legacy",
        "aliases": {"MHFI": "SPGI"},
    },
    "BULL_2017": {
        "universe": UNIVERSE_2017,
        "preferred_source": "yfinance",
        "aliases": {"MHFI": "SPGI"},
    },
    "COVID_2020": {
        "universe": UNIVERSE_2020,
        "preferred_source": "yfinance",
        "aliases": {"UTX": "RTX"},
    },
    "BEAR_2022": {
        "universe": UNIVERSE_2022,
        "preferred_source": "yfinance",
        "aliases": {},
    },
}

EXPECTED_COMMON_CANDIDATES = 1278

STATE_VARS = [
    "spy_roc5_pct",
    "spy_dist_ma50_pct",
    "spy_dist_ma200_pct",
    "spy_ma50_slope10_pct",
    "spy_rv20_ann_pct",
    "breadth_above_ma50_pct",
    "breadth_5d_positive_pct",
    "cluster_size",
    "cluster_share_pct",
]

OUTCOMES = {
    "stop": "stop_hit_20d",
    "hit3": "hit_3pct_before_stop",
    "hit5": "hit_5pct_before_stop",
    "ret20": "ret_20d_pct",
}

# Fixed, coarse, economically interpretable bins.
# They are NOT optimized from the results.
BIN_SPECS = {
    "spy_roc5_pct": (
        [-np.inf, -4, -2, 0, 2, np.inf],
        ["<=-4", "(-4,-2]", "(-2,0]", "(0,2]", ">2"],
    ),
    "spy_dist_ma50_pct": (
        [-np.inf, -5, -2, 0, 2, np.inf],
        ["<=-5", "(-5,-2]", "(-2,0]", "(0,2]", ">2"],
    ),
    "spy_dist_ma200_pct": (
        [-np.inf, -5, 0, 5, np.inf],
        ["<=-5", "(-5,0]", "(0,5]", ">5"],
    ),
    "spy_ma50_slope10_pct": (
        [-np.inf, -1, 0, 1, np.inf],
        ["<=-1", "(-1,0]", "(0,1]", ">1"],
    ),
    "spy_rv20_ann_pct": (
        [-np.inf, 10, 20, 30, 40, np.inf],
        ["<=10", "(10,20]", "(20,30]", "(30,40]", ">40"],
    ),
    "breadth_above_ma50_pct": (
        [-np.inf, 25, 50, 75, np.inf],
        ["<=25", "(25,50]", "(50,75]", ">75"],
    ),
    "breadth_5d_positive_pct": (
        [-np.inf, 25, 50, 75, np.inf],
        ["<=25", "(25,50]", "(50,75]", ">75"],
    ),
    "cluster_size": (
        [-np.inf, 1, 4, 9, np.inf],
        ["1", "2-4", "5-9", "10+"],
    ),
    "cluster_share_pct": (
        [-np.inf, 2, 5, 10, np.inf],
        ["<=2", "(2,5]", "(5,10]", ">10"],
    ),
}


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


def add_market_indicators(d: pd.DataFrame):
    x = d.copy()
    c = x["Close"]

    x["MA50"] = c.rolling(50).mean()
    x["MA200"] = c.rolling(200).mean()

    x["ROC5_PCT"] = (c / c.shift(5) - 1) * 100
    x["DIST_MA50_PCT"] = (c / x["MA50"] - 1) * 100
    x["DIST_MA200_PCT"] = (c / x["MA200"] - 1) * 100
    x["MA50_SLOPE10_PCT"] = (x["MA50"] / x["MA50"].shift(10) - 1) * 100

    logret = np.log(c / c.shift(1))
    x["RV20_ANN_PCT"] = logret.rolling(20).std(ddof=1) * np.sqrt(252) * 100

    return x


def add_stock_breadth_fields(d: pd.DataFrame):
    x = d.copy()
    c = x["Close"]
    x["MA50"] = c.rolling(50).mean()
    x["ABOVE_MA50"] = c > x["MA50"]
    x["RET5_POSITIVE"] = (c / c.shift(5) - 1) > 0
    return x[["ABOVE_MA50", "RET5_POSITIVE"]]


def prepare_signal_dates(events: pd.DataFrame):
    dates_by_window = {}
    for window, g in events.groupby("window"):
        dates_by_window[window] = pd.DatetimeIndex(
            pd.to_datetime(g["signal_date"]).drop_duplicates().sort_values()
        )
    return dates_by_window


def build_spy_state(events: pd.DataFrame):
    parts = []
    dates_by_window = prepare_signal_dates(events)

    for window, dates in dates_by_window.items():
        cfg = WINDOWS[window]
        spy, source = load_symbol("SPY", cfg, etf=True)

        if spy is None:
            raise RuntimeError(f"Could not load SPY for {window}")

        x = add_market_indicators(spy)

        use = x.reindex(dates).copy()
        use["window"] = window
        use["signal_date"] = use.index
        use["spy_state_source"] = source
        use["spy_roc5_pct"] = use["ROC5_PCT"]
        use["spy_dist_ma50_pct"] = use["DIST_MA50_PCT"]
        use["spy_dist_ma200_pct"] = use["DIST_MA200_PCT"]
        use["spy_ma50_slope10_pct"] = use["MA50_SLOPE10_PCT"]
        use["spy_rv20_ann_pct"] = use["RV20_ANN_PCT"]
        use["spy_close_reloaded"] = use["Close"]
        use["spy_ma50_reloaded"] = use["MA50"]
        use["spy_ma200_reloaded"] = use["MA200"]

        parts.append(
            use[
                [
                    "window",
                    "signal_date",
                    "spy_state_source",
                    "spy_roc5_pct",
                    "spy_dist_ma50_pct",
                    "spy_dist_ma200_pct",
                    "spy_ma50_slope10_pct",
                    "spy_rv20_ann_pct",
                    "spy_close_reloaded",
                    "spy_ma50_reloaded",
                    "spy_ma200_reloaded",
                ]
            ].reset_index(drop=True)
        )

    return pd.concat(parts, ignore_index=True)


def build_breadth(events: pd.DataFrame):
    dates_by_window = prepare_signal_dates(events)
    rows = []
    coverage_rows = []

    for window, dates in dates_by_window.items():
        cfg = WINDOWS[window]

        above = []
        ret5pos = []
        loaded = 0

        for ticker in cfg["universe"]:
            d, source = load_symbol(ticker, cfg, etf=False)

            if d is None:
                coverage_rows.append({
                    "window": window,
                    "ticker": ticker,
                    "loaded": False,
                    "source": source,
                })
                continue

            loaded += 1
            coverage_rows.append({
                "window": window,
                "ticker": ticker,
                "loaded": True,
                "source": source,
            })

            b = add_stock_breadth_fields(d).reindex(dates)

            s1 = b["ABOVE_MA50"].rename(ticker)
            s2 = b["RET5_POSITIVE"].rename(ticker)

            above.append(s1)
            ret5pos.append(s2)

            time.sleep(0.015)

        if not above:
            raise RuntimeError(f"No breadth data for {window}")

        above_df = pd.concat(above, axis=1)
        ret5_df = pd.concat(ret5pos, axis=1)

        # Denominator uses non-missing observations on each date.
        above_pct = 100 * above_df.sum(axis=1) / above_df.notna().sum(axis=1)
        ret5_pct = 100 * ret5_df.sum(axis=1) / ret5_df.notna().sum(axis=1)

        for dt in dates:
            rows.append({
                "window": window,
                "signal_date": dt,
                "breadth_above_ma50_pct": float(above_pct.loc[dt]),
                "breadth_5d_positive_pct": float(ret5_pct.loc[dt]),
                "breadth_loaded_universe": int(loaded),
            })

    return pd.DataFrame(rows), pd.DataFrame(coverage_rows)


def add_cluster_fields(events: pd.DataFrame, coverage: pd.DataFrame):
    out = events.copy()

    cluster = (
        out.groupby(["window", "signal_date"])
        .agg(
            cluster_size=("ticker", "size"),
            cluster_stocks=("ticker", "nunique"),
        )
        .reset_index()
    )

    loaded_map = (
        coverage[coverage["loaded"]]
        .groupby("window")["ticker"]
        .nunique()
        .to_dict()
    )

    cluster["loaded_universe"] = cluster["window"].map(loaded_map)
    cluster["cluster_share_pct"] = (
        100 * cluster["cluster_size"] / cluster["loaded_universe"]
    )

    return out.merge(cluster, on=["window", "signal_date"], how="left")


def enrich_candidates(events: pd.DataFrame):
    events = events.copy()
    events["signal_date"] = pd.to_datetime(events["signal_date"])

    spy_state = build_spy_state(events)
    breadth, coverage = build_breadth(events)

    out = events.merge(
        spy_state,
        on=["window", "signal_date"],
        how="left",
        validate="many_to_one",
    )

    out = out.merge(
        breadth,
        on=["window", "signal_date"],
        how="left",
        validate="many_to_one",
    )

    out = add_cluster_fields(out, coverage)

    # Drift check against the SPY values already stored in the frozen event file.
    out["spy_close_drift_pct"] = (
        (out["spy_close_reloaded"] / out["spy_close"] - 1) * 100
    )
    out["spy_ma50_drift_pct"] = (
        (out["spy_ma50_reloaded"] / out["spy_ma50"] - 1) * 100
    )
    out["spy_ma200_drift_pct"] = (
        (out["spy_ma200_reloaded"] / out["spy_ma200"] - 1) * 100
    )

    return out, coverage


def summarize_group(g: pd.DataFrame):
    return {
        "n": len(g),
        "stocks": g["ticker"].nunique(),
        "stop_rate_pct": 100 * g["stop_hit_20d"].mean(),
        "hit3_pct": 100 * g["hit_3pct_before_stop"].mean(),
        "hit5_pct": 100 * g["hit_5pct_before_stop"].mean(),
        "avg_mfe_pct": g["mfe_before_stop_pct"].mean(),
        "avg_mae_pct": g["mae_before_stop_pct"].mean(),
        "avg_ret20_pct": g["ret_20d_pct"].mean(),
    }


def macd_lift(g: pd.DataFrame):
    p = g[g["macd_group"] == "MACD_PASS"]
    f = g[g["macd_group"] == "MACD_FAIL"]

    if len(p) == 0 or len(f) == 0:
        return None

    return {
        "pass_n": len(p),
        "fail_n": len(f),
        "delta_stop_pp": 100*p["stop_hit_20d"].mean() - 100*f["stop_hit_20d"].mean(),
        "delta_hit3_pp": 100*p["hit_3pct_before_stop"].mean() - 100*f["hit_3pct_before_stop"].mean(),
        "delta_hit5_pp": 100*p["hit_5pct_before_stop"].mean() - 100*f["hit_5pct_before_stop"].mean(),
        "delta_mfe_pp": p["mfe_before_stop_pct"].mean() - f["mfe_before_stop_pct"].mean(),
        "delta_mae_pp": p["mae_before_stop_pct"].mean() - f["mae_before_stop_pct"].mean(),
        "delta_ret20_pp": p["ret_20d_pct"].mean() - f["ret_20d_pct"].mean(),
    }


def selection_state_table(df: pd.DataFrame):
    rows = []

    for window, wg in list(df.groupby("window")) + [("ALL_WINDOWS_POOLED", df)]:
        for group, g in wg.groupby("macd_group"):
            row = {
                "window": window,
                "macd_group": group,
                "n": len(g),
            }

            for v in STATE_VARS:
                row[f"{v}_mean"] = g[v].mean()
                row[f"{v}_median"] = g[v].median()

            rows.append(row)

    return pd.DataFrame(rows)


def add_fixed_bins(df: pd.DataFrame):
    out = df.copy()

    for v, (edges, labels) in BIN_SPECS.items():
        out[f"{v}_bin"] = pd.cut(
            out[v],
            bins=edges,
            labels=labels,
            include_lowest=True,
            right=True,
        )

    return out


def interaction_bins(df: pd.DataFrame):
    rows = []

    for variable in BIN_SPECS:
        bcol = f"{variable}_bin"

        for bucket, g in df.groupby(bcol, observed=True):
            lift = macd_lift(g)
            if lift is None:
                continue

            row = {
                "window": "ALL_WINDOWS_POOLED",
                "variable": variable,
                "bucket": str(bucket),
                "bucket_n": len(g),
            }
            row.update(lift)
            rows.append(row)

        for window, wg in df.groupby("window"):
            for bucket, g in wg.groupby(bcol, observed=True):
                lift = macd_lift(g)
                if lift is None:
                    continue

                row = {
                    "window": window,
                    "variable": variable,
                    "bucket": str(bucket),
                    "bucket_n": len(g),
                }
                row.update(lift)
                rows.append(row)

    return pd.DataFrame(rows)


def consistency_table(interactions: pd.DataFrame):
    rows = []

    # For each variable/bucket, ask whether MACD effect has the same sign
    # across windows with minimum 5 PASS and 5 FAIL candidates.
    x = interactions[interactions["window"] != "ALL_WINDOWS_POOLED"].copy()
    x = x[(x["pass_n"] >= 5) & (x["fail_n"] >= 5)]

    for (variable, bucket), g in x.groupby(["variable", "bucket"]):
        rows.append({
            "variable": variable,
            "bucket": bucket,
            "windows_eligible": len(g),
            "windows_macd_lower_stop": int((g["delta_stop_pp"] < 0).sum()),
            "windows_macd_higher_hit3": int((g["delta_hit3_pp"] > 0).sum()),
            "windows_macd_higher_hit5": int((g["delta_hit5_pp"] > 0).sum()),
            "windows_macd_higher_ret20": int((g["delta_ret20_pp"] > 0).sum()),
            "median_delta_stop_pp": g["delta_stop_pp"].median(),
            "median_delta_hit3_pp": g["delta_hit3_pp"].median(),
            "median_delta_hit5_pp": g["delta_hit5_pp"].median(),
            "median_delta_ret20_pp": g["delta_ret20_pp"].median(),
        })

    return pd.DataFrame(rows)


def interaction_regressions(df: pd.DataFrame):
    rows = []
    work = df.copy()

    work["macd_pass_int"] = (work["macd_group"] == "MACD_PASS").astype(int)
    work["stop_int"] = work["stop_hit_20d"].astype(int)
    work["hit3_int"] = work["hit_3pct_before_stop"].astype(int)
    work["hit5_int"] = work["hit_5pct_before_stop"].astype(int)
    work["cluster_key"] = (
        work["window"].astype(str)
        + "__"
        + pd.to_datetime(work["signal_date"]).dt.strftime("%Y-%m-%d")
    )

    # One-variable-at-a-time interaction models.
    # State variables are standardized only for comparability of coefficients.
    for variable in STATE_VARS:
        vals = pd.to_numeric(work[variable], errors="coerce")
        mean = vals.mean()
        sd = vals.std(ddof=0)

        if not np.isfinite(sd) or sd == 0:
            continue

        zcol = f"z_{variable}"
        work[zcol] = (vals - mean) / sd

        for outcome_name, outcome_col in {
            "stop": "stop_int",
            "hit3": "hit3_int",
            "hit5": "hit5_int",
            "ret20": "ret_20d_pct",
        }.items():
            formula = (
                f"{outcome_col} ~ macd_pass_int * {zcol} + C(window)"
            )

            use = work[
                [
                    outcome_col,
                    "macd_pass_int",
                    zcol,
                    "window",
                    "cluster_key",
                ]
            ].dropna()

            try:
                model = smf.ols(formula, data=use).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": use["cluster_key"]},
                )

                term = f"macd_pass_int:{zcol}"

                rows.append({
                    "variable": variable,
                    "outcome": outcome_name,
                    "n": int(model.nobs),
                    "state_mean": mean,
                    "state_sd": sd,
                    "interaction_coef_per_1sd": model.params.get(term, np.nan),
                    "interaction_se": model.bse.get(term, np.nan),
                    "interaction_pvalue": model.pvalues.get(term, np.nan),
                    "macd_main_coef_at_state_mean": model.params.get("macd_pass_int", np.nan),
                })
            except Exception as exc:
                rows.append({
                    "variable": variable,
                    "outcome": outcome_name,
                    "n": len(use),
                    "state_mean": mean,
                    "state_sd": sd,
                    "interaction_coef_per_1sd": np.nan,
                    "interaction_se": np.nan,
                    "interaction_pvalue": np.nan,
                    "macd_main_coef_at_state_mean": np.nan,
                    "error": str(exc),
                })

    return pd.DataFrame(rows)


def extreme_clusters(df: pd.DataFrame):
    day = (
        df.groupby(["window", "signal_date"])
        .agg(
            signals=("ticker", "size"),
            stocks=("ticker", "nunique"),
            macd_pass_share_pct=("macd_pass", lambda s: 100*np.mean(s.astype(bool))),
            stop_rate_pct=("stop_hit_20d", lambda s: 100*np.mean(s.astype(bool))),
            avg_ret20_pct=("ret_20d_pct", "mean"),
            spy_roc5_pct=("spy_roc5_pct", "first"),
            spy_dist_ma50_pct=("spy_dist_ma50_pct", "first"),
            spy_dist_ma200_pct=("spy_dist_ma200_pct", "first"),
            spy_ma50_slope10_pct=("spy_ma50_slope10_pct", "first"),
            spy_rv20_ann_pct=("spy_rv20_ann_pct", "first"),
            breadth_above_ma50_pct=("breadth_above_ma50_pct", "first"),
            breadth_5d_positive_pct=("breadth_5d_positive_pct", "first"),
        )
        .reset_index()
    )

    return day.sort_values(
        ["signals", "signal_date"],
        ascending=[False, True],
    )


def drift_summary(df: pd.DataFrame):
    rows = []

    for window, g in df.groupby("window"):
        rows.append({
            "window": window,
            "n": len(g),
            "max_abs_spy_close_drift_pct": g["spy_close_drift_pct"].abs().max(),
            "max_abs_spy_ma50_drift_pct": g["spy_ma50_drift_pct"].abs().max(),
            "max_abs_spy_ma200_drift_pct": g["spy_ma200_drift_pct"].abs().max(),
            "median_abs_spy_close_drift_pct": g["spy_close_drift_pct"].abs().median(),
        })

    return pd.DataFrame(rows)


def write_report(
    df,
    selection,
    interactions,
    consistency,
    regressions,
    cluster_days,
    drift,
):
    report = [
        "# Market-state diagnostic — MACD_PASS vs MACD_FAIL\n\n",
        "## Status\n",
        "**Diagnostic only. No V0.1/V0.2 rule is changed and no V0.2-C is created.**\n\n",
        f"Sample: **{len(df):,} common candidates** from the frozen V0.2-B "
        "common-candidate stream.\n\n",
        "## Pre-specified market-state variables\n",
        "- SPY ROC5: five-session speed of decline/rebound.\n",
        "- SPY distance to MA50.\n",
        "- SPY distance to MA200.\n",
        "- SPY MA50 10-session slope.\n",
        "- SPY 20-session annualized realized volatility.\n",
        "- Breadth: percentage of the point-in-time universe above MA50.\n",
        "- Breadth: percentage of the point-in-time universe with positive 5-session return.\n",
        "- Same-day candidate cluster size and cluster share of the loaded universe.\n\n",
        "All state variables are measured at the **signal close**. No future information is used.\n\n",
        "## What this diagnostic is allowed to answer\n",
        "We are looking for a state variable whose interaction with MACD is "
        "directionally stable across multiple windows. Fixed coarse buckets and "
        "one-variable-at-a-time interaction regressions are descriptive tools only; "
        "they do not define a trading threshold.\n\n",
        "## MACD selection by market state — pooled means\n\n",
        "|Group|N|ROC5 %|Dist MA50 %|Dist MA200 %|MA50 slope10 %|RV20 %|"
        "Breadth >MA50 %|Breadth 5d+ %|Cluster size|\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]

    pooled_sel = selection[selection["window"] == "ALL_WINDOWS_POOLED"]

    for _, r in pooled_sel.iterrows():
        report.append(
            f"|{r.macd_group}|{int(r.n)}|"
            f"{r.spy_roc5_pct_mean:.2f}|"
            f"{r.spy_dist_ma50_pct_mean:.2f}|"
            f"{r.spy_dist_ma200_pct_mean:.2f}|"
            f"{r.spy_ma50_slope10_pct_mean:.2f}|"
            f"{r.spy_rv20_ann_pct_mean:.2f}|"
            f"{r.breadth_above_ma50_pct_mean:.2f}|"
            f"{r.breadth_5d_positive_pct_mean:.2f}|"
            f"{r.cluster_size_mean:.2f}|\n"
        )

    report += [
        "\n## Continuous interaction regressions\n\n",
        "Each row estimates whether a **1 standard-deviation change in the state "
        "variable changes the observed MACD_PASS minus MACD_FAIL relationship**, "
        "controlling for window fixed effects and clustering standard errors by "
        "signal date. These are descriptive associations, not causal estimates.\n\n",
        "|Variable|Outcome|Interaction / 1 SD|SE|p-value|N|\n",
        "|---|---|---:|---:|---:|---:|\n",
    ]

    reg_sorted = regressions.sort_values(
        ["outcome", "interaction_pvalue"],
        na_position="last",
    )

    for _, r in reg_sorted.iterrows():
        report.append(
            f"|{r.variable}|{r.outcome}|"
            f"{r.interaction_coef_per_1sd:.4f}|"
            f"{r.interaction_se:.4f}|"
            f"{r.interaction_pvalue:.4f}|{int(r.n)}|\n"
        )

    report += [
        "\n## Cross-window consistency guardrail\n\n",
        "The fixed-bin file records, for every state bucket, how many eligible "
        "windows show MACD with fewer stops, more +3, more +5 and higher Ret20. "
        "A future V0.2-C should only be considered if a relationship is repeated "
        "across several windows with adequate PASS and FAIL sample sizes.\n\n",
        "The full tables are written to CSV because the interaction matrix is too "
        "large to render safely in this report.\n\n",
        "## Largest signal clusters\n\n",
        "|Window|Date|Signals|MACD pass %|Stop %|Ret20 %|ROC5 %|RV20 %|Breadth >MA50 %|Breadth 5d+ %|\n",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]

    for _, r in cluster_days.head(20).iterrows():
        report.append(
            f"|{r.window}|{pd.Timestamp(r.signal_date).date()}|{int(r.signals)}|"
            f"{r.macd_pass_share_pct:.1f}|{r.stop_rate_pct:.1f}|"
            f"{r.avg_ret20_pct:.2f}|{r.spy_roc5_pct:.2f}|"
            f"{r.spy_rv20_ann_pct:.2f}|{r.breadth_above_ma50_pct:.1f}|"
            f"{r.breadth_5d_positive_pct:.1f}|\n"
        )

    report += [
        "\n## SPY data-state drift check\n\n",
        "|Window|N|Max abs Close drift %|Max abs MA50 drift %|Max abs MA200 drift %|\n",
        "|---|---:|---:|---:|---:|\n",
    ]

    for _, r in drift.iterrows():
        report.append(
            f"|{r.window}|{int(r.n)}|"
            f"{r.max_abs_spy_close_drift_pct:.5f}|"
            f"{r.max_abs_spy_ma50_drift_pct:.5f}|"
            f"{r.max_abs_spy_ma200_drift_pct:.5f}|\n"
        )

    report += [
        "\n## Guardrails\n",
        "- Do not create a rule from the best-looking single bucket.\n",
        "- Do not scan additional thresholds after seeing the output.\n",
        "- A candidate explanatory variable should show the same directional "
        "interaction in several distinct windows and retain adequate sample size.\n",
        "- Breadth and clustering are treated as separate concepts: breadth measures "
        "the cross-section of the point-in-time universe; clustering measures the "
        "density of actual candidate signals.\n",
        "- This remains Layer 1 underlying-price research, not short-put P/L.\n",
    ]

    (RESULTS / "MARKET_STATE_MACD_DIAGNOSTIC.md").write_text(
        "".join(report),
        encoding="utf-8",
    )


def main():
    if not SOURCE_EVENTS.exists():
        raise FileNotFoundError(
            f"Required frozen candidate file not found: {SOURCE_EVENTS}"
        )

    events = pd.read_csv(SOURCE_EVENTS)

    required = {
        "window",
        "ticker",
        "signal_date",
        "macd_group",
        "macd_pass",
        "stop_hit_20d",
        "hit_3pct_before_stop",
        "hit_5pct_before_stop",
        "mfe_before_stop_pct",
        "mae_before_stop_pct",
        "ret_20d_pct",
        "spy_close",
        "spy_ma50",
        "spy_ma200",
    }

    missing = required - set(events.columns)
    if missing:
        raise RuntimeError(f"Source events missing columns: {sorted(missing)}")

    if len(events) != EXPECTED_COMMON_CANDIDATES:
        raise RuntimeError(
            f"Expected {EXPECTED_COMMON_CANDIDATES} frozen common candidates, "
            f"found {len(events)}. Stop and inspect before running diagnostic."
        )

    expected_groups = {"MACD_PASS", "MACD_FAIL"}
    if set(events["macd_group"].dropna().unique()) != expected_groups:
        raise RuntimeError("Unexpected MACD group labels in frozen source.")

    events["signal_date"] = pd.to_datetime(events["signal_date"])

    enriched, coverage = enrich_candidates(events)

    # Ensure state fields are substantially populated.
    for v in STATE_VARS:
        missing_rate = enriched[v].isna().mean()
        if missing_rate > 0.05:
            raise RuntimeError(
                f"{v} missing for {missing_rate:.1%} of candidates."
            )

    enriched = add_fixed_bins(enriched)

    selection = selection_state_table(enriched)
    interactions = interaction_bins(enriched)
    consistency = consistency_table(interactions)
    regressions = interaction_regressions(enriched)
    cluster_days = extreme_clusters(enriched)
    drift = drift_summary(enriched)

    enriched.to_csv(
        RESULTS / "market_state_enriched_common_candidates.csv",
        index=False,
    )
    coverage.to_csv(
        RESULTS / "market_state_breadth_coverage.csv",
        index=False,
    )
    selection.to_csv(
        RESULTS / "market_state_macd_selection.csv",
        index=False,
    )
    interactions.to_csv(
        RESULTS / "market_state_macd_interaction_bins.csv",
        index=False,
    )
    consistency.to_csv(
        RESULTS / "market_state_macd_consistency.csv",
        index=False,
    )
    regressions.to_csv(
        RESULTS / "market_state_macd_interaction_regressions.csv",
        index=False,
    )
    cluster_days.to_csv(
        RESULTS / "market_state_cluster_days.csv",
        index=False,
    )
    drift.to_csv(
        RESULTS / "market_state_spy_drift_check.csv",
        index=False,
    )

    write_report(
        enriched,
        selection,
        interactions,
        consistency,
        regressions,
        cluster_days,
        drift,
    )

    print(
        f"Market-state diagnostic completed on {len(enriched)} frozen candidates."
    )


if __name__ == "__main__":
    main()
