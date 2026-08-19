from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.formula.api as smf

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

# =====================================================================
# INDEPENDENT VALIDATION SAMPLE — H1
# =====================================================================
#
# DISCOVERY-SAMPLE H1, FROZEN BEFORE THIS SCRIPT IS RUN:
#
#   "The incremental value of MACD should be greater when very few
#    Dividend Aristocrats have a positive 5-session return, and should
#    diminish as the short-term recovery becomes broad."
#
# Primary state variable:
#   breadth_5d_positive_pct
#
# Frozen descriptive buckets inherited from the discovery diagnostic:
#   <=25%, (25,50], (50,75], >75%
#
# This script DOES NOT:
# - alter V0.1 or V0.2-B;
# - optimize thresholds;
# - scan ROC windows;
# - add a new filter;
# - create V0.2-C.
#
# It independently rebuilds the common-candidate stream in four windows
# that were not used to discover H1:
#   2018
#   2019
#   2021
#   2023-2024
#
# Candidate definition is the V0.2-B/base stream:
#   stock MA50 > MA200
#   SPY regime GREEN or YELLOW
#   Slow Stochastic 14,3,3 crosses upward through 20
#   common 20-session per-stock cooldown
#
# Only AFTER common candidate selection is each event labelled
# MACD_PASS / MACD_FAIL using the frozen V0.1 MACD rule.
#
# All state variables are measured at the signal close.
# =====================================================================

COOLDOWN = 20
HORIZONS = (5, 10, 15, 20)

# 2019 source has 57 names and states that four were new:
# CB, CAT, PBCT and UTX. Therefore the 2018 set is the same list
# without those four names (53 names).
UNIVERSE_2019 = """
MMM ABT ABBV AFL APD ADM T ADP BDX BF-B CAH CAT CVX CB CINF CTAS CLX KO CL ED
DOV ECL EMR XOM FRT BEN GD GPC GWW HRL ITW JNJ KMB LEG LIN LOW MKC MCD MDT NUE
PNR PBCT PEP PPG PG ROP SPGI SHW AOS SWK SYY TROW TGT UTX VFC WBA WMT
""".split()

UNIVERSE_2018 = [
    t for t in UNIVERSE_2019 if t not in {"CB", "CAT", "PBCT", "UTX"}
]

# 2021 list: 65 names. This is the Jan-2022 start snapshot previously
# used by the project plus LEG, which remained an Aristocrat through 2021.
UNIVERSE_2021 = """
ABBV ABT ADM ADP AFL ALB AMCR AOS APD ATO BDX BEN BF-B CAH CAT CB CINF CL CLX
CTAS CVX DOV ECL ED EMR ESS EXPD FRT GD GPC GWW HRL IBM ITW JNJ KMB KO LEG LIN
LOW MCD MDT MKC MMM NEE NUE O PBCT PEP PG PNR PPG ROP SHW SPGI SWK SYY T TGT
TROW VFC WBA WMT WST XOM
""".split()

# 2023 list: 67 names.
# Relative to the Jan-2022 project snapshot:
# - T and PBCT are absent,
# - BRO and CHD are present,
# - CHRW, SJM and NDSN were added for 2023.
# The universe is frozen at the start of this validation window through
# the end of 2024, intentionally retaining later removals and excluding
# later additions to avoid hindsight/survivorship filtering.
UNIVERSE_2023 = """
ABBV ABT ADM ADP AFL ALB AMCR AOS APD ATO BDX BEN BF-B BRO CAH CAT CB CHD CHRW
CINF CL CLX CTAS CVX DOV ECL ED EMR ESS EXPD FRT GD GPC GWW HRL IBM ITW JNJ KMB
KO LIN LOW MCD MDT MKC MMM NDSN NEE NUE O PEP PG PNR PPG ROP SHW SJM SPGI SWK
SYY TGT TROW VFC WBA WMT WST XOM
""".split()

assert len(UNIVERSE_2018) == 53
assert len(UNIVERSE_2019) == 57
assert len(UNIVERSE_2021) == 65
assert len(UNIVERSE_2023) == 67

WINDOWS = {
    "VALIDATION_2018": {
        "universe": UNIVERSE_2018,
        "start": pd.Timestamp("2018-02-01"),
        "end": pd.Timestamp("2018-12-31"),
        "aliases": {},
    },
    "VALIDATION_2019": {
        "universe": UNIVERSE_2019,
        "start": pd.Timestamp("2019-02-01"),
        "end": pd.Timestamp("2019-12-31"),
        "aliases": {"UTX": "RTX"},
    },
    "VALIDATION_2021": {
        "universe": UNIVERSE_2021,
        "start": pd.Timestamp("2021-02-01"),
        "end": pd.Timestamp("2021-12-31"),
        "aliases": {},
    },
    "VALIDATION_2023_2024": {
        "universe": UNIVERSE_2023,
        "start": pd.Timestamp("2023-02-01"),
        "end": pd.Timestamp("2024-12-31"),
        "aliases": {},
    },
}

YF_START = "2015-01-01"
YF_END = "2025-03-15"

BREADTH_EDGES = [-np.inf, 25, 50, 75, np.inf]
BREADTH_LABELS = ["<=25", "(25,50]", "(50,75]", ">75"]

MIN_PASS_FAIL_PER_WINDOW_BUCKET = 5
BOOT_REPS = 2000
SEED = 20260819


# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------

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


def load_symbol(ticker: str, aliases: dict):
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


def add_indicators(d: pd.DataFrame):
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

    x["RET5_POSITIVE"] = (c / c.shift(5) - 1) > 0
    return x


def load_spy():
    d = load_symbol("SPY", {})
    if d is None:
        raise RuntimeError("SPY could not be loaded from yfinance")

    s = add_indicators(d)
    s["REGIME"] = np.select(
        [
            s["MA50"] <= s["MA200"],
            (s["MA50"] > s["MA200"]) & (s["Close"] <= s["MA200"]),
            (s["MA50"] > s["MA200"]) & (s["Close"] > s["MA200"]),
        ],
        ["RED", "YELLOW", "GREEN"],
        default="UNKNOWN",
    )

    s["SPY_ROC5_PCT"] = (s["Close"] / s["Close"].shift(5) - 1) * 100
    return s


def attach_spy(x: pd.DataFrame, spy: pd.DataFrame):
    r = spy[
        ["Close", "MA50", "MA200", "REGIME", "SPY_ROC5_PCT"]
    ].rename(
        columns={
            "Close": "SPY_CLOSE",
            "MA50": "SPY_MA50",
            "MA200": "SPY_MA200",
        }
    )

    return x.join(r, how="left").ffill()


# ---------------------------------------------------------------------
# FROZEN COMMON-CANDIDATE LOGIC
# ---------------------------------------------------------------------

def candidate_mask_and_macd(x: pd.DataFrame):
    trend = x["MA50"] > x["MA200"]
    market = x["REGIME"].isin(["GREEN", "YELLOW"])
    stoch_cross = (x["STOCH_K"].shift(1) <= 20) & (x["STOCH_K"] > 20)

    h = x["MACD_HIST"]
    macd_pass = (
        (h < 0)
        & (h > h.shift(1))
        & (h.shift(1) > h.shift(2))
    )

    base = (trend & market & stoch_cross).fillna(False)
    return base, macd_pass.fillna(False)


def spaced(mask: pd.Series):
    raw = np.flatnonzero(mask.to_numpy())
    out = []
    last = -10**9

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


def evaluate(x, i, ticker, window, macd_group):
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
        "signal_date": x.index[i],
        "entry_date": x.index[entry_i],
        "entry_open": entry,
        "regime": x["REGIME"].iloc[i],
        "spy_close": x["SPY_CLOSE"].iloc[i],
        "spy_ma50": x["SPY_MA50"].iloc[i],
        "spy_ma200": x["SPY_MA200"].iloc[i],
        "spy_roc5_pct": x["SPY_ROC5_PCT"].iloc[i],
        "macd_group": macd_group,
        "macd_pass": macd_group == "MACD_PASS",
        "macd_hist": x["MACD_HIST"].iloc[i],
        "swing_start": swing_start,
        "swing_low": sl,
        "swing_distance_pct": (entry/sl - 1)*100 if sl > 0 else np.nan,
        "stop_hit_20d": stop_rel is not None,
        "days_to_stop": stop_rel if stop_rel is not None else np.nan,
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
# WINDOW BUILD
# ---------------------------------------------------------------------

def run_window(window: str, cfg: dict, spy: pd.DataFrame):
    events = []
    coverage = []
    breadth_series = []

    for ticker in cfg["universe"]:
        d = load_symbol(ticker, cfg["aliases"])

        if d is None:
            coverage.append({
                "window": window,
                "ticker": ticker,
                "loaded": False,
                "data_ticker": cfg["aliases"].get(ticker, ticker),
            })
            continue

        coverage.append({
            "window": window,
            "ticker": ticker,
            "loaded": True,
            "data_ticker": cfg["aliases"].get(ticker, ticker),
            "first": d.index.min(),
            "last": d.index.max(),
            "rows": len(d),
        })

        x0 = add_indicators(d)
        breadth_series.append(
            x0["RET5_POSITIVE"].rename(ticker)
        )

        x = attach_spy(x0, spy)
        base, macd_pass = candidate_mask_and_macd(x)

        for i in spaced(base):
            if cfg["start"] <= x.index[i] <= cfg["end"]:
                group = "MACD_PASS" if bool(macd_pass.iloc[i]) else "MACD_FAIL"
                e = evaluate(x, i, ticker, window, group)
                if e is not None:
                    events.append(e)

        time.sleep(0.02)

    ev = pd.DataFrame(events)
    cov = pd.DataFrame(coverage)

    if ev.empty:
        raise RuntimeError(f"No validation events generated for {window}")

    breadth_df = pd.concat(breadth_series, axis=1).sort_index()

    # Breadth denominator uses the loaded point-in-time universe and
    # excludes missing observations on each date.
    breadth_pct = (
        100
        * breadth_df.sum(axis=1)
        / breadth_df.notna().sum(axis=1)
    )

    ev["signal_date"] = pd.to_datetime(ev["signal_date"])
    ev["breadth_5d_positive_pct"] = ev["signal_date"].map(breadth_pct)

    cluster = (
        ev.groupby("signal_date")
        .agg(
            cluster_size=("ticker", "size"),
            cluster_stocks=("ticker", "nunique"),
        )
    )

    ev["cluster_size"] = ev["signal_date"].map(cluster["cluster_size"])

    loaded_n = int(cov["loaded"].sum())
    ev["loaded_universe"] = loaded_n
    ev["cluster_share_pct"] = (
        100 * ev["cluster_size"] / loaded_n
    )

    ev["breadth_bucket"] = pd.cut(
        ev["breadth_5d_positive_pct"],
        bins=BREADTH_EDGES,
        labels=BREADTH_LABELS,
        include_lowest=True,
        right=True,
    )

    return ev, cov


# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------

def lift(g: pd.DataFrame):
    p = g[g["macd_group"] == "MACD_PASS"]
    f = g[g["macd_group"] == "MACD_FAIL"]

    if len(p) == 0 or len(f) == 0:
        return None

    return {
        "n": len(g),
        "pass_n": len(p),
        "fail_n": len(f),
        "delta_stop_pp": (
            100*p["stop_hit_20d"].mean()
            - 100*f["stop_hit_20d"].mean()
        ),
        "delta_hit3_pp": (
            100*p["hit_3pct_before_stop"].mean()
            - 100*f["hit_3pct_before_stop"].mean()
        ),
        "delta_hit5_pp": (
            100*p["hit_5pct_before_stop"].mean()
            - 100*f["hit_5pct_before_stop"].mean()
        ),
        "delta_mfe_pp": (
            p["mfe_before_stop_pct"].mean()
            - f["mfe_before_stop_pct"].mean()
        ),
        "delta_mae_pp": (
            p["mae_before_stop_pct"].mean()
            - f["mae_before_stop_pct"].mean()
        ),
        "delta_ret20_pp": (
            p["ret_20d_pct"].mean()
            - f["ret_20d_pct"].mean()
        ),
    }


def bucket_lifts(df: pd.DataFrame):
    rows = []

    for bucket, g in df.groupby("breadth_bucket", observed=True):
        r = {
            "window": "ALL_VALIDATION_POOLED",
            "breadth_bucket": str(bucket),
        }
        v = lift(g)
        if v:
            r.update(v)
            rows.append(r)

    for window, wg in df.groupby("window"):
        for bucket, g in wg.groupby("breadth_bucket", observed=True):
            r = {
                "window": window,
                "breadth_bucket": str(bucket),
            }
            v = lift(g)
            if v:
                r.update(v)
                rows.append(r)

    return pd.DataFrame(rows)


def overall_lifts(df: pd.DataFrame):
    rows = []

    v = lift(df)
    if v:
        r = {"window": "ALL_VALIDATION_POOLED"}
        r.update(v)
        rows.append(r)

    for window, g in df.groupby("window"):
        v = lift(g)
        if v:
            r = {"window": window}
            r.update(v)
            rows.append(r)

    return pd.DataFrame(rows)


def consistency(df_lifts: pd.DataFrame):
    x = df_lifts[
        df_lifts["window"] != "ALL_VALIDATION_POOLED"
    ].copy()

    x = x[
        (x["pass_n"] >= MIN_PASS_FAIL_PER_WINDOW_BUCKET)
        & (x["fail_n"] >= MIN_PASS_FAIL_PER_WINDOW_BUCKET)
    ]

    rows = []

    for bucket, g in x.groupby("breadth_bucket"):
        rows.append({
            "breadth_bucket": bucket,
            "eligible_windows": len(g),
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


# ---------------------------------------------------------------------
# PRE-SPECIFIED CONTINUOUS INTERACTION
# ---------------------------------------------------------------------

def interaction_regression(df: pd.DataFrame):
    x = df.copy()

    x["macd_pass_int"] = (
        x["macd_group"] == "MACD_PASS"
    ).astype(int)

    breadth = pd.to_numeric(
        x["breadth_5d_positive_pct"], errors="coerce"
    )
    mean = breadth.mean()
    sd = breadth.std(ddof=0)
    x["breadth_z"] = (breadth - mean) / sd

    x["cluster_key"] = (
        x["window"].astype(str)
        + "__"
        + pd.to_datetime(x["signal_date"]).dt.strftime("%Y-%m-%d")
    )

    rows = []

    outcomes = {
        "ret20": "ret_20d_pct",
        "stop": "stop_hit_20d",
        "hit3": "hit_3pct_before_stop",
        "hit5": "hit_5pct_before_stop",
    }

    for name, col in outcomes.items():
        work = x[
            [col, "macd_pass_int", "breadth_z", "window", "cluster_key"]
        ].dropna().copy()

        if name != "ret20":
            work[col] = work[col].astype(int)

        formula = (
            f"{col} ~ macd_pass_int * breadth_z + C(window)"
        )

        model = smf.ols(formula, data=work).fit(
            cov_type="cluster",
            cov_kwds={"groups": work["cluster_key"]},
        )

        term = "macd_pass_int:breadth_z"

        rows.append({
            "outcome": name,
            "n": int(model.nobs),
            "breadth_mean": mean,
            "breadth_sd": sd,
            "interaction_coef_per_1sd": model.params.get(term, np.nan),
            "interaction_se": model.bse.get(term, np.nan),
            "interaction_pvalue": model.pvalues.get(term, np.nan),
            "macd_main_at_mean_breadth": model.params.get(
                "macd_pass_int", np.nan
            ),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# CLUSTER BOOTSTRAP FOR LOW/HIGH BREADTH BUCKETS
# ---------------------------------------------------------------------

def bootstrap_lift(g: pd.DataFrame, reps=BOOT_REPS, seed=SEED):
    x = g.copy()

    if (
        (x["macd_group"] == "MACD_PASS").sum() == 0
        or (x["macd_group"] == "MACD_FAIL").sum() == 0
    ):
        return {}

    x["cluster_key"] = (
        x["window"].astype(str)
        + "__"
        + pd.to_datetime(x["signal_date"]).dt.strftime("%Y-%m-%d")
    )

    grouped = {
        key: part.copy()
        for key, part in x.groupby("cluster_key")
    }
    clusters = np.array(list(grouped.keys()))

    rng = np.random.default_rng(seed)
    ret20 = []
    hit5 = []
    stop = []

    for _ in range(reps):
        sampled = rng.choice(
            clusters,
            size=len(clusters),
            replace=True,
        )

        b = pd.concat(
            [grouped[k] for k in sampled],
            ignore_index=True,
        )

        v = lift(b)
        if v is not None:
            ret20.append(v["delta_ret20_pp"])
            hit5.append(v["delta_hit5_pp"])
            stop.append(v["delta_stop_pp"])

    def ci(values):
        if not values:
            return (np.nan, np.nan)
        a = np.asarray(values)
        return (
            np.quantile(a, 0.025),
            np.quantile(a, 0.975),
        )

    rlo, rhi = ci(ret20)
    hlo, hhi = ci(hit5)
    slo, shi = ci(stop)

    return {
        "delta_ret20_low95": rlo,
        "delta_ret20_high95": rhi,
        "delta_hit5_low95": hlo,
        "delta_hit5_high95": hhi,
        "delta_stop_low95": slo,
        "delta_stop_high95": shi,
    }


def bootstrap_table(df: pd.DataFrame):
    rows = []

    for bucket in ["<=25", ">75"]:
        g = df[df["breadth_bucket"].astype(str) == bucket]
        r = {
            "breadth_bucket": bucket,
            "n": len(g),
        }
        r.update(
            bootstrap_lift(
                g,
                seed=SEED + (1 if bucket == "<=25" else 2),
            )
        )
        rows.append(r)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# FROZEN VALIDATION VERDICT
# ---------------------------------------------------------------------

def make_verdict(
    bucket_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    regression_df: pd.DataFrame,
):
    pooled = bucket_df[
        bucket_df["window"] == "ALL_VALIDATION_POOLED"
    ].set_index("breadth_bucket")

    low_ok = False
    high_relation_ok = False
    interaction_ok = False
    cross_window_ok = False

    if "<=25" in pooled.index:
        low_ok = pooled.loc["<=25", "delta_ret20_pp"] > 0

    if "<=25" in pooled.index and ">75" in pooled.index:
        high_relation_ok = (
            pooled.loc["<=25", "delta_ret20_pp"]
            > pooled.loc[">75", "delta_ret20_pp"]
        )

    ret = regression_df[
        regression_df["outcome"] == "ret20"
    ]
    if not ret.empty:
        interaction_ok = (
            ret.iloc[0]["interaction_coef_per_1sd"] < 0
        )

    low_cons = consistency_df[
        consistency_df["breadth_bucket"] == "<=25"
    ]

    if not low_cons.empty:
        eligible = int(low_cons.iloc[0]["eligible_windows"])
        positive = int(
            low_cons.iloc[0]["windows_macd_higher_ret20"]
        )

        # Pre-specified: at least 3 eligible validation windows and
        # at least 75% of those windows must show positive Ret20 lift.
        cross_window_ok = (
            eligible >= 3
            and positive / eligible >= 0.75
        )

    criteria = {
        "low_breadth_pooled_ret20_positive": low_ok,
        "low_breadth_ret20_greater_than_high_breadth": high_relation_ok,
        "continuous_interaction_ret20_negative": interaction_ok,
        "low_breadth_cross_window_consistency": cross_window_ok,
    }

    score = sum(bool(v) for v in criteria.values())

    if score == 4:
        verdict = "VALIDATED_DIRECTIONALLY"
    elif score >= 2:
        verdict = "PARTIAL_NOT_ENOUGH_FOR_V02C"
    else:
        verdict = "FAILED_VALIDATION"

    return verdict, criteria


# ---------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------

def main():
    spy = load_spy()

    all_events = []
    all_coverage = []

    for window, cfg in WINDOWS.items():
        ev, cov = run_window(window, cfg, spy)
        all_events.append(ev)
        all_coverage.append(cov)

    events = pd.concat(all_events, ignore_index=True)
    coverage = pd.concat(all_coverage, ignore_index=True)

    events = events.sort_values(
        ["window", "signal_date", "ticker"]
    )

    bucket_df = bucket_lifts(events)
    overall_df = overall_lifts(events)
    consistency_df = consistency(bucket_df)
    regression_df = interaction_regression(events)
    bootstrap_df = bootstrap_table(events)

    verdict, criteria = make_verdict(
        bucket_df,
        consistency_df,
        regression_df,
    )

    events.to_csv(
        RESULTS / "validation_h1_common_candidates.csv",
        index=False,
    )
    coverage.to_csv(
        RESULTS / "validation_h1_coverage.csv",
        index=False,
    )
    overall_df.to_csv(
        RESULTS / "validation_h1_macd_overall_lift.csv",
        index=False,
    )
    bucket_df.to_csv(
        RESULTS / "validation_h1_breadth_bucket_lift.csv",
        index=False,
    )
    consistency_df.to_csv(
        RESULTS / "validation_h1_cross_window_consistency.csv",
        index=False,
    )
    regression_df.to_csv(
        RESULTS / "validation_h1_interaction_regression.csv",
        index=False,
    )
    bootstrap_df.to_csv(
        RESULTS / "validation_h1_cluster_bootstrap.csv",
        index=False,
    )

    verdict_rows = [
        {"criterion": k, "passed": bool(v)}
        for k, v in criteria.items()
    ]
    verdict_rows.append({
        "criterion": "FINAL_VERDICT",
        "passed": verdict,
    })
    pd.DataFrame(verdict_rows).to_csv(
        RESULTS / "validation_h1_verdict.csv",
        index=False,
    )

    # Coverage summary
    cov_summary = (
        coverage.groupby("window")
        .agg(
            universe=("ticker", "size"),
            loaded=("loaded", "sum"),
        )
        .reset_index()
    )

    # Report
    report = [
        "# Independent validation — H1: short-term breadth × MACD\n\n",
        "## Status\n",
        "**Validation sample only. No strategy rule is changed and V0.2-C "
        "is not created by this test.**\n\n",
        "## Frozen hypothesis from the discovery sample\n",
        "> The incremental value of MACD should be greater when very few "
        "Dividend Aristocrats have a positive 5-session return, and should "
        "diminish as the short-term recovery becomes broad.\n\n",
        "Primary state variable: `breadth_5d_positive_pct`.\n\n",
        "Frozen breadth buckets: `<=25%`, `(25,50]`, `(50,75]`, `>75%`.\n\n",
        "## Independent validation windows\n",
        "- 2018 — late-cycle volatility / Q4 sell-off.\n",
        "- 2019 — broad bull recovery.\n",
        "- 2021 — post-COVID bull market.\n",
        "- 2023–2024 — post-2022 recovery / modern bull market.\n\n",
        "These windows were not used to discover H1.\n\n",
        "## Coverage\n\n",
        "|Window|Universe|Loaded|\n",
        "|---|---:|---:|\n",
    ]

    for _, r in cov_summary.iterrows():
        report.append(
            f"|{r.window}|{int(r.universe)}|{int(r.loaded)}|\n"
        )

    report += [
        "\n## Overall MACD lift — validation sample\n\n",
        "Negative ΔStop favors MACD_PASS. Positive Δ+3, Δ+5 and ΔRet20 favor MACD_PASS.\n\n",
        "|Window|N|Pass N|Fail N|ΔStop pp|Δ+3 pp|Δ+5 pp|ΔRet20 pp|\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|\n",
    ]

    for _, r in overall_df.iterrows():
        report.append(
            f"|{r.window}|{int(r.n)}|{int(r.pass_n)}|{int(r.fail_n)}|"
            f"{r.delta_stop_pp:.2f}|{r.delta_hit3_pp:.2f}|"
            f"{r.delta_hit5_pp:.2f}|{r.delta_ret20_pp:.2f}|\n"
        )

    report += [
        "\n## Primary H1 test — pooled fixed breadth buckets\n\n",
        "|Breadth 5d+|N|Pass N|Fail N|ΔStop pp|Δ+3 pp|Δ+5 pp|ΔRet20 pp|\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|\n",
    ]

    pooled_buckets = bucket_df[
        bucket_df["window"] == "ALL_VALIDATION_POOLED"
    ]

    for _, r in pooled_buckets.iterrows():
        report.append(
            f"|{r.breadth_bucket}|{int(r.n)}|{int(r.pass_n)}|{int(r.fail_n)}|"
            f"{r.delta_stop_pp:.2f}|{r.delta_hit3_pp:.2f}|"
            f"{r.delta_hit5_pp:.2f}|{r.delta_ret20_pp:.2f}|\n"
        )

    report += [
        "\n## Cross-window consistency — low breadth `<=25%`\n\n",
        "A validation window is eligible for this check only if the low-breadth "
        "bucket contains at least 5 MACD_PASS and 5 MACD_FAIL candidates.\n\n",
        "|Bucket|Eligible windows|MACD lower stop|MACD higher +3|MACD higher +5|MACD higher Ret20|Median ΔRet20|\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]

    for _, r in consistency_df.iterrows():
        if r["breadth_bucket"] != "<=25":
            continue
        report.append(
            f"|{r.breadth_bucket}|{int(r.eligible_windows)}|"
            f"{int(r.windows_macd_lower_stop)}|"
            f"{int(r.windows_macd_higher_hit3)}|"
            f"{int(r.windows_macd_higher_hit5)}|"
            f"{int(r.windows_macd_higher_ret20)}|"
            f"{r.median_delta_ret20_pp:.2f}|\n"
        )

    report += [
        "\n## Continuous interaction\n\n",
        "Model: `outcome ~ MACD + breadth_z + MACD×breadth_z + window fixed effects`, "
        "with standard errors clustered by signal date.\n\n",
        "|Outcome|Interaction per 1 SD breadth|SE|p-value|N|\n",
        "|---|---:|---:|---:|---:|\n",
    ]

    for _, r in regression_df.iterrows():
        report.append(
            f"|{r.outcome}|{r.interaction_coef_per_1sd:.4f}|"
            f"{r.interaction_se:.4f}|{r.interaction_pvalue:.4f}|"
            f"{int(r.n)}|\n"
        )

    report += [
        "\n## Cluster-bootstrap 95% intervals\n\n",
        "|Breadth bucket|N|ΔRet20 low|ΔRet20 high|Δ+5 low|Δ+5 high|ΔStop low|ΔStop high|\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|\n",
    ]

    for _, r in bootstrap_df.iterrows():
        report.append(
            f"|{r.breadth_bucket}|{int(r.n)}|"
            f"{r.delta_ret20_low95:.2f}|{r.delta_ret20_high95:.2f}|"
            f"{r.delta_hit5_low95:.2f}|{r.delta_hit5_high95:.2f}|"
            f"{r.delta_stop_low95:.2f}|{r.delta_stop_high95:.2f}|\n"
        )

    report += [
        "\n## Pre-specified directional verdict\n",
        "Four criteria were frozen before seeing the validation results:\n",
        "1. MACD Ret20 lift is positive in pooled breadth `<=25%`.\n",
        "2. MACD Ret20 lift in `<=25%` is greater than in `>75%`.\n",
        "3. The continuous MACD×breadth interaction for Ret20 is negative.\n",
        "4. At least 3 eligible windows exist in `<=25%`, and at least 75% "
        "of them show positive MACD Ret20 lift.\n\n",
    ]

    for k, v in criteria.items():
        report.append(f"- `{k}`: **{bool(v)}**\n")

    report += [
        f"\n### Final validation status: **{verdict}**\n\n",
        "Interpretation rule:\n",
        "- `VALIDATED_DIRECTIONALLY`: H1 may justify a separately specified "
        "V0.2-C experiment, but is not itself a trading rule.\n",
        "- `PARTIAL_NOT_ENOUGH_FOR_V02C`: evidence is interesting but insufficient; "
        "do not create a breadth threshold rule.\n",
        "- `FAILED_VALIDATION`: treat the discovery result as in-sample and do not "
        "promote H1.\n\n",
        "## Guardrails\n",
        "- No threshold search is permitted after this result.\n",
        "- No alternate breadth horizon is tested here.\n",
        "- No ROC/volatility combination is added.\n",
        "- Positive Layer-1 results still do not prove short-put profitability.\n",
    ]

    (RESULTS / "VALIDATION_H1_BREADTH_MACD.md").write_text(
        "".join(report),
        encoding="utf-8",
    )

    print("".join(report))


if __name__ == "__main__":
    main()
