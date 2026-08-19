from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import binomtest

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

DISCOVERY_FILE = RESULTS / "v02b_no_macd_common_candidate_events.csv"
VALIDATION_FILE = RESULTS / "validation_h1_common_candidates.csv"

EXPECTED_DISCOVERY = 1278
EXPECTED_VALIDATION = 1001
EXPECTED_TOTAL = 2279

BOOT_REPS = 3000
SEED = 20260819

# ---------------------------------------------------------------------
# MASTER LAYER-1 DIAGNOSTIC
# ---------------------------------------------------------------------
#
# Diagnostic only. No strategy rule is changed.
#
# Purpose:
#   consolidate the 1,278 discovery common candidates and the 1,001
#   independent validation common candidates, then decide whether the
#   Layer-1 evidence is mature enough to justify proceeding to Layer 2
#   (PUT monetization) without pretending that regime heterogeneity has
#   been solved.
#
# Core questions:
#   1) What are the absolute directional characteristics of the frozen
#      base candidate stream (trend + regime + stochastic)?
#   2) What is the incremental MACD_PASS vs MACD_FAIL effect over all
#      2,279 common candidates?
#   3) Does that effect replicate from discovery to validation?
#   4) How heterogeneous is it across windows, SPY GREEN/YELLOW regimes,
#      and broad descriptive market families?
#   5) How much same-day clustering exists?
#
# IMPORTANT:
# - The base candidate stream is NOT compared here with unconditional
#   market returns. Therefore positive absolute returns are not proof of
#   standalone alpha.
# - This is still underlying-price Layer 1, not option P/L.
# - Window-family labels are descriptive summaries of the original test
#   design, not trading rules.
# ---------------------------------------------------------------------

WINDOW_FAMILY = {
    "GFC_2007_2009": "CRISIS",
    "BULL_2013_2014": "BULL",
    "CHOP_2015_2016": "CORRECTION_BEAR",
    "BULL_2017": "BULL",
    "COVID_2020": "CRISIS",
    "BEAR_2022": "CORRECTION_BEAR",
    "VALIDATION_2018": "CORRECTION_BEAR",
    "VALIDATION_2019": "BULL",
    "VALIDATION_2021": "BULL",
    "VALIDATION_2023_2024": "BULL",
}

WINDOW_ORDER = [
    "GFC_2007_2009",
    "BULL_2013_2014",
    "CHOP_2015_2016",
    "BULL_2017",
    "COVID_2020",
    "BEAR_2022",
    "VALIDATION_2018",
    "VALIDATION_2019",
    "VALIDATION_2021",
    "VALIDATION_2023_2024",
]

REQUIRED_COLUMNS = {
    "window",
    "ticker",
    "signal_date",
    "regime",
    "macd_group",
    "macd_pass",
    "stop_hit_20d",
    "hit_3pct_before_stop",
    "hit_5pct_before_stop",
    "mfe_before_stop_pct",
    "mae_before_stop_pct",
    "ret_5d_pct",
    "ret_10d_pct",
    "ret_15d_pct",
    "ret_20d_pct",
}


def to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }

    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
        .astype("boolean")
    )


def load_source(path: Path, sample: str, expected_n: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required source file missing: {path}")

    d = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(d.columns)
    if missing:
        raise RuntimeError(
            f"{path.name} missing required columns: {sorted(missing)}"
        )

    if len(d) != expected_n:
        raise RuntimeError(
            f"{path.name}: expected {expected_n} rows, found {len(d)}. "
            "Stop and inspect source before running the master diagnostic."
        )

    d = d.copy()
    d["sample"] = sample
    d["signal_date"] = pd.to_datetime(d["signal_date"])

    for col in [
        "macd_pass",
        "stop_hit_20d",
        "hit_3pct_before_stop",
        "hit_5pct_before_stop",
    ]:
        d[col] = to_bool(d[col])

    if d[
        [
            "macd_pass",
            "stop_hit_20d",
            "hit_3pct_before_stop",
            "hit_5pct_before_stop",
        ]
    ].isna().any().any():
        raise RuntimeError(f"Boolean parsing failed in {path.name}")

    expected_groups = {"MACD_PASS", "MACD_FAIL"}
    if set(d["macd_group"].dropna().unique()) != expected_groups:
        raise RuntimeError(
            f"Unexpected MACD groups in {path.name}: "
            f"{sorted(d['macd_group'].dropna().unique())}"
        )

    if not np.array_equal(
        d["macd_pass"].astype(bool).to_numpy(),
        (d["macd_group"] == "MACD_PASS").to_numpy(),
    ):
        raise RuntimeError(f"macd_pass and macd_group disagree in {path.name}")

    return d


def prepare_master() -> pd.DataFrame:
    discovery = load_source(
        DISCOVERY_FILE, "DISCOVERY", EXPECTED_DISCOVERY
    )
    validation = load_source(
        VALIDATION_FILE, "VALIDATION", EXPECTED_VALIDATION
    )

    overlap = set(discovery["window"]).intersection(set(validation["window"]))
    if overlap:
        raise RuntimeError(
            f"Discovery/validation window overlap detected: {sorted(overlap)}"
        )

    d = pd.concat([discovery, validation], ignore_index=True, sort=False)

    if len(d) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} total candidates, found {len(d)}"
        )

    unknown = sorted(set(d["window"]) - set(WINDOW_FAMILY))
    if unknown:
        raise RuntimeError(f"Unclassified windows: {unknown}")

    d["window_family"] = d["window"].map(WINDOW_FAMILY)

    d["cluster_key"] = (
        d["window"].astype(str)
        + "__"
        + d["signal_date"].dt.strftime("%Y-%m-%d")
    )

    cluster_size = d.groupby("cluster_key")["ticker"].transform("size")
    d["cluster_size_master"] = cluster_size.astype(int)

    d["macd_pass_int"] = (d["macd_group"] == "MACD_PASS").astype(int)
    d["stop_int"] = d["stop_hit_20d"].astype(int)
    d["hit3_int"] = d["hit_3pct_before_stop"].astype(int)
    d["hit5_int"] = d["hit_5pct_before_stop"].astype(int)

    return d


# ---------------------------------------------------------------------
# ABSOLUTE BASE-CANDIDATE METRICS
# ---------------------------------------------------------------------

def absolute_metrics(g: pd.DataFrame) -> dict:
    r20 = pd.to_numeric(g["ret_20d_pct"], errors="coerce")

    return {
        "n": len(g),
        "stocks": g["ticker"].nunique(),
        "signal_days": g["cluster_key"].nunique(),
        "stop_rate_pct": 100 * g["stop_hit_20d"].mean(),
        "hit3_pct": 100 * g["hit_3pct_before_stop"].mean(),
        "hit5_pct": 100 * g["hit_5pct_before_stop"].mean(),
        "avg_mfe_pct": g["mfe_before_stop_pct"].mean(),
        "avg_mae_pct": g["mae_before_stop_pct"].mean(),
        "avg_ret5_pct": g["ret_5d_pct"].mean(),
        "avg_ret10_pct": g["ret_10d_pct"].mean(),
        "avg_ret15_pct": g["ret_15d_pct"].mean(),
        "avg_ret20_pct": r20.mean(),
        "median_ret20_pct": r20.median(),
        "positive_ret20_pct": 100 * (r20 > 0).mean(),
        "ret20_q05_pct": r20.quantile(0.05),
        "ret20_q10_pct": r20.quantile(0.10),
        "ret20_q90_pct": r20.quantile(0.90),
        "ret20_q95_pct": r20.quantile(0.95),
        "worst_decile_mean_ret20_pct": r20[r20 <= r20.quantile(0.10)].mean(),
    }


def absolute_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(label_type, label, g):
        r = {"group_type": label_type, "group": label}
        r.update(absolute_metrics(g))
        rows.append(r)

    add("TOTAL", "ALL_2279", df)

    for sample, g in df.groupby("sample"):
        add("SAMPLE", sample, g)

    for window in WINDOW_ORDER:
        g = df[df["window"] == window]
        if not g.empty:
            add("WINDOW", window, g)

    for regime, g in df.groupby("regime"):
        add("REGIME", regime, g)

    for family, g in df.groupby("window_family"):
        add("WINDOW_FAMILY", family, g)

    for macd, g in df.groupby("macd_group"):
        add("MACD_GROUP", macd, g)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# MACD LIFT
# ---------------------------------------------------------------------

def macd_lift(g: pd.DataFrame) -> dict | None:
    p = g[g["macd_group"] == "MACD_PASS"]
    f = g[g["macd_group"] == "MACD_FAIL"]

    if len(p) == 0 or len(f) == 0:
        return None

    return {
        "n": len(g),
        "pass_n": len(p),
        "fail_n": len(f),
        "pass_share_pct": 100 * len(p) / len(g),
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
        "delta_ret5_pp": p["ret_5d_pct"].mean() - f["ret_5d_pct"].mean(),
        "delta_ret10_pp": p["ret_10d_pct"].mean() - f["ret_10d_pct"].mean(),
        "delta_ret15_pp": p["ret_15d_pct"].mean() - f["ret_15d_pct"].mean(),
        "delta_ret20_pp": p["ret_20d_pct"].mean() - f["ret_20d_pct"].mean(),
    }


def lift_table(df: pd.DataFrame, group_col: str, order=None) -> pd.DataFrame:
    rows = []

    groups = list(df.groupby(group_col))

    if order is not None:
        gm = {k: g for k, g in groups}
        groups = [(k, gm[k]) for k in order if k in gm]

    for key, g in groups:
        v = macd_lift(g)
        if v is None:
            continue
        r = {group_col: key}
        r.update(v)
        rows.append(r)

    return pd.DataFrame(rows)


def pooled_lift(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for label, g in [
        ("ALL_2279", df),
        ("DISCOVERY", df[df["sample"] == "DISCOVERY"]),
        ("VALIDATION", df[df["sample"] == "VALIDATION"]),
    ]:
        v = macd_lift(g)
        r = {"group": label}
        r.update(v or {})
        rows.append(r)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# DATE-CLUSTER BOOTSTRAP
# ---------------------------------------------------------------------

def effect_value(g: pd.DataFrame, metric: str) -> float:
    v = macd_lift(g)
    if v is None:
        return np.nan
    return float(v[metric])


BOOT_METRICS = [
    "delta_stop_pp",
    "delta_hit3_pp",
    "delta_hit5_pp",
    "delta_mfe_pp",
    "delta_mae_pp",
    "delta_ret20_pp",
]


def date_cluster_bootstrap(
    g: pd.DataFrame,
    reps: int = BOOT_REPS,
    seed: int = SEED,
) -> dict:
    clusters = g["cluster_key"].drop_duplicates().to_numpy()

    if len(clusters) < 2:
        return {}

    grouped = {
        k: part.copy()
        for k, part in g.groupby("cluster_key")
    }

    rng = np.random.default_rng(seed)
    store = {m: [] for m in BOOT_METRICS}

    for _ in range(reps):
        sampled = rng.choice(
            clusters,
            size=len(clusters),
            replace=True,
        )

        pieces = [grouped[k] for k in sampled]
        b = pd.concat(pieces, ignore_index=True)

        for m in BOOT_METRICS:
            val = effect_value(b, m)
            if np.isfinite(val):
                store[m].append(val)

    out = {
        "clusters": len(clusters),
        "bootstrap_reps": reps,
    }

    for m, vals in store.items():
        if vals:
            arr = np.asarray(vals)
            out[f"{m}_low95"] = np.quantile(arr, 0.025)
            out[f"{m}_high95"] = np.quantile(arr, 0.975)
        else:
            out[f"{m}_low95"] = np.nan
            out[f"{m}_high95"] = np.nan

    return out


def bootstrap_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    groups = [
        ("ALL_2279", df),
        ("DISCOVERY", df[df["sample"] == "DISCOVERY"]),
        ("VALIDATION", df[df["sample"] == "VALIDATION"]),
    ]

    for i, (label, g) in enumerate(groups):
        r = {"group": label}
        r.update(
            date_cluster_bootstrap(
                g,
                reps=BOOT_REPS,
                seed=SEED + i,
            )
        )
        rows.append(r)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# WINDOW-BALANCED EVIDENCE
# ---------------------------------------------------------------------

FAVORABLE_DIRECTION = {
    "delta_stop_pp": "negative",
    "delta_hit3_pp": "positive",
    "delta_hit5_pp": "positive",
    "delta_mfe_pp": "positive",
    "delta_mae_pp": "positive",  # less-negative MAE
    "delta_ret20_pp": "positive",
}


def favorable(series: pd.Series, metric: str) -> pd.Series:
    if FAVORABLE_DIRECTION[metric] == "negative":
        return series < 0
    return series > 0


def window_balanced_summary(window_lifts: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for sample_label, sub in [
        ("ALL_WINDOWS", window_lifts),
        (
            "DISCOVERY_WINDOWS",
            window_lifts[
                window_lifts["window"].isin(
                    [
                        "GFC_2007_2009",
                        "BULL_2013_2014",
                        "CHOP_2015_2016",
                        "BULL_2017",
                        "COVID_2020",
                        "BEAR_2022",
                    ]
                )
            ],
        ),
        (
            "VALIDATION_WINDOWS",
            window_lifts[
                window_lifts["window"].isin(
                    [
                        "VALIDATION_2018",
                        "VALIDATION_2019",
                        "VALIDATION_2021",
                        "VALIDATION_2023_2024",
                    ]
                )
            ],
        ),
    ]:
        for metric in FAVORABLE_DIRECTION:
            vals = sub[metric].dropna()
            n = len(vals)
            fav = int(favorable(vals, metric).sum())

            # Exact sign-test against 50%, descriptive only.
            p = (
                binomtest(fav, n=n, p=0.5, alternative="greater").pvalue
                if n > 0
                else np.nan
            )

            rows.append({
                "window_set": sample_label,
                "metric": metric,
                "n_windows": n,
                "favorable_windows": fav,
                "favorable_share_pct": 100*fav/n if n else np.nan,
                "equal_weight_mean": vals.mean(),
                "median": vals.median(),
                "std_across_windows": vals.std(ddof=1),
                "min": vals.min(),
                "max": vals.max(),
                "sign_test_p_one_sided": p,
            })

    return pd.DataFrame(rows)


def window_bootstrap(window_lifts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(SEED + 1000)

    for sample_label, sub in [
        ("ALL_WINDOWS", window_lifts),
        (
            "DISCOVERY_WINDOWS",
            window_lifts[
                ~window_lifts["window"].str.startswith("VALIDATION_")
            ],
        ),
        (
            "VALIDATION_WINDOWS",
            window_lifts[
                window_lifts["window"].str.startswith("VALIDATION_")
            ],
        ),
    ]:
        for metric in FAVORABLE_DIRECTION:
            vals = sub[metric].dropna().to_numpy()

            if len(vals) == 0:
                continue

            means = np.empty(BOOT_REPS)

            for i in range(BOOT_REPS):
                sample = rng.choice(
                    vals,
                    size=len(vals),
                    replace=True,
                )
                means[i] = sample.mean()

            rows.append({
                "window_set": sample_label,
                "metric": metric,
                "n_windows": len(vals),
                "equal_weight_mean": vals.mean(),
                "bootstrap_low95": np.quantile(means, 0.025),
                "bootstrap_high95": np.quantile(means, 0.975),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# HETEROGENEITY TESTS
# ---------------------------------------------------------------------

OUTCOME_FORMULAS = {
    "stop": "stop_int",
    "hit3": "hit3_int",
    "hit5": "hit5_int",
    "ret20": "ret_20d_pct",
}


def clustered_model(df: pd.DataFrame, formula: str):
    use_cols = [
        "macd_pass_int",
        "window",
        "sample",
        "window_family",
        "cluster_key",
        "stop_int",
        "hit3_int",
        "hit5_int",
        "ret_20d_pct",
    ]
    use = df[use_cols].dropna().copy()

    return smf.ols(formula, data=use).fit(
        cov_type="cluster",
        cov_kwds={"groups": use["cluster_key"]},
    )


def joint_interaction_pvalue(model, token: str) -> tuple[int, float]:
    names = model.model.exog_names
    idx = [
        i for i, name in enumerate(names)
        if token in name and ":" in name
    ]

    if not idx:
        return 0, np.nan

    R = np.zeros((len(idx), len(names)))
    for row, col in enumerate(idx):
        R[row, col] = 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test = model.wald_test(R, scalar=True)

    return len(idx), float(test.pvalue)


def heterogeneity_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for outcome_name, col in OUTCOME_FORMULAS.items():
        # Window heterogeneity.
        m_window = clustered_model(
            df,
            f"{col} ~ macd_pass_int * C(window)",
        )
        n_terms, p = joint_interaction_pvalue(
            m_window,
            "macd_pass_int:C(window)",
        )
        rows.append({
            "outcome": outcome_name,
            "heterogeneity_dimension": "WINDOW",
            "interaction_terms": n_terms,
            "pvalue_joint_interactions": p,
            "n": int(m_window.nobs),
        })

        # Discovery vs validation shift.
        m_sample = clustered_model(
            df,
            f"{col} ~ macd_pass_int * C(sample) + C(window)",
        )
        n_terms, p = joint_interaction_pvalue(
            m_sample,
            "macd_pass_int:C(sample)",
        )

        # Extract the one interaction coefficient if available.
        coef = np.nan
        for name, val in m_sample.params.items():
            if "macd_pass_int:C(sample)" in name:
                coef = float(val)

        rows.append({
            "outcome": outcome_name,
            "heterogeneity_dimension": "DISCOVERY_VS_VALIDATION",
            "interaction_terms": n_terms,
            "pvalue_joint_interactions": p,
            "interaction_coef": coef,
            "n": int(m_sample.nobs),
        })

        # Descriptive market family.
        m_family = clustered_model(
            df,
            f"{col} ~ macd_pass_int * C(window_family) + C(window)",
        )
        n_terms, p = joint_interaction_pvalue(
            m_family,
            "macd_pass_int:C(window_family)",
        )
        rows.append({
            "outcome": outcome_name,
            "heterogeneity_dimension": "WINDOW_FAMILY",
            "interaction_terms": n_terms,
            "pvalue_joint_interactions": p,
            "n": int(m_family.nobs),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# CLUSTERING
# ---------------------------------------------------------------------

def clustering_metrics(g: pd.DataFrame) -> dict:
    day = (
        g.groupby(["window", "signal_date"])
        .size()
        .rename("signals")
        .reset_index()
    )

    if day.empty:
        return {}

    signals_total = int(day["signals"].sum())

    def share_on(threshold: int) -> float:
        selected = day.loc[
            day["signals"] >= threshold,
            "signals",
        ].sum()
        return 100 * selected / signals_total if signals_total else np.nan

    return {
        "signals": signals_total,
        "signal_days": len(day),
        "avg_signals_per_signal_day": day["signals"].mean(),
        "median_signals_per_signal_day": day["signals"].median(),
        "max_same_day": int(day["signals"].max()),
        "days_ge3": int((day["signals"] >= 3).sum()),
        "days_ge5": int((day["signals"] >= 5).sum()),
        "days_ge10": int((day["signals"] >= 10).sum()),
        "share_signals_on_ge3_days_pct": share_on(3),
        "share_signals_on_ge5_days_pct": share_on(5),
        "share_signals_on_ge10_days_pct": share_on(10),
    }


def clustering_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for window in WINDOW_ORDER:
        wg = df[df["window"] == window]
        if wg.empty:
            continue

        for group_label, g in [
            ("ALL_BASE", wg),
            ("MACD_PASS", wg[wg["macd_group"] == "MACD_PASS"]),
            ("MACD_FAIL", wg[wg["macd_group"] == "MACD_FAIL"]),
        ]:
            r = {
                "window": window,
                "group": group_label,
            }
            r.update(clustering_metrics(g))
            rows.append(r)

    return pd.DataFrame(rows)


def top_clusters(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (window, dt), g in df.groupby(["window", "signal_date"]):
        rows.append({
            "window": window,
            "sample": g["sample"].iloc[0],
            "window_family": g["window_family"].iloc[0],
            "signal_date": dt,
            "signals": len(g),
            "stocks": g["ticker"].nunique(),
            "macd_pass_share_pct": 100*g["macd_pass"].mean(),
            "stop_rate_pct": 100*g["stop_hit_20d"].mean(),
            "hit3_pct": 100*g["hit_3pct_before_stop"].mean(),
            "hit5_pct": 100*g["hit_5pct_before_stop"].mean(),
            "avg_ret20_pct": g["ret_20d_pct"].mean(),
            "regime": g["regime"].mode().iloc[0]
            if not g["regime"].mode().empty else "UNKNOWN",
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["signals", "signal_date"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------

def fmt(x, digits=2):
    if pd.isna(x):
        return "—"
    return f"{x:.{digits}f}"


def main():
    df = prepare_master()

    # Save master sample.
    df.to_csv(
        RESULTS / "master_layer1_common_candidates_2279.csv",
        index=False,
    )

    absolute = absolute_summary(df)
    absolute.to_csv(
        RESULTS / "master_layer1_absolute_metrics.csv",
        index=False,
    )

    pooled = pooled_lift(df)
    pooled.to_csv(
        RESULTS / "master_layer1_macd_pooled.csv",
        index=False,
    )

    by_window = lift_table(df, "window", order=WINDOW_ORDER)
    by_window.to_csv(
        RESULTS / "master_layer1_macd_by_window.csv",
        index=False,
    )

    by_sample = lift_table(df, "sample")
    by_sample.to_csv(
        RESULTS / "master_layer1_macd_by_sample.csv",
        index=False,
    )

    by_regime = lift_table(df, "regime")
    by_regime.to_csv(
        RESULTS / "master_layer1_macd_by_regime.csv",
        index=False,
    )

    by_family = lift_table(df, "window_family")
    by_family.to_csv(
        RESULTS / "master_layer1_macd_by_window_family.csv",
        index=False,
    )

    boot = bootstrap_summary(df)
    boot.to_csv(
        RESULTS / "master_layer1_date_cluster_bootstrap.csv",
        index=False,
    )

    balanced = window_balanced_summary(by_window)
    balanced.to_csv(
        RESULTS / "master_layer1_window_balanced_summary.csv",
        index=False,
    )

    wboot = window_bootstrap(by_window)
    wboot.to_csv(
        RESULTS / "master_layer1_window_bootstrap.csv",
        index=False,
    )

    hetero = heterogeneity_tests(df)
    hetero.to_csv(
        RESULTS / "master_layer1_heterogeneity_tests.csv",
        index=False,
    )

    clustering = clustering_summary(df)
    clustering.to_csv(
        RESULTS / "master_layer1_clustering.csv",
        index=False,
    )

    clusters = top_clusters(df)
    clusters.to_csv(
        RESULTS / "master_layer1_top_clusters.csv",
        index=False,
    )

    # Integrity summary.
    integrity = pd.DataFrame([
        {
            "sample": "DISCOVERY",
            "expected_n": EXPECTED_DISCOVERY,
            "actual_n": int((df["sample"] == "DISCOVERY").sum()),
        },
        {
            "sample": "VALIDATION",
            "expected_n": EXPECTED_VALIDATION,
            "actual_n": int((df["sample"] == "VALIDATION").sum()),
        },
        {
            "sample": "TOTAL",
            "expected_n": EXPECTED_TOTAL,
            "actual_n": len(df),
        },
    ])
    integrity["match"] = (
        integrity["expected_n"] == integrity["actual_n"]
    )
    integrity.to_csv(
        RESULTS / "master_layer1_integrity.csv",
        index=False,
    )

    # Build compact markdown report.
    total_abs = absolute[
        (absolute["group_type"] == "TOTAL")
        & (absolute["group"] == "ALL_2279")
    ].iloc[0]

    report = [
        "# MASTER Layer-1 diagnostic — 2,279 common candidates\n\n",
        "## Status\n",
        "**Diagnostic only. No V0.1/V0.2 rule is changed.**\n\n",
        "This consolidates the frozen **1,278 discovery candidates** and the "
        "**1,001 independent validation candidates**. The purpose is to decide "
        "whether Layer 1 is mature enough to justify testing PUT monetization "
        "in Layer 2, while preserving the observed regime heterogeneity.\n\n",
        "## Integrity\n",
        f"- Discovery: **{EXPECTED_DISCOVERY:,}** candidates.\n",
        f"- Validation: **{EXPECTED_VALIDATION:,}** candidates.\n",
        f"- Total: **{EXPECTED_TOTAL:,}** candidates.\n",
        f"- Windows: **{df['window'].nunique()}**.\n",
        f"- Distinct signal dates/clusters: **{df['cluster_key'].nunique()}**.\n\n",
        "## 1. Absolute base-candidate behavior\n\n",
        "The base stream is MA50>MA200 + SPY GREEN/YELLOW + Stochastic cross "
        "with common per-stock cooldown. These absolute results are **not an "
        "unconditional-market alpha test**.\n\n",
        "|N|Stop %|+3 %|+5 %|MFE %|MAE %|Ret20 avg %|Ret20 median %|Ret20 >0 %|Ret20 q05 %|\n",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        f"|{int(total_abs.n)}|{fmt(total_abs.stop_rate_pct)}|"
        f"{fmt(total_abs.hit3_pct)}|{fmt(total_abs.hit5_pct)}|"
        f"{fmt(total_abs.avg_mfe_pct)}|{fmt(total_abs.avg_mae_pct)}|"
        f"{fmt(total_abs.avg_ret20_pct)}|{fmt(total_abs.median_ret20_pct)}|"
        f"{fmt(total_abs.positive_ret20_pct)}|{fmt(total_abs.ret20_q05_pct)}|\n\n",
        "## 2. MACD incremental effect — pooled and replication\n\n",
        "Negative ΔStop favors MACD_PASS. Positive Δ+3, Δ+5, ΔMFE, ΔMAE "
        "(less-negative MAE) and ΔRet20 favor MACD_PASS.\n\n",
        "|Sample|N|Pass N|Fail N|ΔStop pp|Δ+3 pp|Δ+5 pp|ΔMFE|ΔMAE|ΔRet20 pp|\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]

    for _, r in pooled.iterrows():
        report.append(
            f"|{r['group']}|{int(r.n)}|{int(r.pass_n)}|{int(r.fail_n)}|"
            f"{fmt(r.delta_stop_pp)}|{fmt(r.delta_hit3_pp)}|"
            f"{fmt(r.delta_hit5_pp)}|{fmt(r.delta_mfe_pp)}|"
            f"{fmt(r.delta_mae_pp)}|{fmt(r.delta_ret20_pp)}|\n"
        )

    report += [
        "\n## 3. MACD effect by window\n\n",
        "|Window|N|Pass %|ΔStop pp|Δ+3 pp|Δ+5 pp|ΔRet20 pp|\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]

    for _, r in by_window.iterrows():
        report.append(
            f"|{r.window}|{int(r.n)}|{fmt(r.pass_share_pct)}|"
            f"{fmt(r.delta_stop_pp)}|{fmt(r.delta_hit3_pp)}|"
            f"{fmt(r.delta_hit5_pp)}|{fmt(r.delta_ret20_pp)}|\n"
        )

    report += [
        "\n## 4. SPY regime and descriptive window family\n\n",
        "### By SPY regime\n\n",
        "|Regime|N|ΔStop pp|Δ+3 pp|Δ+5 pp|ΔRet20 pp|\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]

    for _, r in by_regime.iterrows():
        report.append(
            f"|{r.regime}|{int(r.n)}|{fmt(r.delta_stop_pp)}|"
            f"{fmt(r.delta_hit3_pp)}|{fmt(r.delta_hit5_pp)}|"
            f"{fmt(r.delta_ret20_pp)}|\n"
        )

    report += [
        "\n### By descriptive window family\n\n",
        "|Family|N|ΔStop pp|Δ+3 pp|Δ+5 pp|ΔRet20 pp|\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]

    for _, r in by_family.iterrows():
        report.append(
            f"|{r.window_family}|{int(r.n)}|{fmt(r.delta_stop_pp)}|"
            f"{fmt(r.delta_hit3_pp)}|{fmt(r.delta_hit5_pp)}|"
            f"{fmt(r.delta_ret20_pp)}|\n"
        )

    report += [
        "\n## 5. Date-cluster bootstrap 95% intervals\n\n",
        "|Group|Clusters|ΔStop low|high|Δ+3 low|high|Δ+5 low|high|ΔRet20 low|high|\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]

    for _, r in boot.iterrows():
        report.append(
            f"|{r['group']}|{int(r.clusters)}|"
            f"{fmt(r.delta_stop_pp_low95)}|{fmt(r.delta_stop_pp_high95)}|"
            f"{fmt(r.delta_hit3_pp_low95)}|{fmt(r.delta_hit3_pp_high95)}|"
            f"{fmt(r.delta_hit5_pp_low95)}|{fmt(r.delta_hit5_pp_high95)}|"
            f"{fmt(r.delta_ret20_pp_low95)}|{fmt(r.delta_ret20_pp_high95)}|\n"
        )

    report += [
        "\n## 6. Window-balanced robustness\n\n",
        "This prevents large windows from dominating the signal-weighted pooled result.\n\n",
        "|Window set|Metric|Windows|Favorable|Mean effect|Median|Min|Max|Sign-test p|\n",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|\n",
    ]

    for _, r in balanced.iterrows():
        if r.metric not in {
            "delta_stop_pp",
            "delta_hit3_pp",
            "delta_hit5_pp",
            "delta_ret20_pp",
        }:
            continue

        report.append(
            f"|{r.window_set}|{r.metric}|{int(r.n_windows)}|"
            f"{int(r.favorable_windows)}|{fmt(r.equal_weight_mean)}|"
            f"{fmt(r['median'])}|{fmt(r['min'])}|{fmt(r['max'])}|"
            f"{fmt(r.sign_test_p_one_sided, 4)}|\n"
        )

    report += [
        "\n## 7. Formal heterogeneity diagnostics\n\n",
        "Linear-probability/OLS interaction models use standard errors clustered "
        "by signal date. The window test asks whether the MACD effect differs "
        "across windows; the discovery-vs-validation test asks whether the "
        "effect changed materially out of sample.\n\n",
        "|Outcome|Dimension|Interaction terms|p-value|N|\n",
        "|---|---|---:|---:|---:|\n",
    ]

    for _, r in hetero.iterrows():
        report.append(
            f"|{r.outcome}|{r.heterogeneity_dimension}|"
            f"{int(r.interaction_terms)}|"
            f"{fmt(r.pvalue_joint_interactions, 4)}|{int(r.n)}|\n"
        )

    report += [
        "\n## 8. Clustering\n\n",
        "|Window|Group|Signals|Signal days|Max same day|Share on ≥5-signal days %|Share on ≥10-signal days %|\n",
        "|---|---|---:|---:|---:|---:|---:|\n",
    ]

    for _, r in clustering.iterrows():
        if r["group"] not in {"ALL_BASE", "MACD_PASS"}:
            continue

        report.append(
            f"|{r.window}|{r['group']}|{int(r.signals)}|"
            f"{int(r.signal_days)}|{int(r.max_same_day)}|"
            f"{fmt(r.share_signals_on_ge5_days_pct)}|"
            f"{fmt(r.share_signals_on_ge10_days_pct)}|\n"
        )

    report += [
        "\n## 9. Largest same-day candidate clusters\n\n",
        "|Window|Date|Signals|MACD pass %|Stop %|+3 %|+5 %|Ret20 %|Regime|\n",
        "|---|---|---:|---:|---:|---:|---:|---:|---|\n",
    ]

    for _, r in clusters.head(20).iterrows():
        report.append(
            f"|{r.window}|{pd.Timestamp(r.signal_date).date()}|{int(r.signals)}|"
            f"{fmt(r.macd_pass_share_pct)}|{fmt(r.stop_rate_pct)}|"
            f"{fmt(r.hit3_pct)}|{fmt(r.hit5_pct)}|"
            f"{fmt(r.avg_ret20_pct)}|{r.regime}|\n"
        )

    report += [
        "\n## Interpretation guardrails\n",
        "- No rule is promoted or rejected automatically by this diagnostic.\n",
        "- Positive base-candidate Ret20 does not prove alpha versus an unconditional benchmark.\n",
        "- A significant pooled MACD effect does not erase window heterogeneity.\n",
        "- A non-significant discovery-vs-validation interaction is evidence of replication, "
        "not proof of universality.\n",
        "- Same-day clusters are correlated macro events and must not be treated as independent trades.\n",
        "- Layer 2, if opened, must still test option-chain economics, IV/theta, DTE/delta, "
        "earnings/ex-dividend exclusions, contract granularity and portfolio capacity.\n",
        "- This report is designed to support the human decision: continue Layer-1 signal research "
        "or freeze a provisional architecture and proceed to Layer 2.\n",
    ]

    (RESULTS / "MASTER_LAYER1_DIAGNOSTIC.md").write_text(
        "".join(report),
        encoding="utf-8",
    )

    print("".join(report))


if __name__ == "__main__":
    main()
