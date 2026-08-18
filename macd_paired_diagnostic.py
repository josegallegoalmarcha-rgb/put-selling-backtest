from __future__ import annotations

import importlib
from pathlib import Path
import numpy as np
import pandas as pd

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

WINDOWS = [
    ("GFC_2007_2009", "backtest", "UNIVERSE_2007_07"),
    ("2015_2016", "backtest_2015_2016", "UNIVERSE_2015_01"),
    ("COVID_2020", "backtest_covid_2020", "UNIVERSE_2020_01"),
    ("2022", "backtest_2022", "UNIVERSE_2022_01"),
]

BOOT_REPS = 2000
SEED = 20260818

def summary(g: pd.DataFrame) -> dict:
    if len(g) == 0:
        return {}
    return {
        "n": len(g),
        "stocks": g["ticker"].nunique(),
        "stop_rate_pct": 100.0 * g["stop_hit_20d"].mean(),
        "hit2_before_stop_pct": 100.0 * g["hit_2pct_before_stop"].mean(),
        "hit3_before_stop_pct": 100.0 * g["hit_3pct_before_stop"].mean(),
        "hit5_before_stop_pct": 100.0 * g["hit_5pct_before_stop"].mean(),
        "avg_mfe_before_stop_pct": g["mfe_before_stop_pct"].mean(),
        "avg_mae_before_stop_pct": g["mae_before_stop_pct"].mean(),
        "avg_ret_5d_pct": g["ret_5d_pct"].mean(),
        "avg_ret_10d_pct": g["ret_10d_pct"].mean(),
        "avg_ret_15d_pct": g["ret_15d_pct"].mean(),
        "avg_ret_20d_pct": g["ret_20d_pct"].mean(),
        "median_swing_distance_pct": g["swing_distance_pct"].median(),
    }

def common_candidate_events(module, universe, window_name: str) -> pd.DataFrame:
    spy = module.load_spy()
    events = []
    coverage = []

    for ticker in universe:
        d = module.load_symbol(ticker)
        if d is None:
            coverage.append({"window":window_name, "ticker":ticker, "loaded":False})
            continue

        coverage.append({
            "window":window_name,
            "ticker":ticker,
            "loaded":True,
            "first":d.index.min(),
            "last":d.index.max(),
            "rows":len(d),
        })

        x = module.attach_spy(module.add_indicators(d), spy)
        v01_raw, control_raw = module.masks(x)

        # KEY DESIGN:
        # one single cooldown is applied to the BASE/CONTROL candidate stream.
        # MACD pass/fail is assigned afterwards. This removes the distortion
        # created when cooldown is applied separately to V0.1 and control.
        for i in module.spaced(control_raw):
            if module.START <= x.index[i] <= module.END:
                group = "MACD_PASS" if bool(v01_raw.iloc[i]) else "MACD_FAIL"
                e = module.evaluate(x, i, ticker, group)
                if e is not None:
                    e["window"] = window_name
                    e["macd_group"] = group
                    events.append(e)

    pd.DataFrame(coverage).to_csv(
        RESULTS / f"macd_paired_coverage_{window_name}.csv", index=False
    )

    if not events:
        return pd.DataFrame()

    ev = pd.DataFrame(events).sort_values(["signal_date","ticker","macd_group"])
    return ev

def lift_table(ev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window, g in ev.groupby("window"):
        p = g[g["macd_group"]=="MACD_PASS"]
        f = g[g["macd_group"]=="MACD_FAIL"]
        if len(p)==0 or len(f)==0:
            continue
        rows.append({
            "window":window,
            "n_pass":len(p),
            "n_fail":len(f),
            "pass_share_pct":100*len(p)/len(g),
            "delta_stop_pp_PASS_minus_FAIL":
                100*p["stop_hit_20d"].mean() - 100*f["stop_hit_20d"].mean(),
            "delta_hit3_pp_PASS_minus_FAIL":
                100*p["hit_3pct_before_stop"].mean() - 100*f["hit_3pct_before_stop"].mean(),
            "delta_hit5_pp_PASS_minus_FAIL":
                100*p["hit_5pct_before_stop"].mean() - 100*f["hit_5pct_before_stop"].mean(),
            "delta_MFE_pp_PASS_minus_FAIL":
                p["mfe_before_stop_pct"].mean() - f["mfe_before_stop_pct"].mean(),
            "delta_MAE_pp_PASS_minus_FAIL":
                p["mae_before_stop_pct"].mean() - f["mae_before_stop_pct"].mean(),
            "delta_ret20_pp_PASS_minus_FAIL":
                p["ret_20d_pct"].mean() - f["ret_20d_pct"].mean(),
        })

    # pooled descriptive row
    p = ev[ev["macd_group"]=="MACD_PASS"]
    f = ev[ev["macd_group"]=="MACD_FAIL"]
    if len(p) and len(f):
        rows.append({
            "window":"ALL_WINDOWS_POOLED",
            "n_pass":len(p),
            "n_fail":len(f),
            "pass_share_pct":100*len(p)/len(ev),
            "delta_stop_pp_PASS_minus_FAIL":
                100*p["stop_hit_20d"].mean() - 100*f["stop_hit_20d"].mean(),
            "delta_hit3_pp_PASS_minus_FAIL":
                100*p["hit_3pct_before_stop"].mean() - 100*f["hit_3pct_before_stop"].mean(),
            "delta_hit5_pp_PASS_minus_FAIL":
                100*p["hit_5pct_before_stop"].mean() - 100*f["hit_5pct_before_stop"].mean(),
            "delta_MFE_pp_PASS_minus_FAIL":
                p["mfe_before_stop_pct"].mean() - f["mfe_before_stop_pct"].mean(),
            "delta_MAE_pp_PASS_minus_FAIL":
                p["mae_before_stop_pct"].mean() - f["mae_before_stop_pct"].mean(),
            "delta_ret20_pp_PASS_minus_FAIL":
                p["ret_20d_pct"].mean() - f["ret_20d_pct"].mean(),
        })
    return pd.DataFrame(rows)

def cluster_bootstrap_one(g: pd.DataFrame, reps=BOOT_REPS, seed=SEED) -> dict:
    # Resample signal dates as clusters so same-day cross-sectional signals
    # remain together. This is more conservative than treating every stock
    # signal as independent.
    rng = np.random.default_rng(seed)
    x = g.copy()
    x["cluster"] = pd.to_datetime(x["signal_date"]).dt.strftime("%Y-%m-%d")
    clusters = x["cluster"].unique().tolist()

    stats = {"stop":[], "hit3":[], "ret20":[]}
    if len(clusters) < 2:
        return {}

    by_cluster = {c:x[x["cluster"]==c] for c in clusters}
    n = len(clusters)

    for _ in range(reps):
        picks = rng.choice(clusters, size=n, replace=True)
        samp = pd.concat([by_cluster[c] for c in picks], ignore_index=True)
        p = samp[samp["macd_group"]=="MACD_PASS"]
        f = samp[samp["macd_group"]=="MACD_FAIL"]
        if len(p)==0 or len(f)==0:
            continue
        stats["stop"].append(
            100*p["stop_hit_20d"].mean() - 100*f["stop_hit_20d"].mean()
        )
        stats["hit3"].append(
            100*p["hit_3pct_before_stop"].mean() - 100*f["hit_3pct_before_stop"].mean()
        )
        stats["ret20"].append(
            p["ret_20d_pct"].mean() - f["ret_20d_pct"].mean()
        )

    out = {}
    for k, vals in stats.items():
        if len(vals) >= 100:
            lo, hi = np.percentile(vals, [2.5,97.5])
            out[f"{k}_ci95_low"] = lo
            out[f"{k}_ci95_high"] = hi
            out[f"{k}_boot_valid"] = len(vals)
    return out

def bootstrap_table(ev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, (window, g) in enumerate(ev.groupby("window")):
        r = {"window":window}
        r.update(cluster_bootstrap_one(g, seed=SEED+idx))
        rows.append(r)

    # pooled: window+date cluster prevents same calendar date in different
    # historical windows being treated as same cluster.
    pooled = ev.copy()
    pooled["signal_date"] = (
        pooled["window"].astype(str) + "__" +
        pd.to_datetime(pooled["signal_date"]).dt.strftime("%Y-%m-%d")
    )
    r = {"window":"ALL_WINDOWS_POOLED"}
    r.update(cluster_bootstrap_one(pooled, seed=SEED+999))
    rows.append(r)
    return pd.DataFrame(rows)

def md_table(df, cols, names=None):
    names = names or cols
    out = ["|"+"|".join(names)+"|\n", "|"+"|".join(["---"]*len(cols))+"|\n"]
    for _, row in df.iterrows():
        vals=[]
        for c in cols:
            v=row[c]
            if isinstance(v, (float, np.floating)):
                vals.append("" if pd.isna(v) else f"{v:.2f}")
            else:
                vals.append(str(v))
        out.append("|"+"|".join(vals)+"|\n")
    return "".join(out)

def main():
    all_events = []

    for window_name, module_name, universe_name in WINDOWS:
        module = importlib.import_module(module_name)
        universe = getattr(module, universe_name)
        ev = common_candidate_events(module, universe, window_name)
        if len(ev):
            all_events.append(ev)

    if not all_events:
        raise RuntimeError("No paired diagnostic events generated.")

    ev = pd.concat(all_events, ignore_index=True)
    ev.to_csv(RESULTS/"macd_paired_events_all_windows.csv", index=False)

    rows = []
    for (window, grp), g in ev.groupby(["window","macd_group"]):
        r = {"window":window, "macd_group":grp}
        r.update(summary(g))
        rows.append(r)

    # pooled descriptive summary
    for grp, g in ev.groupby("macd_group"):
        r = {"window":"ALL_WINDOWS_POOLED", "macd_group":grp}
        r.update(summary(g))
        rows.append(r)

    summ = pd.DataFrame(rows)
    summ.to_csv(RESULTS/"macd_paired_summary.csv", index=False)

    regime_rows = []
    for (window, regime, grp), g in ev.groupby(["window","regime","macd_group"]):
        r={"window":window,"regime":regime,"macd_group":grp}
        r.update(summary(g))
        regime_rows.append(r)
    pd.DataFrame(regime_rows).to_csv(
        RESULTS/"macd_paired_summary_by_regime.csv", index=False
    )

    lift = lift_table(ev)
    lift.to_csv(RESULTS/"macd_paired_lift.csv", index=False)

    boot = bootstrap_table(ev)
    boot.to_csv(RESULTS/"macd_paired_cluster_bootstrap.csv", index=False)

    merged = lift.merge(boot, on="window", how="left")

    report = [
        "# MACD common-candidate diagnostic — four frozen windows\n\n",
        "## Purpose\n",
        "This is a **diagnostic only**. It does not change V0.1 and is not a new strategy version.\n\n",
        "The prior V0.1-vs-control comparison applied the 20-session cooldown separately "
        "to each signal stream. That can make V0.1 and control samples differ for reasons "
        "unrelated to MACD. Here we remove that distortion.\n\n",
        "## Design\n",
        "1. Build the same base candidate pool: stock MA50>MA200, allowed SPY regime, "
        "Slow Stochastic 14,3,3 cross above 20.\n",
        "2. Apply **one common 20-session cooldown to the base candidate stream**.\n",
        "3. Only after candidate selection, label each episode `MACD_PASS` or `MACD_FAIL` "
        "using the frozen V0.1 MACD rule.\n",
        "4. Compare outcomes inside that common candidate pool.\n",
        "5. Bootstrap confidence intervals resample **signal dates as clusters**, not "
        "individual stocks, to preserve same-day market clustering.\n\n",
        "A negative `Δ Stop` favors MACD_PASS. Positive `Δ +3`, `Δ +5`, MFE and "
        "`Δ Ret20` favor MACD_PASS. Less-negative MAE also favors MACD_PASS.\n\n",
        "## Summary by window and MACD label\n\n",
    ]

    report.append(md_table(
        summ,
        ["window","macd_group","n","stop_rate_pct","hit3_before_stop_pct",
         "hit5_before_stop_pct","avg_mfe_before_stop_pct",
         "avg_mae_before_stop_pct","avg_ret_20d_pct"],
        ["Window","Group","N","Stop %","+3%","+5%","MFE %","MAE %","Ret20 %"]
    ))

    report += ["\n## Incremental MACD lift inside common candidates\n\n"]
    report.append(md_table(
        merged,
        ["window","n_pass","n_fail","pass_share_pct",
         "delta_stop_pp_PASS_minus_FAIL","delta_hit3_pp_PASS_minus_FAIL",
         "delta_hit5_pp_PASS_minus_FAIL","delta_MFE_pp_PASS_minus_FAIL",
         "delta_MAE_pp_PASS_minus_FAIL","delta_ret20_pp_PASS_minus_FAIL"],
        ["Window","Pass N","Fail N","Pass %","Δ Stop pp","Δ +3 pp",
         "Δ +5 pp","Δ MFE pp","Δ MAE pp","Δ Ret20 pp"]
    ))

    report += [
        "\n## Cluster-bootstrap 95% intervals for key lifts\n\n",
        "Intervals are descriptive robustness checks, not formal causal proof.\n\n",
    ]
    report.append(md_table(
        merged,
        ["window","stop_ci95_low","stop_ci95_high",
         "hit3_ci95_low","hit3_ci95_high",
         "ret20_ci95_low","ret20_ci95_high"],
        ["Window","ΔStop low","ΔStop high","Δ+3 low","Δ+3 high",
         "ΔRet20 low","ΔRet20 high"]
    ))

    report += [
        "\n## Guardrails\n",
        "- Do not use this diagnostic to rewrite any historical trade.\n",
        "- Do not promote or delete MACD based on one window alone.\n",
        "- If MACD lift changes sign materially across windows, treat it as regime-dependent "
        "until a pre-specified real-time discriminator is validated.\n",
        "- The next step after interpreting this report is the pre-planned bull-control "
        "windows 2013–2014 and 2017, still with frozen V0.1.\n",
    ]

    (RESULTS/"MACD_PAIRED_DIAGNOSTIC.md").write_text(
        "".join(report), encoding="utf-8"
    )
    print("".join(report))

if __name__ == "__main__":
    main()
