from pathlib import Path
import numpy as np
import pandas as pd

RESULTS = Path("results")
EVENTS = RESULTS / "events_GFC_2007_2009.csv"

# Ex-post diagnostic classification only. Never used as an entry filter.
# Snapshot is 31-Jul-2007. These names were subsequently removed in the
# Dec-2007 rebalance or in the 2009-2010 crisis aftermath.
EARLY_REMOVED = {
    "FHN","MO","SLM",
    "BUD","BAC","CMA","FITB","KEY","PGR","RF","SNV",
    "AVY","BBT","GCI","GE","JCI","MTB","PFE","STT","USB"
}

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

def md_table(df, cols, names=None, n=10):
    q = df.head(n).copy()
    names = names or cols
    out = ["|"+"|".join(names)+"|\n",
           "|"+"|".join(["---"]*len(cols))+"|\n"]
    for _, row in q.iterrows():
        vals = []
        for c in cols:
            val = row[c]
            if isinstance(val, (float, np.floating)):
                vals.append("" if pd.isna(val) else f"{val:.2f}")
            else:
                vals.append(str(val))
        out.append("|"+"|".join(vals)+"|\n")
    return "".join(out)

def main():
    ev = pd.read_csv(EVENTS, parse_dates=["signal_date","entry_date","swing_start"])
    v = ev[ev["signal_type"]=="V0.1"].copy()
    v["future_group"] = np.where(v["ticker"].isin(EARLY_REMOVED),
                                 "REMOVED_2007_2010",
                                 "NOT_REMOVED_2007_2010")

    # 1) Future-status split
    group_rows = []
    for grp, g in v.groupby("future_group"):
        r = {"future_group":grp}
        r.update(summarize(g))
        r["sum_ret_20d_pct_points"] = g["ret_20d_pct"].sum()
        group_rows.append(r)
    dg = pd.DataFrame(group_rows)
    dg.to_csv(RESULTS/"diagnostic_future_groups.csv", index=False)

    # 2) Worst stopped trades
    stops = (v[v["stop_hit_20d"]]
             .sort_values(["mae_before_stop_pct","ret_20d_pct"],
                          ascending=[True,True]))
    stops.to_csv(RESULTS/"diagnostic_all_stops.csv", index=False)
    stops.head(25).to_csv(RESULTS/"diagnostic_worst_stops.csv", index=False)

    # 3) Clustering
    same_day = (v.groupby("signal_date")
                  .agg(signals=("ticker","size"),
                       stocks=("ticker","nunique"),
                       stop_rate=("stop_hit_20d","mean"),
                       avg_ret_20d_pct=("ret_20d_pct","mean"))
                  .reset_index())
    same_day["stop_rate_pct"] = 100*same_day.pop("stop_rate")
    same_day = same_day.sort_values(["signals","signal_date"],
                                    ascending=[False,True])
    same_day.to_csv(RESULTS/"diagnostic_clustering_by_day.csv", index=False)

    x = v.copy()
    dt = pd.to_datetime(x["signal_date"])
    iso = dt.dt.isocalendar()
    x["iso_year"] = iso.year.astype(int)
    x["iso_week"] = iso.week.astype(int)
    weekly = (x.groupby(["iso_year","iso_week"])
                .agg(signals=("ticker","size"),
                     stocks=("ticker","nunique"),
                     stop_rate=("stop_hit_20d","mean"),
                     avg_ret_20d_pct=("ret_20d_pct","mean"))
                .reset_index())
    weekly["stop_rate_pct"] = 100*weekly.pop("stop_rate")
    weekly = weekly.sort_values(["signals","iso_year","iso_week"],
                                ascending=[False,True,True])
    weekly.to_csv(RESULTS/"diagnostic_clustering_by_week.csv", index=False)

    x["month"] = dt.dt.to_period("M").astype(str)
    monthly = (x.groupby("month")
                 .agg(signals=("ticker","size"),
                      stocks=("ticker","nunique"),
                      stop_rate=("stop_hit_20d","mean"),
                      avg_ret_20d_pct=("ret_20d_pct","mean"))
                 .reset_index())
    monthly["stop_rate_pct"] = 100*monthly.pop("stop_rate")
    monthly.to_csv(RESULTS/"diagnostic_clustering_by_month.csv", index=False)

    # 4) Concentration
    by_t = (v.groupby("ticker")
              .agg(signals=("ticker","size"),
                   stops=("stop_hit_20d","sum"),
                   avg_ret_20d_pct=("ret_20d_pct","mean"),
                   sum_ret_20d_pct_points=("ret_20d_pct","sum"),
                   avg_mfe_before_stop_pct=("mfe_before_stop_pct","mean"),
                   avg_mae_before_stop_pct=("mae_before_stop_pct","mean"))
              .reset_index())
    by_t["stop_rate_pct"] = 100*by_t["stops"]/by_t["signals"]
    by_t = by_t.sort_values("sum_ret_20d_pct_points", ascending=False)
    by_t.to_csv(RESULTS/"diagnostic_concentration_by_stock.csv", index=False)

    ranking = by_t["ticker"].tolist()
    robust_rows = []
    for k in [0,1,3,5,10]:
        removed = set(ranking[:k])
        g = v[~v["ticker"].isin(removed)]
        r = {"top_positive_tickers_removed":k,
             "removed_tickers":",".join(ranking[:k])}
        r.update(summarize(g))
        r["sum_ret_20d_pct_points"] = g["ret_20d_pct"].sum()
        robust_rows.append(r)
    robust = pd.DataFrame(robust_rows)
    robust.to_csv(RESULTS/"diagnostic_remove_top_contributors.csv", index=False)

    # Ex-post comparison with the stochastic-only control, by future group
    c = ev[ev["signal_type"]=="STOCH_ONLY_CONTROL"].copy()
    c["future_group"] = np.where(c["ticker"].isin(EARLY_REMOVED),
                                 "REMOVED_2007_2010",
                                 "NOT_REMOVED_2007_2010")
    compare_rows = []
    for typ, frame in [("V0.1",v),("STOCH_ONLY_CONTROL",c)]:
        for grp,g in frame.groupby("future_group"):
            r={"signal_type":typ,"future_group":grp}
            r.update(summarize(g))
            compare_rows.append(r)
    pd.DataFrame(compare_rows).to_csv(
        RESULTS/"diagnostic_signal_vs_control_by_future_group.csv", index=False)

    report = [
        "# GFC diagnostics — V0.1\n\n",
        "This is an **ex-post diagnostic**. Future removal status is never used as a trading filter.\n\n",
        "## Snapshot caveat\n",
        "The first frozen test uses the **31-Jul-2007** published Dividend Aristocrats snapshot, "
        "not the post-Dec-2007 rebalanced list. FHN, MO and SLM therefore remain in this test.\n\n",
        "## Future-status split\n\n",
        md_table(dg,
          ["future_group","n","stocks","stop_rate_pct","hit3_before_stop_pct",
           "avg_mfe_before_stop_pct","avg_mae_before_stop_pct","avg_ret_20d_pct"],
          ["Group","N","Stocks","Stop %","+3% before stop","Avg MFE %","Avg MAE %","Avg 20d %"], 10),
        "\n## Worst stopped trades\n\n",
        md_table(stops,
          ["ticker","signal_date","regime","swing_distance_pct",
           "mae_before_stop_pct","ret_20d_pct","days_to_stop"],
          ["Ticker","Signal date","Regime","Swing dist %","MAE before stop %","20d %","Days to stop"], 15),
        "\n## Largest same-day clusters\n\n",
        md_table(same_day,
          ["signal_date","signals","stocks","stop_rate_pct","avg_ret_20d_pct"],
          ["Date","Signals","Stocks","Stop %","Avg 20d %"], 10),
        "\n## Largest weekly clusters\n\n",
        md_table(weekly,
          ["iso_year","iso_week","signals","stocks","stop_rate_pct","avg_ret_20d_pct"],
          ["Year","Week","Signals","Stocks","Stop %","Avg 20d %"], 10),
        "\n## Largest positive stock contributors\n\n",
        md_table(by_t,
          ["ticker","signals","stop_rate_pct","avg_ret_20d_pct","sum_ret_20d_pct_points"],
          ["Ticker","Signals","Stop %","Avg 20d %","Sum 20d pp"], 10),
        "\n## Largest negative stock contributors\n\n",
        md_table(by_t.tail(10).sort_values("sum_ret_20d_pct_points"),
          ["ticker","signals","stop_rate_pct","avg_ret_20d_pct","sum_ret_20d_pct_points"],
          ["Ticker","Signals","Stop %","Avg 20d %","Sum 20d pp"], 10),
        "\n## Robustness after deleting top positive contributors\n\n",
        md_table(robust,
          ["top_positive_tickers_removed","n","stop_rate_pct","hit3_before_stop_pct",
           "avg_ret_10d_pct","avg_ret_20d_pct"],
          ["Top removed","N","Stop %","+3% before stop","Avg 10d %","Avg 20d %"], 10),
    ]

    (RESULTS/"DIAGNOSTICS_GFC.md").write_text("".join(report),encoding="utf-8")
    print("".join(report))

if __name__ == "__main__":
    main()
