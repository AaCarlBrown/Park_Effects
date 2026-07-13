r"""Stage 8b: do teams fit their lineups to their parks?

Pre-registered by the author 2026-07-13 (expectations_08b.json). Reduced
form: regress the batting skill a team fields at a position (relative to
season x position norms) on the park's measured positional demand (Stage 8
compositional putout-share effects), controlling for the park's overall
offensive effect. Cluster-robust by park. Home lineups primary; road
lineups as the registered informational auxiliary.

Writes output\park_alloc_results.csv and output\alloc_summary.txt.
Run:  py u_08b_park_alloc.py        (from C:\SABR_Mesoball\GitHub)
One streaming pass over plays.csv.
"""
import json
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from config import EXPECTATIONS_DIR, OUT_DIR, PLAYS_CSV

REPORT = []
CHUNK = 2_000_000
GROUPS = {"corners (LF/RF)": [7, 9], "CF": [8], "1B": [3],
          "middle infield": [4, 6], "3B": [5], "C": [2]}


def log(level, msg):
    line = f"{level:5s} {msg}"
    REPORT.append(line)
    print(line, flush=True)


def cluster_ols(y, X, w, clusters):
    W = np.diag(w)
    XtWX = X.T @ (X * w[:, None])
    XtWy = X.T @ (w * y)
    beta = np.linalg.solve(XtWX, XtWy)
    resid = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(clusters):
        m = clusters == c
        s = (X[m] * (w[m] * resid[m])[:, None]).sum(axis=0)
        meat += np.outer(s, s)
    inv = np.linalg.inv(XtWX)
    V = inv @ meat @ inv
    return beta, np.sqrt(np.diag(V))


def main():
    exp = json.loads((EXPECTATIONS_DIR / "expectations_08b.json").read_text())
    log("INFO", f"stage 8b {datetime.now().isoformat(timespec='seconds')}")

    # team-season -> home park (modal, from Stage 4 factors file)
    tf = pd.read_csv(OUT_DIR / "team_season_factors.csv")
    team_park = tf.set_index(["team", "season"])["park_id"].to_dict()

    # one pass: PA per (season, team, pos, batter) split home/road
    acc = defaultdict(float)
    for ch in pd.read_csv(PLAYS_CSV,
                          usecols=["date", "batteam", "site", "bat_f",
                                   "batter", "pa"],
                          chunksize=CHUNK, dtype=str, keep_default_na=False):
        ch = ch[(ch["pa"] == "1")
                & ch["bat_f"].isin([str(p) for p in range(2, 10)])]
        ch["season"] = ch["date"].str[:4].astype(int)
        hp = pd.Series([team_park.get((t, s), "")
                        for t, s in zip(ch["batteam"], ch["season"])],
                       index=ch.index)
        ch["home"] = (ch["site"] == hp).astype(int)
        grp = ch.groupby(["season", "batteam", "bat_f", "batter", "home"])
        for k, v in grp.size().items():
            acc[k] += v
    rows = [{"season": k[0], "team": k[1], "pos": int(k[2]),
             "bat_id": k[3], "home": k[4], "pa": v} for k, v in acc.items()]
    d = pd.DataFrame(rows)
    log("INFO", f"player-position cells: {len(d):,}")

    tb = pd.read_parquet(OUT_DIR / "tau_bat.parquet").set_index("bat_id")["tau"]
    d["tau"] = d["bat_id"].map(tb)
    d = d.dropna(subset=["tau"])
    # season x position demeaning (PA-weighted, home+road combined)
    lg = (d.groupby(["season", "pos"])
          .apply(lambda g: np.average(g["tau"], weights=g["pa"]),
                 include_groups=False).rename("lg").reset_index())
    d = d.merge(lg, on=["season", "pos"])
    d["dev"] = d["tau"] - d["lg"]

    # demand: Stage 8 putout-share effects, standardized within position
    fe = pd.read_csv(OUT_DIR / "park_fielding_effects.csv")
    fe = fe[fe["pos"].between(2, 9)][["park_id", "pos", "conv_eff"]]
    fe["demand_sd"] = fe.groupby("pos")["conv_eff"].transform(
        lambda s: (s - s.mean()) / s.std())
    # control: park overall career effect
    h = pd.read_csv(OUT_DIR / "park_by_season_hindsight.csv")
    overall = (h.groupby("park_id")
               .apply(lambda g: np.average(g["effect"], weights=g["n_pa"]),
                      include_groups=False).rename("park_overall").reset_index())

    def build(home_flag):
        s = d[d["home"] == home_flag]
        cell = (s.groupby(["season", "team", "pos"])
                .apply(lambda g: pd.Series(
                    {"dev": np.average(g["dev"], weights=g["pa"]),
                     "pa": g["pa"].sum()}), include_groups=False)
                .reset_index())
        cell["park_id"] = [team_park.get((t, s_), "")
                           for t, s_ in zip(cell["team"], cell["season"])]
        cell = cell.merge(fe, on=["park_id", "pos"], how="inner")
        cell = cell.merge(overall, on="park_id", how="inner")
        return cell

    res = []
    for label, flag in [("home", 1), ("road", 0)]:
        cell = build(flag)
        for gname, poss in [("ALL", list(range(2, 10)))] + list(GROUPS.items()):
            s = cell[cell["pos"].isin(poss)]
            if len(s) < 50:
                continue
            X = np.column_stack([np.ones(len(s)), s["demand_sd"],
                                 s["park_overall"] * 100])
            beta, se = cluster_ols(s["dev"].values * 1000, X,
                                   s["pa"].values.astype(float),
                                   s["park_id"].values)
            res.append({"lineups": label, "group": gname,
                        "slope_pts_per_sd": round(beta[1], 2),
                        "se": round(se[1], 2),
                        "z": round(beta[1] / se[1], 1),
                        "n_cells": len(s)})
    r = pd.DataFrame(res)
    r.to_csv(OUT_DIR / "park_alloc_results.csv", index=False)
    log("INFO", "results (slope = wOBA-skill pts per SD of park demand):\n"
        + r.to_string(index=False))

    home_all = r[(r["lineups"] == "home") & (r["group"] == "ALL")].iloc[0]
    z = home_all["z"]
    if z <= -exp["slope_detection_se"]:
        lo, hi = exp["expected_slope_range_pts_per_sd"]
        ok = lo <= home_all["slope_pts_per_sd"] <= hi
        log("PASS" if ok else "WARN",
            f"DETECTION: overall home slope {home_all['slope_pts_per_sd']} "
            f"pts/SD at z = {z} (registered range [{lo}, {hi}])")
    elif z >= exp["slope_detection_se"]:
        log("FAIL", f"overall home slope POSITIVE at z = {z}: registered as "
                    f"'confound control failed; report unresolved'")
    else:
        log("PASS", f"NULL at registered power: overall home slope "
                    f"{home_all['slope_pts_per_sd']} pts/SD, z = {z} "
                    f"(the section's 'effect should be small' framing absorbs this)")
    corners = r[(r["lineups"] == "home") & (r["group"] == "corners (LF/RF)")]
    if len(corners):
        log("INFO", f"registered signal locus (corners): slope "
                    f"{corners['slope_pts_per_sd'].iloc[0]} at z = "
                    f"{corners['z'].iloc[0]}")

    fails = sum(1 for l in REPORT if l.startswith("FAIL"))
    warns = sum(1 for l in REPORT if l.startswith("WARN"))
    log("INFO", f"{'GATE: PASSED' if fails == 0 else 'GATE: BLOCKED'} - "
                f"{fails} FAIL, {warns} WARN")
    (OUT_DIR / "alloc_summary.txt").write_text("\n".join(REPORT) + "\n",
                                               encoding="utf-8")


if __name__ == "__main__":
    main()
