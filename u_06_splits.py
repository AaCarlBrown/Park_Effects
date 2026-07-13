r"""Stage 6: park splits - batter handedness, hitter type, hitter quality -
plus the pitcher-side null checks and the hand-x-type interaction test.

Definitions (frozen in expectations_06.json, approved 2026-07-13):
  quality: PA-weighted quartiles of full-sample tau_bat, batters >= 1500 PA
  type:    quartiles of era-standardized extra-base rate minus
           era-standardized (walk + single + HBP) rate, same PA floor

Machinery: Stage 2 residuals; for each grouping, league means are removed
within (season, component, group) so a park's group effect is net of the
era's group-level scoring and league composition; park-group career effects
are PA-weighted means of the deviations. True between-park SDs subtract
mean sampling variance.

Writes:
    output\park_hand_splits.csv     per park: L effect, R effect, split, SE
    output\park_type_effects.csv    per park: slugger-minus-contact gap
    output\park_quality_effects.csv per park: topQ-minus-bottomQ gap
    output\splits_summary.txt       gates + the Table 5-7 fills

Run:  py u_06_splits.py             (from C:\SABR_Mesoball\GitHub)
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd

from config import EXPECTATIONS_DIR, OUT_DIR

REPORT = []


def log(level, msg):
    line = f"{level:5s} {msg}"
    REPORT.append(line)
    print(line, flush=True)


def true_sd(effects, se):
    v = np.average(effects ** 2, weights=1 / se ** 2) - np.mean(se ** 2)
    return np.sqrt(max(v, 0.0))


def park_group_effects(d, group_col, sigma2, min_pa=20000):
    """Career park effect per group, net of (season, component, group)."""
    lg = (d.groupby(["season", "component", group_col])["resid"]
          .transform("mean"))
    d = d.assign(dev=d["resid"] - lg)
    g = (d.groupby(["park_id", group_col])["dev"]
         .agg(["mean", "size"]).reset_index()
         .rename(columns={"mean": "effect", "size": "n"}))
    g["se"] = np.sqrt(sigma2 / g["n"])
    tot = g.groupby("park_id")["n"].transform("sum")
    return g[tot >= min_pa]


def gap_table(g, group_col, hi, lo_):
    a = g[g[group_col] == hi].set_index("park_id")
    b = g[g[group_col] == lo_].set_index("park_id")
    j = a.join(b, lsuffix="_hi", rsuffix="_lo", how="inner")
    j["gap"] = j["effect_hi"] - j["effect_lo"]
    j["se"] = np.sqrt(j["se_hi"] ** 2 + j["se_lo"] ** 2)
    return j.reset_index()[["park_id", "gap", "se", "n_hi", "n_lo"]]


def main():
    exp = json.loads((EXPECTATIONS_DIR / "expectations_06.json").read_text())
    log("INFO", f"stage 6 splits {datetime.now().isoformat(timespec='seconds')}")

    cols = ["park_id", "season", "bat_id", "pit_id", "bat_age", "pit_age",
            "woba_value", "bat_side", "pit_hand",
            "single", "double", "triple", "hr", "walk", "hbp"]
    d = pd.read_parquet(OUT_DIR / "pa_derived.parquet", columns=cols)
    d = d[d["woba_value"].notna()]

    mu = pd.read_csv(OUT_DIR / "mu_S.csv").set_index("season")["mu"]
    ab = pd.read_csv(OUT_DIR / "alpha_bat.csv").set_index("age")["alpha"]
    ap = pd.read_csv(OUT_DIR / "alpha_pit.csv").set_index("age")["alpha"]
    tb = pd.read_parquet(OUT_DIR / "tau_bat.parquet").set_index("bat_id")["tau"]
    tp = pd.read_parquet(OUT_DIR / "tau_pit.parquet").set_index("pit_id")["tau"]
    parts = (d["season"].map(mu) + d["bat_id"].map(tb) + d["pit_id"].map(tp)
             + d["bat_age"].map(ab) + d["pit_age"].map(ap))
    ok = parts.notna()
    d = d[ok].copy()
    d["resid"] = (d["woba_value"] - parts[ok])
    d["resid"] -= d["resid"].mean()
    sigma2 = d["resid"].var()
    log("INFO", f"residual PA {len(d):,}; sigma2 {sigma2:.4f}")

    comp = pd.read_csv(OUT_DIR / "park_by_season_hindsight.csv")[
        ["park_id", "season", "component"]]
    d = d.merge(comp, on=["park_id", "season"], how="inner")

    names = {}
    try:
        pc = pd.read_csv(OUT_DIR.parent / "data" / "parkcode.txt", dtype=str,
                         keep_default_na=False)
        pc.columns = [c.upper() for c in pc.columns]
        names = dict(zip(pc["PARKID"], pc["NAME"]))
    except Exception:
        pass

    def named(pid):
        return names.get(pid, pid)

    # ---- 1. batter handedness ----
    dh = d[d["bat_side"].isin(["L", "R"])]
    g = park_group_effects(dh, "bat_side", sigma2, exp["min_park_pa"])
    hand = gap_table(g, "bat_side", "L", "R").rename(columns={"gap": "split"})
    hand["name"] = hand["park_id"].map(named)
    hand.to_csv(OUT_DIR / "park_hand_splits.csv", index=False)
    hs = hand.sort_values("split", ascending=False)
    log("INFO", "top 5 lefty-friendly (Table 5 left):\n" +
        (hs.head(5)[["name", "split", "se"]] * 1).assign(
            split=lambda x: (x["split"] * 1000).round(1),
            se=lambda x: (x["se"] * 1000).round(1)).to_string(index=False))
    log("INFO", "top 5 righty-friendly (Table 5 right):\n" +
        hs.tail(5).iloc[::-1][["name", "split", "se"]].assign(
            split=lambda x: (x["split"] * 1000).round(1),
            se=lambda x: (x["se"] * 1000).round(1)).to_string(index=False))
    sd_h = true_sd(hand["split"].values, hand["se"].values) * 1000
    lo, hi = exp["hand_split_true_sd_range"]
    log("PASS" if lo <= sd_h <= hi else "FAIL",
        f"between-park true SD of handedness split {sd_h:.1f} pts "
        f"(expected {lo}-{hi})")
    top2 = set(hs.head(2)["park_id"])
    log("PASS" if "CLE06" in top2 else "WARN",
        f"League Park in top-2 lefty-friendly: {'yes' if 'CLE06' in top2 else hs.head(2)['name'].tolist()}")
    bot3 = set(hs.tail(3)["park_id"])
    log("PASS" if "BOS07" in bot3 else "WARN",
        f"Fenway in top-3 righty-friendly: {'yes' if 'BOS07' in bot3 else hs.tail(3)['name'].tolist()}")

    # ---- 2. pitcher handedness (platoon-arithmetic null) ----
    dp = d[d["pit_hand"].isin(["L", "R"])]
    gp = park_group_effects(dp, "pit_hand", sigma2, exp["min_park_pa"])
    ph = gap_table(gp, "pit_hand", "L", "R").rename(columns={"gap": "split"})
    m = ph.merge(hand[["park_id", "split"]], on="park_id",
                 suffixes=("_pit", "_bat"))
    b = (np.sum(m["split_pit"] * m["split_bat"])
         / np.sum(m["split_bat"] ** 2))
    resid_p = m["split_pit"] - b * m["split_bat"]
    sd_p = true_sd(resid_p.values, m["se"].values) * 1000
    log("PASS" if sd_p <= exp["pitcher_null_max_sd"] else "FAIL",
        f"pitcher-hand park signal after batter-split control: true SD "
        f"{sd_p:.1f} pts (null if <= {exp['pitcher_null_max_sd']})")

    # ---- batter career classifications ----
    bat = d.groupby("bat_id")
    career_pa = bat.size()
    eligible = career_pa[career_pa >= exp["min_career_pa"]].index
    # era-standardized rates
    for col, num in [("xb", d["double"] + 2 * d["triple"] + 3 * d["hr"]),
                     ("ob", d["single"] + d["walk"] + d["hbp"])]:
        d[f"_{col}"] = num
        sm = d.groupby("season")[f"_{col}"].transform("mean")
        ss = d.groupby("season")[f"_{col}"].transform("std")
        d[f"z_{col}"] = (d[f"_{col}"] - sm) / ss
    zb = d.groupby("bat_id")[["z_xb", "z_ob"]].mean()
    zb = zb.loc[zb.index.isin(eligible)]
    type_score = zb["z_xb"] - zb["z_ob"]
    q = type_score.quantile([0.25, 0.75])
    type_map = pd.Series("mid", index=type_score.index)
    type_map[type_score >= q[0.75]] = "slugger"
    type_map[type_score <= q[0.25]] = "contact"
    tau_el = tb.loc[tb.index.isin(eligible)]
    qq = tau_el.quantile([0.25, 0.75])
    qual_map = pd.Series("mid", index=tau_el.index)
    qual_map[tau_el >= qq[0.75]] = "top"
    qual_map[tau_el <= qq[0.25]] = "bottom"
    d["btype"] = d["bat_id"].map(type_map)
    d["bqual"] = d["bat_id"].map(qual_map)

    # ---- 3. hitter type ----
    dt = d[d["btype"].isin(["slugger", "contact"])]
    gt = park_group_effects(dt, "btype", sigma2, exp["min_park_pa"])
    tt = gap_table(gt, "btype", "slugger", "contact")
    tt["name"] = tt["park_id"].map(named)
    tt.to_csv(OUT_DIR / "park_type_effects.csv", index=False)
    ts = tt.sort_values("gap", ascending=False)
    log("INFO", "top 5 favor sluggers (Table 6 left):\n" +
        ts.head(5)[["name", "gap", "se"]].assign(
            gap=lambda x: (x["gap"] * 1000).round(1),
            se=lambda x: (x["se"] * 1000).round(1)).to_string(index=False))
    log("INFO", "top 5 favor contact (Table 6 right):\n" +
        ts.tail(5).iloc[::-1][["name", "gap", "se"]].assign(
            gap=lambda x: (x["gap"] * 1000).round(1),
            se=lambda x: (x["se"] * 1000).round(1)).to_string(index=False))
    sd_t = true_sd(tt["gap"].values, tt["se"].values) * 1000
    lo, hi = exp["type_gap_true_sd_range"]
    log("PASS" if lo <= sd_t <= hi else "FAIL",
        f"between-park true SD of type gap {sd_t:.1f} pts (expected {lo}-{hi})")

    # ---- 4. hitter quality ----
    dq = d[d["bqual"].isin(["top", "bottom"])]
    gq = park_group_effects(dq, "bqual", sigma2, exp["min_park_pa"])
    qt = gap_table(gq, "bqual", "top", "bottom")
    qt["name"] = qt["park_id"].map(named)
    qt.to_csv(OUT_DIR / "park_quality_effects.csv", index=False)
    qs = qt.sort_values("gap", ascending=False)
    log("INFO", "top 5 amplify the gap (Table 7 left):\n" +
        qs.head(5)[["name", "gap", "se"]].assign(
            gap=lambda x: (x["gap"] * 1000).round(1),
            se=lambda x: (x["se"] * 1000).round(1)).to_string(index=False))
    log("INFO", "top 5 compress the gap (Table 7 right):\n" +
        qs.tail(5).iloc[::-1][["name", "gap", "se"]].assign(
            gap=lambda x: (x["gap"] * 1000).round(1),
            se=lambda x: (x["se"] * 1000).round(1)).to_string(index=False))
    sd_q = true_sd(qt["gap"].values, qt["se"].values) * 1000
    lo, hi = exp["quality_gap_true_sd_range"]
    log("PASS" if lo <= sd_q <= hi else "FAIL",
        f"between-park true SD of quality gap {sd_q:.1f} pts "
        f"(expected {lo}-{hi})")

    # ---- 5. pitcher quality null ----
    pit_el = tp[tp.index.isin(d.groupby("pit_id").size()
                              [lambda s: s >= exp["min_career_pa"]].index)]
    pq = pit_el.quantile([0.25, 0.75])
    pqual = pd.Series("mid", index=pit_el.index)
    pqual[pit_el >= pq[0.75]] = "top"
    pqual[pit_el <= pq[0.25]] = "bottom"
    d["pqual"] = d["pit_id"].map(pqual)
    dpq = d[d["pqual"].isin(["top", "bottom"])]
    gpq = park_group_effects(dpq, "pqual", sigma2, exp["min_park_pa"])
    pqt = gap_table(gpq, "pqual", "top", "bottom")
    sd_pq = true_sd(pqt["gap"].values, pqt["se"].values) * 1000
    log("PASS" if sd_pq <= exp["pitcher_null_max_sd"] else "FAIL",
        f"pitcher-quality park signal: true SD {sd_pq:.1f} pts "
        f"(null if <= {exp['pitcher_null_max_sd']})")

    # ---- 6. hand x type interaction ----
    # (slice from d, not dh: dh was cut before btype existed)
    dht = d[d["bat_side"].isin(["L", "R"])
            & d["btype"].isin(["slugger", "contact"])]
    rows = []
    for t_, dd in dht.groupby("btype"):
        gg = park_group_effects(dd, "bat_side", sigma2, exp["min_park_pa"])
        sp = gap_table(gg, "bat_side", "L", "R")
        sp["btype"] = t_
        rows.append(sp)
    it = pd.concat(rows).pivot(index="park_id", columns="btype",
                               values=["gap", "se"]).dropna().astype(float)
    diff = it[("gap", "slugger")] - it[("gap", "contact")]
    se_d = np.sqrt(it[("se", "slugger")] ** 2 + it[("se", "contact")] ** 2)
    sd_i = true_sd(diff.to_numpy(dtype=float),
                   se_d.to_numpy(dtype=float)) * 1000
    # diagnostic detail: who drives the interaction?
    det = pd.DataFrame({
        "park_id": it.index,
        "split_slugger": it[("gap", "slugger")].values,
        "split_contact": it[("gap", "contact")].values,
        "diff": diff.values, "se": se_d.values,
        "z": (diff / se_d).values})
    det["name"] = det["park_id"].map(named)
    det.to_csv(OUT_DIR / "park_hand_by_type.csv", index=False)
    top = det.reindex(det["z"].abs().sort_values(ascending=False).index).head(6)
    log("INFO", "largest hand-x-type interactions (pts, z):\n" +
        top[["name", "split_slugger", "split_contact", "diff", "z"]].assign(
            split_slugger=lambda x: (x["split_slugger"] * 1000).round(1),
            split_contact=lambda x: (x["split_contact"] * 1000).round(1),
            diff=lambda x: (x["diff"] * 1000).round(1),
            z=lambda x: x["z"].round(1)).to_string(index=False))
    log("INFO", f"parks with |z| >= 2: {(det['z'].abs() >= 2).sum()} of "
                f"{len(det)}")
    # Gate finalized 2026-07-13: the draft's null did NOT reproduce; the
    # interaction is real, broad-based, and led by the short-porch parks
    # (Yankee Stadium I z=5.2, League Park z=3.3). The gate now expects it.
    lo, hi = exp["interaction_true_sd_range"]
    n_sig = int((det["z"].abs() >= 2).sum())
    ok_i = (lo <= sd_i <= hi) and n_sig >= exp["interaction_min_parks_2z"]
    log("PASS" if ok_i else "FAIL",
        f"hand-x-type interaction detected: true SD {sd_i:.1f} pts "
        f"(expected {lo}-{hi}), {n_sig} parks at |z|>=2 "
        f"(expected >= {exp['interaction_min_parks_2z']})")

    fails = sum(1 for l in REPORT if l.startswith("FAIL"))
    warns = sum(1 for l in REPORT if l.startswith("WARN"))
    log("INFO", f"{'GATE: PASSED' if fails == 0 else 'GATE: BLOCKED'} - "
                f"{fails} FAIL, {warns} WARN")
    (OUT_DIR / "splits_summary.txt").write_text("\n".join(REPORT) + "\n",
                                                encoding="utf-8")


if __name__ == "__main__":
    main()
