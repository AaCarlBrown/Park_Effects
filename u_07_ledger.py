r"""Stage 7: career park-adjustment ledgers, mesoball vs conventional.

Batters (3,000+ PA) and pitchers (3,000+ BF):
  mesoball adj    : PA-weighted mean shrunk park effect over every plate
                    appearance the player actually took (wOBA points)
  conventional adj: the as-practiced method - his TEAM's own home/road run
                    factor (centered 5-year mean, regressed), halved,
                    converted to wOBA points, weighted by season PA
  ex-ante variant : mesoball adj recomputed from the real-time series
  hand-aware      : mesoball adj plus the park's career handedness offset
                    for the side the batter actually used

Writes:
    output\career_ledger_batters.csv / _pitchers.csv
    output\ledger_summary.txt   (gates + Table 8/9 fills)

Run:  py u_07_ledger.py             (from C:\SABR_Mesoball\GitHub)
Expect several minutes and a few GB of RAM.
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


def names_map():
    try:
        bio = pd.read_csv(OUT_DIR.parent / "data" / "biofile0.csv",
                          dtype=str, keep_default_na=False)
        bio.columns = [c.strip().lower() for c in bio.columns]
        idc = next(c for c in bio.columns if c in ("id", "playerid", "retroid"))
        nc = next(c for c in bio.columns if "fullname" in c)
        return dict(zip(bio[idc], bio[nc]))
    except Exception as e:
        print(f"(biofile names unavailable: {e})")
        return {}


def centered_team_factor(tf_team, season, window=2, reg=2):
    h = tf_team[(tf_team["season"] >= season - window)
                & (tf_team["season"] <= season + window)]
    if not len(h):
        return np.nan
    n = len(h)
    return (h["tf"].mean() * n + 1.0 * reg) / (n + reg)


def main():
    exp = json.loads((EXPECTATIONS_DIR / "expectations_07.json").read_text())
    log("INFO", f"stage 7 ledger {datetime.now().isoformat(timespec='seconds')}")
    nm = names_map()

    cols = ["bat_id", "pit_id", "park_id", "season", "bat_team", "pit_team",
            "bat_side", "bat_age", "pit_age", "woba_value"]
    d = pd.read_parquet(OUT_DIR / "pa_derived.parquet", columns=cols)

    shr = pd.read_csv(OUT_DIR / "park_by_season_shrunk.csv")[
        ["park_id", "season", "effect_shrunk"]]
    rt = pd.read_csv(OUT_DIR / "park_by_season_realtime.csv")[
        ["park_id", "season", "effect"]].rename(columns={"effect": "eff_rt"})
    d = d.merge(shr, on=["park_id", "season"], how="left")
    d = d.merge(rt, on=["park_id", "season"], how="left")
    # short-tenure venues have no shrunk series; fall back to the raw
    # hindsight effect so every plate appearance counts in the ledger
    hraw = pd.read_csv(OUT_DIR / "park_by_season_hindsight.csv")[
        ["park_id", "season", "effect"]].rename(columns={"effect": "eff_raw"})
    d = d.merge(hraw, on=["park_id", "season"], how="left")
    cov = d["effect_shrunk"].notna().mean()
    d["effect_shrunk"] = d["effect_shrunk"].fillna(d["eff_raw"])
    d["eff_rt"] = d["eff_rt"].fillna(d["eff_raw"])
    print(f"shrunk-series coverage {cov:.2%}; fallback raw for the rest "
          f"(total coverage {d['effect_shrunk'].notna().mean():.2%})")

    # conventional: team-keyed centered factor -> wOBA points via the
    # calibration slope (raw park wOBA differential per unit (tf-1))
    tf = pd.read_csv(OUT_DIR / "team_season_factors.csv")
    hind = pd.read_csv(OUT_DIR / "park_by_season_hindsight.csv")
    raw = (d[d["woba_value"].notna()]
           .groupby(["park_id", "season"])["woba_value"]
           .agg(["mean", "size"]).reset_index())
    raw = raw.merge(hind[["park_id", "season", "component"]],
                    on=["park_id", "season"], how="inner")
    cm = (raw.groupby(["season", "component"])
          .apply(lambda g: np.average(g["mean"], weights=g["size"]),
                 include_groups=False).rename("cmean").reset_index())
    raw = raw.merge(cm, on=["season", "component"])
    raw["diff"] = raw["mean"] - raw["cmean"]
    cal = raw.merge(tf.groupby(["park_id", "season"])["tf"].mean().reset_index(),
                    on=["park_id", "season"]).dropna()
    slope = (np.sum(cal["diff"] * (cal["tf"] - 1))
             / np.sum((cal["tf"] - 1) ** 2))
    log("INFO", f"factor->wOBA slope {slope:.4f}")
    tf_conv = {}
    for team, g in tf.groupby("team"):
        g = g.sort_values("season")
        for s in g["season"]:
            tf_conv[(team, s)] = centered_team_factor(g, s)
    conv_pts = {k: (v - 1) * slope for k, v in tf_conv.items()}

    # park x side career handedness offsets (residual machinery)
    mu = pd.read_csv(OUT_DIR / "mu_S.csv").set_index("season")["mu"]
    abv = pd.read_csv(OUT_DIR / "alpha_bat.csv").set_index("age")["alpha"]
    apv = pd.read_csv(OUT_DIR / "alpha_pit.csv").set_index("age")["alpha"]
    tb = pd.read_parquet(OUT_DIR / "tau_bat.parquet").set_index("bat_id")["tau"]
    tp = pd.read_parquet(OUT_DIR / "tau_pit.parquet").set_index("pit_id")["tau"]
    parts = (d["season"].map(mu) + d["bat_id"].map(tb) + d["pit_id"].map(tp)
             + d["bat_age"].map(abv) + d["pit_age"].map(apv))
    okr = parts.notna() & d["woba_value"].notna()
    dr = d[okr].copy()
    dr["resid"] = d.loc[okr, "woba_value"] - parts[okr]
    dr["resid"] -= dr["resid"].mean()
    dr = dr.merge(hind[["park_id", "season", "component"]],
                  on=["park_id", "season"], how="inner")
    dh = dr[dr["bat_side"].isin(["L", "R"])]
    lg = dh.groupby(["season", "component", "bat_side"])["resid"].transform("mean")
    dh = dh.assign(dev=dh["resid"] - lg)
    side_eff = (dh.groupby(["park_id", "bat_side"])["dev"].mean()
                .rename("eff").reset_index())
    all_eff = (dh.groupby("park_id")["dev"].mean().rename("eff_all"))
    side_eff = side_eff.merge(all_eff, on="park_id")
    side_eff["offset"] = side_eff["eff"] - side_eff["eff_all"]
    off = side_eff.set_index(["park_id", "bat_side"])["offset"]

    def ledger(id_col, team_col, min_pa):
        # eligibility on ALL career PA; effects averaged where available
        pa_tot = d.groupby(id_col).size()
        keep = pa_tot[pa_tot >= min_pa].index
        g = d.dropna(subset=["effect_shrunk"])
        gg = g[g[id_col].isin(keep)]
        meso = gg.groupby(id_col)["effect_shrunk"].mean() * 1000
        rt_ok = gg.dropna(subset=["eff_rt"])
        meso_rt = rt_ok.groupby(id_col)["eff_rt"].mean() * 1000
        # conventional: team-keyed factor per player-season
        ps = (gg.groupby([id_col, "season", team_col]).size()
              .rename("pa").reset_index())
        ps["cv"] = [conv_pts.get((t, s), np.nan)
                    for t, s in zip(ps[team_col], ps["season"])]
        ps = ps.dropna(subset=["cv"])
        conv = (ps.groupby(id_col)
                .apply(lambda x: np.average(x["cv"], weights=x["pa"]) / 2,
                       include_groups=False) * 1000)
        out = pd.DataFrame({"meso": meso, "conv": conv,
                            "meso_rt": meso_rt, "pa": pa_tot[keep]})
        n_el = len(keep)
        n_no_meso = out["meso"].isna().sum()
        n_no_conv = out["conv"].isna().sum()
        log("INFO", f"{id_col}: eligible {n_el:,}; missing meso "
                    f"{n_no_meso:,}; missing conventional {n_no_conv:,}")
        out = out.dropna(subset=["meso", "conv"])
        out["diff"] = out["meso"] - out["conv"]
        out["name"] = [nm.get(i, i) for i in out.index]
        return out

    bat = ledger("bat_id", "bat_team", exp["min_pa"])
    # hand-aware for batters
    gb = d.dropna(subset=["effect_shrunk"])
    gb = gb[gb["bat_id"].isin(bat.index) & gb["bat_side"].isin(["L", "R"])]
    key = pd.MultiIndex.from_arrays([gb["park_id"], gb["bat_side"]])
    gb = gb.assign(o=off.reindex(key).values)
    hand_delta = gb.dropna(subset=["o"]).groupby("bat_id")["o"].mean() * 1000
    bat["hand_delta"] = hand_delta
    bat["meso_hand"] = bat["meso"] + bat["hand_delta"].fillna(0)
    bat.to_csv(OUT_DIR / "career_ledger_batters.csv")
    pit = ledger("pit_id", "pit_team", exp["min_pa"])
    pit.to_csv(OUT_DIR / "career_ledger_pitchers.csv")

    # ---- gates and fills ----
    lo, hi = exp["n_batters_range"]
    log("PASS" if lo <= len(bat) <= hi else "FAIL",
        f"batters with {exp['min_pa']}+ PA and both adjustments: "
        f"{len(bat):,} (paper says 1,999)")
    med = bat["diff"].abs().median()
    p90 = bat["diff"].abs().quantile(0.90)
    pct5 = (bat["diff"].abs() > 5).mean() * 100
    lo, hi = exp["median_gap_range"]
    log("PASS" if lo <= med <= hi else "FAIL",
        f"median |meso - conv| = {med:.2f} pts (paper 1.2)")
    lo, hi = exp["p90_gap_range"]
    log("PASS" if lo <= p90 <= hi else "FAIL",
        f"90th pct = {p90:.2f} pts (paper 3.4)")
    lo, hi = exp["pct_over5_range"]
    log("PASS" if lo <= pct5 <= hi else "FAIL",
        f"{pct5:.1f}% of careers > 5 pts apart (paper ~3%)")
    exq = (bat["meso"] - bat["meso_rt"]).abs().median()
    lo, hi = exp["exante_median_range"]
    log("PASS" if lo <= exq <= hi else "FAIL",
        f"median hindsight-vs-ex-ante = {exq:.2f} pts (paper 0.7)")
    hmove = (bat["hand_delta"].abs() > 2).mean() * 100
    lo, hi = exp["hand_move_pct_range"]
    log("PASS" if lo <= hmove <= hi else "FAIL",
        f"{hmove:.1f}% of careers move > 2 pts under hand-aware "
        f"adjustment (paper ~8%)")

    t8 = bat.reindex(bat["diff"].abs().sort_values(ascending=False).index)
    log("INFO", "Table 8 fill - top 10 batter gaps:\n" +
        t8.head(10)[["name", "meso", "conv", "diff", "pa"]]
        .round(1).to_string())
    t9 = pit.reindex(pit["diff"].abs().sort_values(ascending=False).index)
    log("INFO", "Table 9 fill - top 10 pitcher gaps:\n" +
        t9.head(10)[["name", "meso", "conv", "diff", "pa"]]
        .round(1).to_string())
    moore_ids = [i for i in bat.index
                 if "terry" in nm.get(i, "").lower()
                 and "moore" in nm.get(i, "").lower()]
    moore = moore_ids[0] if moore_ids else "moort101"
    for label, pid in [("Musial", "musis101"), ("Yastrzemski", "yastc101"),
                       ("Terry Moore", moore), ("Walker", "walkl001")]:
        if pid in bat.index:
            r = bat.loc[pid]
            log("INFO", f"{label}: meso {r['meso']:+.1f}, conv "
                        f"{r['conv']:+.1f}, hand delta "
                        f"{r['hand_delta']:+.1f}" if pd.notna(r["hand_delta"])
                        else f"{label}: meso {r['meso']:+.1f}")

    # top-100 ranking comparison
    okw = d["woba_value"].notna() & d["bat_id"].isin(bat.index)
    raww = d.loc[okw].groupby("bat_id")["woba_value"].mean() * 1000
    top100 = raww.sort_values(ascending=False).head(100).index
    r_m = (raww - bat["meso"]).loc[top100].rank(ascending=False)
    r_c = (raww - bat["conv"]).loc[top100].rank(ascending=False)
    moves = (r_m - r_c).abs()
    lo, hi = exp["top100_moves5_range"]
    log("PASS" if lo <= (moves >= 5).sum() <= hi else "FAIL",
        f"top-100: {(moves >= 5).sum()} players 5+ spots apart between "
        f"systems (paper 27), max {moves.max():.0f} (paper 20), median "
        f"{moves.median():.0f} (paper 2)")

    fails = sum(1 for l in REPORT if l.startswith("FAIL"))
    warns = sum(1 for l in REPORT if l.startswith("WARN"))
    log("INFO", f"{'GATE: PASSED' if fails == 0 else 'GATE: BLOCKED'} - "
                f"{fails} FAIL, {warns} WARN")
    (OUT_DIR / "ledger_summary.txt").write_text("\n".join(REPORT) + "\n",
                                                encoding="utf-8")


if __name__ == "__main__":
    main()
