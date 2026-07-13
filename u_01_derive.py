r"""Stage 1: panel derivations and cross-validation.

The panel already carries park_id, per-PA handedness, and ages, so this stage
(1) attaches the two fields it lacks -- game date and day/night -- from the
Retrosheet game logs, (2) cross-validates the panel's park_id against the
game logs and parkcode.txt (the five unverified park codes, BOS08 first),
and (3) runs field-quality checks.

Writes:
    output\pa_derived.parquet      panel + date, daynight (the Stage 2+ input)
    output\park_season_pa.csv      PA count per park-season
    output\park_crosswalk_audit.csv park_ids vs parkcode.txt and game logs
    output\derive_summary.txt      PASS / WARN / FAIL lines and the stage gate

Run:  py u_01_derive.py            (from C:\SABR_Mesoball\GitHub)

Game log fields used (1-indexed per Retrosheet spec): 1 date yyyymmdd,
2 game number (0 single, 1/2 doubleheader), 7 home team, 13 day/night,
17 park id. The panel's game_id is home + yyyymmdd + game number.
"""
import json
import sys
from datetime import datetime

import pandas as pd

from config import (EXPECTATIONS_DIR, GAMELOG_DIR, OUT_DIR, PANEL, PARKCODE)

REPORT = []


def log(level, msg):
    line = f"{level:5s} {msg}"
    REPORT.append(line)
    print(line)


def load_gamelogs():
    frames = []
    for f in sorted(GAMELOG_DIR.iterdir()):
        if not f.name.lower().endswith(".txt"):
            continue
        gl = pd.read_csv(f, header=None, usecols=[0, 1, 6, 12, 16],
                         names=["date", "gamenum", "hometeam", "daynight", "gl_park"],
                         dtype=str, keep_default_na=False)
        frames.append(gl)
    gl = pd.concat(frames, ignore_index=True)
    gl["game_id"] = gl["hometeam"] + gl["date"] + gl["gamenum"]
    gl["daynight"] = gl["daynight"].str.upper().str.strip()
    return gl[["game_id", "date", "daynight", "gl_park"]]


def main():
    exp = json.loads((EXPECTATIONS_DIR / "expectations_01.json").read_text())
    log("INFO", f"stage 1 run {datetime.now().isoformat(timespec='seconds')}")

    panel = pd.read_parquet(PANEL)
    if len(panel) == exp["panel_rows"]:
        log("PASS", f"panel rows {len(panel):,}")
    else:
        log("FAIL", f"panel rows {len(panel):,} != expected {exp['panel_rows']:,}")

    # ---- null checks on key columns ----
    for c in exp["null_free_columns"]:
        n = panel[c].isna().sum()
        if n == 0:
            log("PASS", f"no nulls in {c}")
        else:
            log("FAIL", f"{n:,} nulls in {c}")

    # woba_value is null exactly for intentional walks (wOBA excludes IBB;
    # established by diagnose_01b on 2026-07-12)
    nul = panel["woba_value"].isna()
    if (nul == (panel["iw"] == 1)).all():
        log("PASS", f"woba_value nulls ({nul.sum():,}) coincide exactly "
                    f"with intentional walks (wOBA excludes IBB)")
    else:
        bad = (nul != (panel["iw"] == 1)).sum()
        log("FAIL", f"woba_value nulls do not coincide with iw flag "
                    f"({bad:,} rows differ)")

    # ---- game log merge ----
    gl = load_gamelogs()
    log("INFO", f"game log games: {len(gl):,}")
    dupes = gl["game_id"].duplicated().sum()
    if dupes:
        log("FAIL", f"{dupes} duplicate game_ids in game logs")
    panel = panel.merge(gl, on="game_id", how="left")
    cov = panel["daynight"].notna().mean()
    if cov >= exp["gamelog_join_min_coverage"]:
        log("PASS", f"game log join coverage {cov:.4%}")
    else:
        log("FAIL", f"game log join coverage {cov:.4%} < "
                    f"{exp['gamelog_join_min_coverage']:.1%}")
    bad_dn = ~panel["daynight"].isin(exp["daynight_values"]) & panel["daynight"].notna()
    if bad_dn.any():
        vals = panel.loc[bad_dn, "daynight"].value_counts().head(5).to_dict()
        log("WARN", f"unexpected day/night values: {vals}")

    # ---- park_id cross-validation: panel vs game logs ----
    both = panel["gl_park"].notna() & (panel["park_id"] != "")
    mism = (panel.loc[both, "park_id"] != panel.loc[both, "gl_park"]).mean()
    if mism <= exp["park_id_vs_gamelog_park_mismatch_max"]:
        log("PASS", f"panel park_id vs game log park mismatch rate {mism:.5%}")
    else:
        log("FAIL", f"panel park_id vs game log park mismatch rate {mism:.5%}")
        top = (panel.loc[both & (panel["park_id"] != panel["gl_park"]),
                         ["park_id", "gl_park", "season"]]
               .value_counts().head(10))
        log("INFO", f"top mismatches:\n{top}")

    # ---- parkcode.txt audit ----
    audit_rows = []
    if PARKCODE.exists():
        pc = pd.read_csv(PARKCODE, dtype=str, keep_default_na=False)
        pc.columns = [c.strip().upper() for c in pc.columns]
        known = set(pc["PARKID"])
        used = panel.groupby("park_id").agg(
            pa=("pa", "size"), first=("season", "min"), last=("season", "max"))
        for pid, row in used.iterrows():
            in_pc = pid in known
            name = (pc.loc[pc["PARKID"] == pid, "NAME"].iloc[0] if in_pc else "")
            audit_rows.append({"park_id": pid, "in_parkcode": in_pc,
                               "name": name, "pa": row["pa"],
                               "first_season": row["first"],
                               "last_season": row["last"]})
        unknown = [a["park_id"] for a in audit_rows if not a["in_parkcode"]]
        if unknown:
            log("WARN", f"park_ids not in parkcode.txt: {unknown}")
        else:
            log("PASS", "every panel park_id appears in parkcode.txt")
        bos08 = [a for a in audit_rows if a["park_id"] == "BOS08"]
        if bos08:
            a = bos08[0]
            log("INFO", f"BOS08: {a['name']}, seasons {a['first_season']}-"
                        f"{a['last_season']}, {a['pa']:,} PA "
                        f"(verify: Braves Field, 1915-1952)")
        pd.DataFrame(audit_rows).to_csv(OUT_DIR / "park_crosswalk_audit.csv",
                                        index=False)
    else:
        log("WARN", "parkcode.txt not present; crosswalk audit skipped")

    # ---- park-season PA counts ----
    psp = (panel.groupby(["park_id", "season"]).size()
           .rename("pa").reset_index())
    psp.to_csv(OUT_DIR / "park_season_pa.csv", index=False)
    lo, hi = exp["park_season_pa_mean_range"]
    m = psp["pa"].mean()
    if lo <= m <= hi:
        log("PASS", f"mean park-season PA {m:,.0f} (expected {lo:,}-{hi:,})")
    else:
        log("WARN", f"mean park-season PA {m:,.0f} outside {lo:,}-{hi:,}")

    # ---- Wrigley day games since 1950 (the wind sample) ----
    wr = panel[(panel["park_id"] == exp["wrigley_park_id"])
               & (panel["season"] >= 1950) & (panel["daynight"] == "D")]
    n_games = wr["game_id"].nunique()
    target, tol = (exp["wrigley_day_games_since_1950_approx"],
                   exp["wrigley_day_games_tolerance"])
    if abs(n_games - target) <= tol:
        log("PASS", f"Wrigley day games since 1950: {n_games:,} "
                    f"(paper: {target:,})")
    else:
        log("WARN", f"Wrigley day games since 1950: {n_games:,} vs paper "
                    f"{target:,} (diff {n_games - target:+,}) - reconcile "
                    f"before Stage 5")

    # ---- derive per-PA batter side ----
    # Diagnosis (diagnose_01b): bat_hand is the roster hand, so switch
    # hitters carry "B" on every PA. The per-PA side is therefore derived:
    # non-switch hitters bat their listed side; switch hitters are assigned
    # the side opposite the pitcher's throwing hand (the near-universal
    # choice; rare same-side exceptions are not recoverable from this panel).
    side = panel["bat_hand"].where(panel["bat_hand"].isin(["L", "R"]))
    need = side.isna() & panel["pit_hand"].isin(["L", "R"])
    side = side.mask(need & (panel["pit_hand"] == "L"), "R")
    side = side.mask(need & (panel["pit_hand"] == "R"), "L")
    panel["bat_side"] = side
    cov_s = side.notna().mean()
    if cov_s >= exp["bat_side_min_coverage"]:
        log("PASS", f"bat_side derived for {cov_s:.4%} of PA "
                    f"({side.isna().sum():,} unresolved)")
    else:
        log("FAIL", f"bat_side coverage only {cov_s:.4%}")
    log("INFO", f"switch-hitter PA share {(panel['bat_bats'] == 'B').mean():.2%}; "
                f"their side is derived from pitcher hand")

    # ---- write outputs ----
    panel.to_parquet(OUT_DIR / "pa_derived.parquet", index=False)
    log("INFO", f"wrote pa_derived.parquet ({len(panel):,} rows, "
                f"{len(panel.columns)} cols)")

    fails = sum(1 for l in REPORT if l.startswith("FAIL"))
    warns = sum(1 for l in REPORT if l.startswith("WARN"))
    log("INFO", f"{'GATE: PASSED' if fails == 0 else 'GATE: BLOCKED'} - "
                f"{fails} FAIL, {warns} WARN")
    (OUT_DIR / "derive_summary.txt").write_text("\n".join(REPORT) + "\n",
                                                encoding="utf-8")
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
