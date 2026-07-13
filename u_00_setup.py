r"""Stage 0: input verification for the SABR Mesoball park-effects rerun.

Checks that every input the pipeline needs is present in data\, verifies the
plate-appearance panel against pre-registered expectations, and writes:

    output\data_manifest.csv     one row per input file (size, mtime, sha256)
    output\season_coverage.csv   PA count per season with outlier flags
    output\setup_report.txt      PASS / WARN / FAIL lines and the stage gate

Run:  py u_00_setup.py            (from C:\SABR_Mesoball\GitHub)
      py u_00_setup.py --no-hash  (skip sha256 of the parquet; faster)

Exit code 0 = gate passed (no FAIL lines). Warnings do not block the gate but
must be reviewed before Stage 1.
"""
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from config import (BIOFILE, DATA_DIR, EXPECTATIONS_DIR, GAMELOG_DIR,
                    GITHUB_DIR, OUT_DIR, PANEL, PARKCODE, PITCHING, TEAMS,
                    WIND_DIR, WOBA_WEIGHTS)

REPORT = []


def log(level, msg):
    line = f"{level:5s} {msg}"
    REPORT.append(line)
    print(line)


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_row(path, do_hash):
    p = Path(path)
    if not p.exists():
        return {"file": p.name, "path": str(p), "exists": False,
                "bytes": None, "mtime": None, "sha256": None}
    st = p.stat()
    return {"file": p.name, "path": str(p), "exists": True,
            "bytes": st.st_size,
            "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            "sha256": sha256(p) if do_hash else "skipped"}


def check_gamelogs(exp):
    y0, y1 = exp["gamelog_year_range"]
    if not GAMELOG_DIR.exists():
        log("FAIL", f"game log directory missing: {GAMELOG_DIR}")
        return []
    years = set()
    for f in GAMELOG_DIR.iterdir():
        m = re.match(r"(?i)gl(\d{4})\.txt$", f.name)
        if m:
            years.add(int(m.group(1)))
    missing = [y for y in range(y0, y1 + 1) if y not in years]
    if missing:
        log("FAIL", f"game logs missing for {len(missing)} season(s): "
                    f"{missing[:10]}{'...' if len(missing) > 10 else ''}")
    else:
        log("PASS", f"game logs present for all seasons {y0}-{y1} ({len(years)} files)")
    extra = sorted(y for y in years if y < y0 or y > y1)
    if extra:
        log("WARN", f"game logs outside expected range (harmless): {extra}")
    return sorted(years)


def check_wind(exp):
    if not WIND_DIR.exists():
        log("FAIL", f"wind directory missing: {WIND_DIR}")
        return
    names = [f.name for f in WIND_DIR.iterdir()]
    for label, spec in exp["wind_stations"].items():
        st = spec["station"]
        wban = st.split("-")[1]
        y0, y1 = spec["year_range"]
        found = set()
        for n in names:
            # match by WBAN: pre-1973 US station-years often file under
            # USAF 999999 instead of the modern USAF id
            m = re.match(r"\d{6}-" + re.escape(wban) + r"-(\d{4})(\.gz)?$", n)
            if m:
                found.add(int(m.group(1)))
        missing = [y for y in range(y0, y1 + 1) if y not in found]
        if missing:
            log("FAIL", f"wind {label} ({st}) missing {len(missing)} year(s): "
                        f"{missing[:10]}{'...' if len(missing) > 10 else ''}")
        else:
            log("PASS", f"wind {label} ({st}) complete {y0}-{y1}")
        # coverage beyond the expected split is worth knowing about (Stage 5
        # confirms the true usable span)
        extra = sorted(y for y in found if y < y0 or y > y1)
        if extra:
            log("WARN", f"wind {label} also has years outside expected split: "
                        f"{extra[0]}-{extra[-1]} ({len(extra)} files)")


def check_woba_weights(exp):
    if not WOBA_WEIGHTS.exists():
        log("FAIL", f"missing {WOBA_WEIGHTS.name}")
        return
    w = pd.read_csv(WOBA_WEIGHTS)
    if "Season" not in w.columns:
        log("FAIL", f"{WOBA_WEIGHTS.name}: no 'Season' column "
                    f"(columns: {list(w.columns)[:8]})")
        return
    y0, y1 = exp["woba_weights_year_range"]
    have = set(w["Season"].astype(int))
    missing = [y for y in range(y0, y1 + 1) if y not in have]
    if missing:
        log("FAIL", f"{WOBA_WEIGHTS.name} missing seasons: {missing[:10]}"
                    f"{'...' if len(missing) > 10 else ''}")
    else:
        log("PASS", f"{WOBA_WEIGHTS.name} covers {y0}-{y1}")


def check_teams():
    if not TEAMS.exists():
        log("FAIL", f"missing {TEAMS.name}")
        return
    t = pd.read_csv(TEAMS, low_memory=False)
    need = {"yearID", "teamID", "lgID", "park"}
    missing_cols = need - set(t.columns)
    if missing_cols:
        log("FAIL", f"{TEAMS.name} missing columns: {sorted(missing_cols)}")
        return
    ymax = int(t["yearID"].max())
    if ymax < 2024:
        log("WARN", f"{TEAMS.name} ends at {ymax}; fine for history, note for "
                    f"recent-season team/park mapping")
    else:
        log("PASS", f"{TEAMS.name} through {ymax}")


def season_from_columns(pf):
    """Return the name of the season column, or None."""
    cols = set(pf.schema_arrow.names)
    for c in ("season", "year", "yearID"):
        if c in cols:
            return c
    return None


def check_panel(exp, do_hash):
    if not PANEL.exists():
        log("FAIL", f"missing {PANEL}")
        return None
    pf = pq.ParquetFile(PANEL)
    nrows = pf.metadata.num_rows
    cols = pf.schema_arrow.names

    if nrows == exp["panel_rows"]:
        log("PASS", f"panel rows = {nrows:,} (matches paper)")
    else:
        log("FAIL", f"panel rows = {nrows:,}, expected {exp['panel_rows']:,} "
                    f"(diff {nrows - exp['panel_rows']:+,})")

    missing = [c for c in exp["required_columns"] if c not in cols]
    if missing:
        log("FAIL", f"panel missing required columns: {missing}")
    else:
        log("PASS", f"panel has required columns {exp['required_columns']}")
    log("INFO", f"panel columns ({len(cols)}): {cols}")

    scol = season_from_columns(pf)
    if scol is None:
        if "game_id" in cols:
            log("WARN", "no season column; deriving season from game_id "
                        "positions 3-7 (Retrosheet HOMYYYYMMDDN)")
            season = (pd.read_parquet(PANEL, columns=["game_id"])["game_id"]
                      .str.slice(3, 7).astype(int))
        else:
            log("FAIL", "no season column and no game_id to derive it from")
            return None
    else:
        season = pd.read_parquet(PANEL, columns=[scol])[scol].astype(int)

    smin, smax = int(season.min()), int(season.max())
    if smin == exp["season_min"] and smax == exp["season_max"]:
        log("PASS", f"panel seasons {smin}-{smax}")
    else:
        log("FAIL", f"panel seasons {smin}-{smax}, expected "
                    f"{exp['season_min']}-{exp['season_max']}")

    # per-season PA with outlier flags (rolling local median, known short
    # seasons exempt)
    cov = season.value_counts().sort_index().rename("pa").to_frame()
    cov.index.name = "season"
    cov = cov.reset_index()
    local_med = cov["pa"].rolling(11, center=True, min_periods=5).median()
    cov["ratio_to_local_median"] = (cov["pa"] / local_med).round(4)
    exempt = set(exp["short_seasons_no_flag"])
    cov["flag"] = ((cov["ratio_to_local_median"] < exp["season_pa_outlier_ratio"])
                   & ~cov["season"].isin(exempt))
    cov.to_csv(OUT_DIR / "season_coverage.csv", index=False)
    flagged = cov.loc[cov["flag"], "season"].tolist()
    if flagged:
        log("WARN", f"seasons with PA < {exp['season_pa_outlier_ratio']:.0%} of "
                    f"local median (excluding known short seasons): {flagged}")
    else:
        log("PASS", "no unexpected low-coverage seasons "
                    "(known short seasons exempt: "
                    f"{sorted(exempt)})")
    log("INFO", f"season PA range: min {cov['pa'].min():,} "
                f"({int(cov.loc[cov['pa'].idxmin(), 'season'])}), "
                f"max {cov['pa'].max():,} "
                f"({int(cov.loc[cov['pa'].idxmax(), 'season'])})")
    return nrows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-hash", action="store_true",
                    help="skip sha256 of large files (parquet)")
    args = ap.parse_args()

    exp_path = EXPECTATIONS_DIR / "expectations_00.json"
    if not exp_path.exists():
        print(f"FAIL  expectations file missing: {exp_path}")
        sys.exit(1)
    exp = json.loads(exp_path.read_text())

    log("INFO", f"stage 0 run {datetime.now().isoformat(timespec='seconds')}")
    for d, name in [(DATA_DIR, "data"), (OUT_DIR, "output"), (GITHUB_DIR, "GitHub")]:
        if d.exists():
            log("PASS", f"{name} directory: {d}")
        else:
            log("FAIL", f"{name} directory missing: {d}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # file manifest
    rows = []
    for f in exp["required_files"]:
        r = file_row(DATA_DIR / f, do_hash=not args.no_hash or f != "pa_panel.parquet")
        rows.append(r)
        log("PASS" if r["exists"] else "FAIL",
            f"{f}: {'present, ' + format(r['bytes'], ',') + ' bytes' if r['exists'] else 'MISSING'}")
    opt_missing_msg = ("not found - park crosswalk will need another source, "
                       "incl. the five unverified park codes (BOS08 priority)")
    for f in exp["optional_files"]:
        r = file_row(DATA_DIR / f, do_hash=True)
        rows.append(r)
        log("PASS" if r["exists"] else "WARN",
            f"{f} (optional): {'present' if r['exists'] else opt_missing_msg}")
    pd.DataFrame(rows).to_csv(OUT_DIR / "data_manifest.csv", index=False)

    check_gamelogs(exp)
    check_wind(exp)
    check_woba_weights(exp)
    check_teams()
    check_panel(exp, do_hash=not args.no_hash)

    fails = sum(1 for l in REPORT if l.startswith("FAIL"))
    warns = sum(1 for l in REPORT if l.startswith("WARN"))
    gate = "GATE: PASSED" if fails == 0 else "GATE: BLOCKED"
    log("INFO", f"{gate} - {fails} FAIL, {warns} WARN")
    (OUT_DIR / "setup_report.txt").write_text("\n".join(REPORT) + "\n",
                                              encoding="utf-8")
    print(f"\nwrote {OUT_DIR / 'data_manifest.csv'}")
    print(f"wrote {OUT_DIR / 'season_coverage.csv'}")
    print(f"wrote {OUT_DIR / 'setup_report.txt'}")
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
