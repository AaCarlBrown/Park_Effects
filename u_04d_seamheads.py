r"""Stage 4d: external cross-check of the rebuilt conventional factors
against the Seamheads Parkfactors Database (December 2024 CSV release).

The manuscript's original draft recalled a 0.94 correlation with published
factor values that the clean rerun could not verify. This stage replaces
that recollection with a reproducible number: our raw one-year team factor
(u_04, runs/game home over road, from Retrosheet game logs) correlated with
the same quantity computed from Seamheads' independently curated home/road
data, and, as the published-value analogue, with the Seamheads-style
one-year factor (per-AB rates plus their other-parks corrector).

LICENSE: the Seamheads database is licensed for individual research use and
may NOT be redistributed. It is not included in this repository; obtain it
from seamheads.com and set SABR_SEAMHEADS_DIR (default
ROOT\CSV Final Ballpark Files 2024). Outputs derived from it
(seamheads_check.csv) are for verification and must not be redistributed.

Gates are pre-registered in expectations\expectations_04d.json.

Run:  py u_04d_seamheads.py          (from C:\SABR_Mesoball\GitHub)
Reads: output\team_season_factors.csv (Stage 4), Seamheads CSVs.
Writes: output\seamheads_check.csv, output\seamheads_summary.txt
"""
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from config import EXPECTATIONS_DIR, OUT_DIR, SEAMHEADS_DIR

YEARS = (1910, 2024)


def log(level, msg):
    print(f"[{level}] {msg}")


def load_seamheads():
    kw = dict(encoding="latin-1", low_memory=False)
    home = pd.read_csv(SEAMHEADS_DIR / "Home_Main_Data_WO_Parks.csv", **kw)
    vis = pd.read_csv(SEAMHEADS_DIR / "Visitor_Main_Data.csv", **kw)
    xref = pd.read_csv(SEAMHEADS_DIR / "Retrosheet_BBDB_Team_XRef.csv",
                       encoding="latin-1")
    # one row per team-season on each side (sum to be safe)
    hcols = ["GP_H", "R_Off_H", "R_Def_H", "AB_Off_H", "AB_Def_H"]
    vcols = ["GP_A", "R_Off_A", "R_Def_A", "AB_Off_A", "AB_Def_A"]
    home = (home.groupby(["Year", "TeamID", "LgID"], as_index=False)[hcols]
            .sum(min_count=1))
    vis = (vis.groupby(["Year", "TeamID", "LgID"], as_index=False)[vcols]
           .sum(min_count=1))
    sh = home.merge(vis, on=["Year", "TeamID", "LgID"], how="inner")
    sh = sh[(sh["Year"] >= YEARS[0]) & (sh["Year"] <= YEARS[1])]

    # runs-per-game factor (mirrors u_04's TF definition)
    sh["tf_sh"] = ((sh["R_Off_H"] + sh["R_Def_H"]) / sh["GP_H"]) / \
                  ((sh["R_Off_A"] + sh["R_Def_A"]) / sh["GP_A"])

    # Seamheads-style published one-year factor: per-AB rates with the
    # documented other-parks corrector F = raw * N / (N - 1 + raw)
    rate_h = (sh["R_Off_H"] + sh["R_Def_H"]) / (sh["AB_Off_H"] + sh["AB_Def_H"])
    rate_a = (sh["R_Off_A"] + sh["R_Def_A"]) / (sh["AB_Off_A"] + sh["AB_Def_A"])
    sh["pf_raw"] = rate_h / rate_a
    n_lg = sh.groupby(["Year", "LgID"])["TeamID"].transform("count")
    sh["pf_sh"] = sh["pf_raw"] * n_lg / (n_lg - 1 + sh["pf_raw"])
    return sh, xref


def main():
    exp = json.loads((EXPECTATIONS_DIR / "expectations_04d.json")
                     .read_text(encoding="utf-8"))
    ours = pd.read_csv(OUT_DIR / "team_season_factors.csv")
    ours = ours[(ours["season"] >= YEARS[0]) & (ours["season"] <= YEARS[1])]
    sh, xref = load_seamheads()

    # XRef is year-specific; a team's rows stop either because the ID
    # difference ended or because the table ends (2021 in the December 2024
    # release, though the ANA/LAA difference persists). Rule: use the
    # year-specific mapping when present; otherwise identity if Seamheads
    # has that (year, team) key; otherwise the team's most recent mapping.
    xmap = {(r.Year, r.RetroID): r.BBDBID for r in xref.itertuples()}
    latest = (xref.sort_values("Year").groupby("RetroID").last()["BBDBID"])
    sh_keys = set(zip(sh["Year"], sh["TeamID"]))

    def to_bbdb(season, team):
        if (season, team) in xmap:
            return xmap[(season, team)]
        if (season, team) in sh_keys:
            return team
        return latest.get(team, team)

    ours["team_bbdb"] = [to_bbdb(s, t) for s, t in
                         zip(ours["season"], ours["team"])]

    m = ours.merge(sh, left_on=["season", "team_bbdb"],
                   right_on=["Year", "TeamID"], how="inner")
    unmatched = ours[~ours.set_index(["season", "team_bbdb"]).index.isin(
        m.set_index(["season", "team_bbdb"]).index)]

    lines = [f"stage 4d {datetime.now().isoformat(timespec='seconds')}",
             "Seamheads Parkfactors December 2024 CSV release "
             "(individual-research license; not redistributed)",
             f"our team-seasons {YEARS[0]}-{YEARS[1]}: {len(ours)}; "
             f"matched: {len(m)}; unmatched: {len(unmatched)}"]
    if len(unmatched):
        top = unmatched.groupby("team").size().sort_values(ascending=False)
        lines.append("unmatched by team: " +
                     ", ".join(f"{t}:{n}" for t, n in top.head(8).items()))

    gates = []

    # primary gate: same-definition factor, ours vs Seamheads data
    p = exp["primary"]
    r1 = m["tf"].corr(m["tf_sh"])
    mad = (m["tf"] - m["tf_sh"]).abs().mean()
    lines += ["", f"PRIMARY runs-per-game TF correlation: {r1:.4f} "
              f"(mean abs diff {mad:.4f})"]
    if len(m) < p["matched_min"]:
        gates.append(("FAIL", f"matched {len(m)} < {p['matched_min']}"))
    else:
        gates.append(("PASS", f"matched {len(m)} >= {p['matched_min']}"))
    if r1 < p["corr_min_fail"]:
        gates.append(("FAIL", f"primary corr {r1:.4f} < {p['corr_min_fail']}"))
    elif r1 < p["corr_min_warn"]:
        gates.append(("WARN", f"primary corr {r1:.4f} in "
                      f"[{p['corr_min_fail']}, {p['corr_min_warn']})"))
    else:
        gates.append(("PASS", f"primary corr {r1:.4f} >= {p['corr_min_warn']}"))

    # secondary, informational: published-style per-AB corrected factor
    s = exp["secondary"]
    lo, hi = s["corr_expected_range"]
    for label, sub in [("pooled 1910-2024", m),
                       ("1910-1987 (MacMillan-era source)",
                        m[m["season"] <= 1987]),
                       ("1988-2024 (game-log source)", m[m["season"] >= 1988])]:
        r2 = sub["tf"].corr(sub["pf_sh"])
        lines.append(f"SECONDARY published-style factor corr, {label}: "
                     f"{r2:.4f} (n={len(sub)})")
        if label.startswith("pooled"):
            inside = lo <= r2 <= hi
            gates.append(("PASS" if inside else "WARN",
                          f"secondary corr {r2:.4f} "
                          f"{'inside' if inside else 'OUTSIDE'} expected "
                          f"[{lo}, {hi}] (informational)"))

    # largest disagreements on the primary metric
    m["absdiff"] = (m["tf"] - m["tf_sh"]).abs()
    worst = m.nlargest(10, "absdiff")[
        ["season", "team", "park_id", "tf", "tf_sh", "absdiff"]]
    lines += ["", "largest primary disagreements:",
              worst.to_string(index=False, float_format="%.4f")]

    lines += ["", "gates:"] + [f"  {lv}  {msg}" for lv, msg in gates]
    worst_level = ("FAIL" if any(g[0] == "FAIL" for g in gates)
                   else "WARN" if any(g[0] == "WARN" for g in gates)
                   else "PASS")
    lines.append(f"STAGE 4d GATE: {worst_level}")

    out = m[["season", "team", "team_bbdb", "park_id", "home_g",
             "tf", "tf_sh", "pf_raw", "pf_sh"]]
    out.to_csv(OUT_DIR / "seamheads_check.csv", index=False)
    (OUT_DIR / "seamheads_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if worst_level != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
