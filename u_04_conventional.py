r"""Stage 4: conventional park factors (BR- and FG-style) from game logs,
plus every system's ex-ante prediction for each park-season.

Recipes (documented cores of the published methods; parameters in
expectations_04.json):
  raw one-year team factor  TF = (runs/game at home) / (runs/game on road)
  BR-style: trailing 3-year mean of TF, regressed toward 1 by adding
            br_reg_seasons of league-average
  FG-style: trailing 5-year mean, regressed by fg_reg_seasons
  mesoball: EB-shrunk trailing exponentially-weighted mean of the real-time
            park-season effects (halflife in expectations), converted to a
            run factor via the season's league runs/PA
  persistence: last season's raw TF;  nothing: 1.0000

Writes:
    output\team_season_factors.csv   raw TF per team-season (the 2,548 claim)
    output\park_factor_predictions.csv  one row per park-season x system
    output\conventional_summary.txt

Run:  py u_04_conventional.py        (from C:\SABR_Mesoball\GitHub)
"""
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd

from config import EXPECTATIONS_DIR, GAMELOG_DIR, OUT_DIR

REPORT = []


def log(level, msg):
    line = f"{level:5s} {msg}"
    REPORT.append(line)
    print(line, flush=True)


def load_gamelogs():
    rows = []
    for f in sorted(GAMELOG_DIR.iterdir()):
        m = re.match(r"(?i)gl(\d{4})\.txt$", f.name)
        if not m:
            continue
        gl = pd.read_csv(f, header=None, usecols=[0, 1, 3, 6, 9, 10, 16],
                         names=["date", "gamenum", "visteam", "hometeam",
                                "vis_runs", "home_runs", "park"],
                         dtype={"date": str, "gamenum": str, "visteam": str,
                                "hometeam": str, "park": str},
                         keep_default_na=False)
        gl["season"] = int(m.group(1))
        rows.append(gl)
    gl = pd.concat(rows, ignore_index=True)
    gl["runs"] = pd.to_numeric(gl["vis_runs"], errors="coerce") + \
        pd.to_numeric(gl["home_runs"], errors="coerce")
    return gl.dropna(subset=["runs"])


def team_factors(gl):
    """Raw one-year TF per team-season, assigned to the team's modal park."""
    out = []
    for (season, team), g_home in gl.groupby(["season", "hometeam"]):
        road = gl[(gl["season"] == season) & (gl["visteam"] == team)]
        if len(g_home) < 20 or len(road) < 20:
            continue
        tf = (g_home["runs"].mean()) / (road["runs"].mean())
        park = g_home["park"].mode().iat[0]
        out.append({"season": season, "team": team, "park_id": park,
                    "tf": tf, "home_g": len(g_home)})
    return pd.DataFrame(out)


def park_year_tf(tf):
    """Average team factors to park-season (multi-tenant parks averaged)."""
    return (tf.groupby(["park_id", "season"])
            .apply(lambda d: np.average(d["tf"], weights=d["home_g"]),
                   include_groups=False).rename("tf").reset_index())


def trailing_regressed(pyt, park, y, window, reg_seasons):
    h = pyt[(pyt["park_id"] == park)
            & (pyt["season"] < y) & (pyt["season"] >= y - window)]
    if not len(h):
        return np.nan
    n = len(h)
    return (h["tf"].mean() * n + 1.0 * reg_seasons) / (n + reg_seasons)


def mesoball_regime(rt_park, y, sigma2, prior_var, min_seg, thresh):
    """The paper's stated method: detect breaks in real time using data
    through y-1 only, then predict the current regime's EB-shrunk mean."""
    from u_03b_breaks import segment
    h = rt_park[rt_park["season"] < y]
    if len(h) < 3:
        return np.nan
    e = h["effect"].values
    n = h["n_pa"].values
    seasons = h["season"].values
    if len(h) >= 2 * min_seg:
        segs = segment(e, n / sigma2, seasons, min_seg, thresh)
        s0 = sorted(segs)[-1][0]
    else:
        s0 = seasons[0]
    mask = seasons >= s0
    est = np.average(e[mask], weights=n[mask])
    se2 = sigma2 / n[mask].sum()
    return est * prior_var / (prior_var + se2)


def main():
    exp = json.loads((EXPECTATIONS_DIR / "expectations_04.json").read_text())
    log("INFO", f"stage 4 {datetime.now().isoformat(timespec='seconds')}")

    gl = load_gamelogs()
    tf = team_factors(gl)
    tf.to_csv(OUT_DIR / "team_season_factors.csv", index=False)
    lo, hi = exp["team_seasons_range"]
    log("PASS" if lo <= len(tf) <= hi else "FAIL",
        f"conventional factors for {len(tf):,} team-seasons "
        f"(expected {lo:,}-{hi:,}; paper says 2,548)")

    pyt = park_year_tf(tf)
    league_rpg = gl.groupby("season")["runs"].mean()
    # league runs per PA per season, for factor<->effect conversion
    pa = pd.read_parquet(OUT_DIR / "pa_derived.parquet",
                         columns=["season", "game_id"])
    pa_per_game = pa.groupby("season").size() / \
        pa.groupby("season")["game_id"].nunique()
    league_rpp = (league_rpg / pa_per_game).rename("league_rpp")

    rt = pd.read_csv(OUT_DIR / "park_by_season_realtime.csv")
    from u_03b_breaks import estimate_sigma2
    sigma2 = estimate_sigma2(rt)
    log("INFO", f"sigma2 (real-time series) = {sigma2:.4f}")
    rt_by_park = {p: g.sort_values("season") for p, g in rt.groupby("park_id")}

    rows = []
    for park, y in pyt[["park_id", "season"]].itertuples(index=False):
        rpp = league_rpp.get(y, np.nan)
        br = trailing_regressed(pyt, park, y, exp["br_window"],
                                exp["br_reg_seasons"])
        fg = trailing_regressed(pyt, park, y, exp["fg_window"],
                                exp["fg_reg_seasons"])
        rtp = rt_by_park.get(park)
        meso_eff = (mesoball_regime(rtp, y, sigma2, exp["meso_prior_var"],
                                    exp["meso_min_segment"],
                                    exp["meso_sup_lr_threshold"])
                    if rtp is not None else np.nan)
        last = pyt[(pyt["park_id"] == park) & (pyt["season"] == y - 1)]
        pers = last["tf"].iat[0] if len(last) else np.nan
        rows.append({
            "park_id": park, "season": y, "league_rpp": rpp,
            "pred_meso_factor": 1 + meso_eff / rpp if rpp else np.nan,
            "pred_br_factor": br, "pred_fg_factor": fg,
            "pred_persistence_factor": pers, "pred_nothing_factor": 1.0,
            "pred_meso_effect": meso_eff,
            "pred_br_effect": (br - 1) * rpp if br == br else np.nan,
            "pred_fg_effect": (fg - 1) * rpp if fg == fg else np.nan,
            "pred_persistence_effect": (pers - 1) * rpp if pers == pers else np.nan,
            "pred_nothing_effect": 0.0})
    pred = pd.DataFrame(rows)
    pred.to_csv(OUT_DIR / "park_factor_predictions.csv", index=False)
    log("INFO", f"predictions for {len(pred):,} park-seasons")

    # optional: correlation with published factors, if provided
    pub = OUT_DIR.parent / "data" / "published_factors.csv"
    if pub.exists():
        p = pd.read_csv(pub)
        m = p.merge(tf, on=["season", "team"])
        c = m["factor"].corr(m["tf"])
        log("PASS" if c >= exp["published_corr_min"] else "WARN",
            f"correlation with published factors {c:.3f} "
            f"(paper says 0.94)")
    else:
        log("WARN", "data\\published_factors.csv not present; the 0.94 "
                    "published-correlation claim stays unverified this run")

    fails = sum(1 for l in REPORT if l.startswith("FAIL"))
    warns = sum(1 for l in REPORT if l.startswith("WARN"))
    log("INFO", f"{'GATE: PASSED' if fails == 0 else 'GATE: BLOCKED'} - "
                f"{fails} FAIL, {warns} WARN")
    (OUT_DIR / "conventional_summary.txt").write_text("\n".join(REPORT) + "\n",
                                                      encoding="utf-8")


if __name__ == "__main__":
    main()
