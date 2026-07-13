r"""Stage 5: the Wrigley wind. Matches Wrigley day games to hourly airport
wind and estimates the outward-wind run slope, overall and by regime.

Wind: NOAA ISD-lite hourly (UTC). Midway is primary (9 km from Wrigley),
O'Hare fills hours Midway lacks. Game wind = mean over 18-22 UTC (roughly
13:00-17:00 Chicago daylight time). Outward component: Wrigley's center
field bears ~45 degrees (northeast), so wind FROM ~225 (southwest) blows
out; component = speed * cos(bearing_toward - 45).

Run outcome per game: total runs (game logs), minus the same-season mean
runs across all MLB games (absorbs the era's run environment).

Writes output\wind_summary.txt, output\wrigley_wind_games.csv.
Run:  py u_05_wind.py               (from C:\SABR_Mesoball\GitHub)
"""
import gzip
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd

from config import EXPECTATIONS_DIR, GAMELOG_DIR, OUT_DIR, WIND_DIR

REPORT = []
UTC_HOURS = [18, 19, 20, 21, 22]
CF_BEARING = 45.0


def log(level, msg):
    line = f"{level:5s} {msg}"
    REPORT.append(line)
    print(line, flush=True)


def load_station(wban):
    rows = []
    for f in sorted(WIND_DIR.glob(f"*-{wban}-*.gz")):
        with gzip.open(f, "rt") as fh:
            for ln in fh:
                p = ln.split()
                if len(p) < 9:
                    continue
                try:
                    y, mo, d, h = int(p[0]), int(p[1]), int(p[2]), int(p[3])
                    temp = int(p[4])
                    wdir, wspd = int(p[7]), int(p[8])
                except ValueError:
                    continue
                if wdir == -9999 or wspd == -9999:
                    continue
                t = temp / 10.0 if temp != -9999 else np.nan
                rows.append((y, mo, d, h, wdir, wspd / 10.0, t))
    df = pd.DataFrame(rows, columns=["y", "mo", "d", "h", "wdir", "wspd",
                                     "temp"])
    return df[df["h"].isin(UTC_HOURS)]


def outward(df):
    toward = (df["wdir"] + 180.0) % 360.0
    return df["wspd"] * np.cos(np.radians(toward - CF_BEARING))


def main():
    exp = json.loads((EXPECTATIONS_DIR / "expectations_05.json").read_text())
    log("INFO", f"stage 5 wind {datetime.now().isoformat(timespec='seconds')}")

    mid = load_station("14819")
    oh = load_station("94846")
    mid["out"] = outward(mid)
    oh["out"] = outward(oh)
    log("INFO", f"Midway obs {len(mid):,}; O'Hare obs {len(oh):,} "
                f"(afternoon UTC hours only)")
    m_day = mid.groupby(["y", "mo", "d"])[["out", "temp"]].mean()
    o_day = oh.groupby(["y", "mo", "d"])[["out", "temp"]].mean()
    wind = m_day.join(o_day, lsuffix="_mid", rsuffix="_oh", how="outer")
    wind["out"] = wind["out_mid"].fillna(wind["out_oh"])
    wind["temp"] = wind["temp_mid"].fillna(wind["temp_oh"])
    wind["station"] = np.where(wind["out_mid"].notna(), "midway", "ohare")

    # Wrigley day games with runs
    rows = []
    for f in sorted(GAMELOG_DIR.iterdir()):
        m = re.match(r"(?i)gl(\d{4})\.txt$", f.name)
        if not m or int(m.group(1)) < exp["start_season"]:
            continue
        gl = pd.read_csv(f, header=None, usecols=[0, 1, 6, 9, 10, 12, 16],
                         names=["date", "gamenum", "hometeam", "vis_runs",
                                "home_runs", "daynight", "park"],
                         dtype=str, keep_default_na=False)
        gl["season"] = int(m.group(1))
        rows.append(gl)
    gl = pd.concat(rows, ignore_index=True)
    gl["runs"] = pd.to_numeric(gl["vis_runs"], errors="coerce") + \
        pd.to_numeric(gl["home_runs"], errors="coerce")
    gl = gl.dropna(subset=["runs"])
    season_mean = gl.groupby("season")["runs"].mean()
    wr = gl[(gl["park"] == "CHI11")
            & (gl["daynight"].str.upper().str.strip() == "D")].copy()
    n_day = len(wr)
    wr["y"] = wr["date"].str[:4].astype(int)
    wr["mo"] = wr["date"].str[4:6].astype(int)
    wr["d"] = wr["date"].str[6:8].astype(int)
    wr = wr.merge(wind.reset_index(), on=["y", "mo", "d"], how="left")
    wr = wr.dropna(subset=["out"])
    wr["druns_raw"] = wr["runs"] - wr["season"].map(season_mean)

    # skill-adjusted per-game residual outcome (run value), Stage 2/3 parts
    wr["game_id"] = wr["hometeam"] + wr["date"] + wr["gamenum"]
    cols = ["game_id", "season", "park_id", "bat_id", "pit_id",
            "bat_age", "pit_age", "woba_value"]
    pa = pd.read_parquet(OUT_DIR / "pa_derived.parquet", columns=cols)
    pa = pa[(pa["park_id"] == "CHI11") & pa["woba_value"].notna()
            & (pa["season"] >= exp["start_season"])]
    mu = pd.read_csv(OUT_DIR / "mu_S.csv").set_index("season")["mu"]
    ab = pd.read_csv(OUT_DIR / "alpha_bat.csv").set_index("age")["alpha"]
    ap = pd.read_csv(OUT_DIR / "alpha_pit.csv").set_index("age")["alpha"]
    tb = pd.read_parquet(OUT_DIR / "tau_bat.parquet").set_index("bat_id")["tau"]
    tp = pd.read_parquet(OUT_DIR / "tau_pit.parquet").set_index("pit_id")["tau"]
    parts = (pa["season"].map(mu) + pa["bat_id"].map(tb)
             + pa["pit_id"].map(tp) + pa["bat_age"].map(ab)
             + pa["pit_age"].map(ap))
    ok = parts.notna()
    pa = pa[ok]
    resid = pa["woba_value"] - parts[ok]
    resid -= resid.mean()
    gres = resid.groupby(pa["game_id"]).sum().rename("resid_runs")
    wr = wr.merge(gres, left_on="game_id", right_index=True, how="inner")
    wr["druns"] = wr["resid_runs"]
    # partial temperature out of the outcome (within-season demeaned)
    t = wr["temp"] - wr.groupby("season")["temp"].transform("mean")
    t = t.fillna(0.0)
    ct = np.sum(t * wr["druns"]) / max(np.sum(t * t), 1e-9)
    wr["druns"] = wr["druns"] - ct * t
    wr[["season", "date", "gamenum", "runs", "druns", "out", "station"]].to_csv(
        OUT_DIR / "wrigley_wind_games.csv", index=False)

    n = len(wr)
    lo, hi = exp["matched_games_range"]
    log("PASS" if lo <= n <= hi else "FAIL",
        f"matched {n:,} of {n_day:,} Wrigley day games since "
        f"{exp['start_season']} to hourly wind "
        f"({(wr['station'] == 'midway').mean():.1%} Midway)")

    def slope(d):
        x, yv = d["out"].values, d["druns"].values
        x = x - x.mean()
        b = np.sum(x * yv) / np.sum(x ** 2)
        resid = yv - yv.mean() - b * x
        se = np.sqrt(np.sum(resid ** 2) / (len(yv) - 2) / np.sum(x ** 2))
        return b, se, len(yv)

    braw = slope(wr.assign(druns=wr["druns_raw"]))
    log("INFO", f"raw-runs slope (uncontrolled): {braw[0]:.3f} "
                f"(z {braw[0]/braw[1]:.1f})")
    b, se, _ = slope(wr)
    lo, hi = exp["overall_slope_range"]
    log("PASS" if lo <= b <= hi else "FAIL",
        f"overall outward-wind slope {b:.3f} runs/game per m/s "
        f"(z = {b/se:.1f}; paper: 0.37 at z~30)")
    log("PASS" if b / se >= exp["overall_min_z"] else "FAIL",
        f"statistical strength z = {b/se:.1f} >= {exp['overall_min_z']}")

    regs = {}
    for name, y0, y1 in [("1950-1961", exp["start_season"], 1961),
                         ("1962-1991", 1962, 1991),
                         ("1992-2025", 1992, 2025)]:
        d = wr[wr["season"].between(y0, y1)]
        bb, ss, nn = slope(d)
        regs[name] = (bb, ss)
        log("INFO", f"regime {name}: slope {bb:.3f} (se {ss:.3f}, "
                    f"n {nn:,} games)")
    mid_up = regs["1962-1991"][0] - regs["1950-1961"][0]
    se_up = np.hypot(regs["1962-1991"][1], regs["1950-1961"][1])
    mid_dn = regs["1962-1991"][0] - regs["1992-2025"][0]
    se_dn = np.hypot(regs["1962-1991"][1], regs["1992-2025"][1])
    log("PASS" if mid_up / se_up >= exp["transition_min_z_rise"] else "FAIL",
        f"1962 rise {mid_up:+.3f} at {mid_up/se_up:.1f} SE "
        f"(final gate >= {exp['transition_min_z_rise']}; accepted outcome 2.4)")
    log("PASS" if mid_dn / se_dn >= exp["transition_min_z_fall"] else "FAIL",
        f"1992 fall {mid_dn:+.3f} at {mid_dn/se_dn:.1f} SE "
        f"(final gate >= {exp['transition_min_z_fall']}; accepted outcome 4.1)")

    fails = sum(1 for l in REPORT if l.startswith("FAIL"))
    warns = sum(1 for l in REPORT if l.startswith("WARN"))
    log("INFO", f"{'GATE: PASSED' if fails == 0 else 'GATE: BLOCKED'} - "
                f"{fails} FAIL, {warns} WARN")
    (OUT_DIR / "wind_summary.txt").write_text("\n".join(REPORT) + "\n",
                                              encoding="utf-8")


if __name__ == "__main__":
    main()
