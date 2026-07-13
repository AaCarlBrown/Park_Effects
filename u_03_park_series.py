r"""Stage 3: park-by-season effect series, hindsight and real-time.

Park effects are mean residuals of the Stage 2 two-way model, by park-season
(the original ballpark_residuals.py approach: within-player, quality-adjusted,
so no park term contaminates the skill model). Residuals are reconstructed
exactly from the saved fixed effects (linear model), so no refit is needed.

Two series:
  hindsight  - residuals from the full-sample fit (Stage 2 outputs)
  realtime   - for each season Y, residuals of season-Y PA computed with the
               as-of-Y ex-ante skills (Stage 2b outputs)

Each season's park-team graph is checked for connectivity (union-find). In a
disconnected season (2020 pods), park means are re-centered within component
and flagged: they are not comparable across components.

Writes:
    output\park_by_season_hindsight.csv
    output\park_by_season_realtime.csv
    output\park_series_summary.txt

Run:  py u_03_park_series.py          (from C:\SABR_Mesoball\GitHub)
"""
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from config import EXPECTATIONS_DIR, OUT_DIR

EXANTE_DIR = OUT_DIR / "exante"
REPORT = []


def log(level, msg):
    line = f"{level:5s} {msg}"
    REPORT.append(line)
    print(line, flush=True)


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def season_components(df_season):
    """Component id per park, from the hometeam-visteam graph."""
    uf = UnionFind()
    pairs = df_season[["hometeam", "visteam"]].drop_duplicates()
    for h, v in pairs.itertuples(index=False):
        uf.union(h, v)
    park_home = df_season[["park_id", "hometeam"]].drop_duplicates()
    roots = {t: uf.find(t) for t in set(park_home["hometeam"])}
    labels = {r: i for i, r in enumerate(sorted(set(roots.values())))}
    return {p: labels[roots[h]] for p, h in park_home.itertuples(index=False)}, \
        len(labels)


def residuals_from_parts(df, mu, tau_b, tau_p, ab, ap):
    parts = (df["season"].map(mu)
             + df["bat_id"].map(tau_b)
             + df["pit_id"].map(tau_p)
             + df["bat_age"].map(ab)
             + df["pit_age"].map(ap))
    ok = parts.notna() & df["woba_value"].notna()
    r = df.loc[ok, "woba_value"] - parts[ok]
    r -= r.mean()          # absorbs the anchoring constant, exactly
    return r, ok


def park_means(df, r, ok):
    d = df.loc[ok, ["park_id", "season", "hometeam", "visteam"]].copy()
    d["resid"] = r.values
    rows = []
    for season, g in d.groupby("season"):
        comp, ncomp = season_components(g)
        g = g.copy()
        g["component"] = g["park_id"].map(comp)
        # re-center within component (within-league-relative effects)
        for c, gc in g.groupby("component"):
            m = gc["resid"].mean()
            pk = gc.groupby("park_id")["resid"].agg(["mean", "size"])
            for pid, row in pk.iterrows():
                rows.append({"park_id": pid, "season": season,
                             "effect": row["mean"] - m, "n_pa": int(row["size"]),
                             "component": c, "n_components": ncomp})
    return pd.DataFrame(rows)


def main():
    exp = json.loads((EXPECTATIONS_DIR / "expectations_03.json").read_text())
    log("INFO", f"stage 3 park series {datetime.now().isoformat(timespec='seconds')}")

    cols = ["park_id", "season", "hometeam", "visteam", "bat_id", "pit_id",
            "bat_age", "pit_age", "woba_value"]
    panel = pd.read_parquet(OUT_DIR / "pa_derived.parquet", columns=cols)
    panel = panel[panel["woba_value"].notna()]

    # ---- hindsight series from Stage 2 outputs ----
    mu = pd.read_csv(OUT_DIR / "mu_S.csv").set_index("season")["mu"]
    ab = pd.read_csv(OUT_DIR / "alpha_bat.csv").set_index("age")["alpha"]
    ap = pd.read_csv(OUT_DIR / "alpha_pit.csv").set_index("age")["alpha"]
    tb = pd.read_parquet(OUT_DIR / "tau_bat.parquet").set_index("bat_id")["tau"]
    tp = pd.read_parquet(OUT_DIR / "tau_pit.parquet").set_index("pit_id")["tau"]
    r, ok = residuals_from_parts(panel, mu, tb, tp, ab, ap)
    log("INFO", f"hindsight residuals on {ok.mean():.2%} of PA")
    hind = park_means(panel, r, ok)
    hind.to_csv(OUT_DIR / "park_by_season_hindsight.csv", index=False)

    # ---- gates on the hindsight series ----
    def regime_mean(pid, y0, y1):
        s = hind[(hind["park_id"] == pid) & hind["season"].between(y0, y1)]
        return np.average(s["effect"], weights=s["n_pa"]) if len(s) else np.nan

    fen = regime_mean("BOS07", 1935, 1984) - regime_mean("BOS07", 1912, 1933)
    lo, hi = exp["fenway_1935_shift_range"]
    log("PASS" if lo <= fen <= hi else "FAIL",
        f"Fenway regime shift (1935-84 minus 1912-33) = {fen:+.4f} "
        f"runs/PA (expected [{lo:+.3f}, {hi:+.3f}])")
    wri = regime_mean("CHI11", 1962, 1991) - regime_mean("CHI11", 1914, 1961)
    lo, hi = exp["wrigley_1962_shift_range"]
    log("PASS" if lo <= wri <= hi else "FAIL",
        f"Wrigley regime shift (1962-91 minus 1914-61) = {wri:+.4f} "
        f"runs/PA (expected [{lo:+.3f}, {hi:+.3f}])")
    n2020 = hind.loc[hind["season"] == 2020, "n_components"].max()
    log("PASS" if n2020 == exp["components_2020"] else "FAIL",
        f"2020 components detected: {n2020} (expected {exp['components_2020']})")
    # AL and NL never meet in the regular season before interleague play
    # (June 1997), so pre-1997 seasons have exactly two components; the
    # within-component re-centering is the league-relative convention.
    nc = hind.groupby("season")["n_components"].max()
    y_il = exp["interleague_from"]
    pre_bad = nc[(nc.index < y_il) & (nc != 2)]
    post_bad = nc[(nc.index >= y_il) & (nc.index != 2020) & (nc != 1)]
    if len(pre_bad) == 0 and len(post_bad) == 0:
        log("PASS", f"components: 2 (AL/NL) every season before {y_il}, "
                    f"1 from {y_il} on (2020 excepted)")
    else:
        log("FAIL", f"component anomalies: pre-{y_il} {dict(pre_bad)}, "
                    f"post {dict(post_bad)}")

    # ---- real-time series from Stage 2b ----
    rt_rows, cov_notes = [], []
    for y in sorted(panel["season"].unique()):
        f_tb = EXANTE_DIR / f"tau_bat_{y}.parquet"
        if not f_tb.exists():
            continue
        tb_y = pd.read_parquet(f_tb).set_index("bat_id")["tau"]
        tp_y = pd.read_parquet(EXANTE_DIR / f"tau_pit_{y}.parquet"
                               ).set_index("pit_id")["tau"]
        ma = pd.read_parquet(EXANTE_DIR / f"mu_alpha_{y}.parquet")
        mu_y = ma[ma["kind"] == "mu"].set_index("key")["value"]
        ab_y = ma[ma["kind"] == "alpha_bat"].set_index("key")["value"]
        ap_y = ma[ma["kind"] == "alpha_pit"].set_index("key")["value"]
        py = panel[panel["season"] == y]
        r_y, ok_y = residuals_from_parts(py, mu_y, tb_y, tp_y, ab_y, ap_y)
        cov_notes.append(ok_y.mean())
        rt_rows.append(park_means(py, r_y, ok_y))
    rt = pd.concat(rt_rows, ignore_index=True)
    rt.to_csv(OUT_DIR / "park_by_season_realtime.csv", index=False)
    cov = float(np.mean(cov_notes))
    lo = exp["realtime_min_mean_coverage"]
    log("PASS" if cov >= lo else "WARN",
        f"realtime residual coverage averages {cov:.2%} of PA "
        f"(as-of skills exist only for already-seen players)")

    # hindsight vs realtime agreement
    m = hind.merge(rt, on=["park_id", "season"], suffixes=("_h", "_r"))
    big = m[m["n_pa_h"] >= 3000]
    c = big["effect_h"].corr(big["effect_r"])
    lo = exp["hindsight_realtime_min_corr"]
    log("PASS" if c >= lo else "FAIL",
        f"hindsight vs realtime park-season correlation {c:.3f} "
        f"(n_pa>=3000; expected >= {lo})")

    fails = sum(1 for l in REPORT if l.startswith("FAIL"))
    warns = sum(1 for l in REPORT if l.startswith("WARN"))
    log("INFO", f"{'GATE: PASSED' if fails == 0 else 'GATE: BLOCKED'} - "
                f"{fails} FAIL, {warns} WARN")
    (OUT_DIR / "park_series_summary.txt").write_text("\n".join(REPORT) + "\n",
                                                     encoding="utf-8")
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
