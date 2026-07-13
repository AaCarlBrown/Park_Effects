r"""Stage 8 (v2): parks and fielders, with leave-own-park-out fielder control.

v1's shift-share used career rates including the park being scored; home
fielders play roughly half their careers in one park, so the park effect
was absorbed into the fielder rate and cells cancelled to zero (Rogers
Centre exactly 0.0 was the tell). v2 scores each park against what its
actual fielders do EVERYWHERE ELSE.

Definitions: BIP = bip==1; home runs excluded from first-touch attribution
(no fielder); conversion metric = putout share (putouts per ball in play
while standing at the position, leave-own-park-out); chances = po+a+e.

Writes park_fielding_effects.csv, park_fielding_cells.csv,
fielding_summary.txt.
Run:  py u_08_fielding.py           (from C:\SABR_Mesoball\GitHub)
Two streaming passes; expect tens of minutes.
"""
import argparse
import json
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from config import EXPECTATIONS_DIR, OUT_DIR, PLAYS_CSV

REPORT = []
POSN = list(range(1, 10))
CHUNK = 2_000_000
USECOLS = (["gid", "site", "bip", "noout", "firstf", "date", "hr", "pitcher"]
           + [f"f{p}" for p in range(2, 10)]
           + [f"po{p}" for p in POSN] + [f"a{p}" for p in POSN]
           + [f"e{p}" for p in POSN])


def log(level, msg):
    line = f"{level:5s} {msg}"
    REPORT.append(line)
    print(line, flush=True)


def fcol(p):
    return "pitcher" if p == 1 else f"f{p}"


def chunks():
    for ch in pd.read_csv(PLAYS_CSV, usecols=USECOLS, chunksize=CHUNK,
                          dtype=str, keep_default_na=False):
        ch["season"] = ch["date"].str[:4]
        yield ch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-cells", action="store_true",
                    help="skip the streaming passes; reuse park_fielding_cells.csv")
    args = ap.parse_args()
    exp = json.loads((EXPECTATIONS_DIR / "expectations_08.json").read_text())
    log("INFO", f"stage 8 v4 {datetime.now().isoformat(timespec='seconds')}")

    if args.from_cells:
        df = pd.read_csv(OUT_DIR / "park_fielding_cells.csv")
        log("INFO", "reusing park_fielding_cells.csv")
    else:
        # ---- pass 1: (fielder, pos, site) accumulators ----
        cn = defaultdict(float)   # first-touches
        co = defaultdict(float)   # outs on them
        ec = defaultdict(float)   # chances
        ee = defaultdict(float)   # errors
        for ch in chunks():
            b = ch[(ch["bip"] == "1") & (ch["hr"] != "1")]
            for p in POSN:
                po = pd.to_numeric(b[f"po{p}"], errors="coerce").fillna(0)
                sub = pd.DataFrame({"f": b[fcol(p)], "s": b["site"],
                                    "po": (po > 0).astype(float)})
                sub = sub[sub["f"] != ""]
                grp = sub.groupby(["f", "s"])
                for k, v in grp.size().items():
                    cn[(k[0], p, k[1])] += v
                for k, v in grp["po"].sum().items():
                    co[(k[0], p, k[1])] += v
            for p in POSN:
                po = pd.to_numeric(ch[f"po{p}"], errors="coerce").fillna(0)
                a_ = pd.to_numeric(ch[f"a{p}"], errors="coerce").fillna(0)
                e_ = pd.to_numeric(ch[f"e{p}"], errors="coerce").fillna(0)
                chn = po + a_ + e_
                m = chn > 0
                if m.any():
                    sub = pd.DataFrame({"f": ch.loc[m, fcol(p)],
                                        "s": ch.loc[m, "site"],
                                        "c": chn[m], "e": e_[m]})
                    grp = sub.groupby(["f", "s"])
                    for k, v in grp["c"].sum().items():
                        ec[(k[0], p, k[1])] += v
                    for k, v in grp["e"].sum().items():
                        ee[(k[0], p, k[1])] += v
        # totals per (fielder, pos)
        def totals(dd):
            t = defaultdict(float)
            for (f_, p, s), v in dd.items():
                t[(f_, p)] += v
            return t
        CN, CO, EC, EE = totals(cn), totals(co), totals(ec), totals(ee)
        log("INFO", f"pass 1 done: {len(CN):,} fielder-position keys")

        def conv_rate_excl(f_, p, site):
            n = CN.get((f_, p), 0) - cn.get((f_, p, site), 0)
            if n < 100:
                return np.nan
            return (CO.get((f_, p), 0) - co.get((f_, p, site), 0)) / n

        def err_rate_excl(f_, p, site):
            c = EC.get((f_, p), 0) - ec.get((f_, p, site), 0)
            if c < 100:
                return np.nan
            return (EE.get((f_, p), 0) - ee.get((f_, p, site), 0)) / c

        # ---- pass 2: park-season cells with leave-own-park-out expectations ----
        cell = defaultdict(lambda: np.zeros(6))
        park_bip = defaultdict(float)      # non-HR BIP per (site, season)
        park_first = defaultdict(float)    # attributed first touches (all pos)
        for ch in chunks():
            b = ch[(ch["bip"] == "1") & (ch["hr"] != "1")]
            for k, v in b.groupby(["site", "season"]).size().items():
                park_bip[k] += v
            # putout conservation numerator: total BIP putouts credited
            for p in POSN:
                po_all = pd.to_numeric(b[f"po{p}"], errors="coerce").fillna(0)
                for k, v in (po_all > 0).groupby(
                        [b["site"], b["season"]]).sum().items():
                    park_first[k] += v
            for p in POSN:
                po = pd.to_numeric(b[f"po{p}"], errors="coerce").fillna(0)
                sub = pd.DataFrame({"site": b["site"], "season": b["season"],
                                    "f": b[fcol(p)],
                                    "po": (po > 0).astype(float)})
                sub = sub[sub["f"] != ""]
                for (site, season), g in sub.groupby(["site", "season"]):
                    rates = g["f"].map(lambda f_: conv_rate_excl(f_, p, site))
                    okm = rates.notna()
                    c = cell[(site, season, p)]
                    c[0] += okm.sum()
                    c[1] += g.loc[okm, "po"].sum()
                    c[2] += rates[okm].sum()
            for p in POSN:
                po = pd.to_numeric(ch[f"po{p}"], errors="coerce").fillna(0)
                a_ = pd.to_numeric(ch[f"a{p}"], errors="coerce").fillna(0)
                e_ = pd.to_numeric(ch[f"e{p}"], errors="coerce").fillna(0)
                chn = po + a_ + e_
                m = chn > 0
                if not m.any():
                    continue
                sub = pd.DataFrame({"site": ch.loc[m, "site"],
                                    "season": ch.loc[m, "season"],
                                    "f": ch.loc[m, fcol(p)],
                                    "c": chn[m], "e": e_[m]})
                for (site, season), g in sub.groupby(["site", "season"]):
                    rates = g["f"].map(lambda f_: err_rate_excl(f_, p, site))
                    okm = rates.notna()
                    c = cell[(site, season, p)]
                    c[3] += g.loc[okm, "c"].sum()
                    c[4] += g.loc[okm, "e"].sum()
                    c[5] += (rates[okm] * g.loc[okm, "c"]).sum()

        rows = []
        for (site, season, p), c in cell.items():
            rows.append({"park_id": site, "season": int(season), "pos": p,
                         "n_first": c[0], "outs": c[1], "exp_outs": c[2],
                         "chances": c[3], "errors": c[4], "exp_errors": c[5],
                         "park_bip": park_bip.get((site, season), np.nan),
                         "park_first": park_first.get((site, season), np.nan)})
        df = pd.DataFrame(rows)
        df.to_csv(OUT_DIR / "park_fielding_cells.csv", index=False)


    g = df.groupby(["park_id", "pos"]).sum(numeric_only=True).reset_index()
    g = g[g["n_first"] >= exp["min_first_touches"]]
    # compositional shares: position's share of the park's ATTRIBUTED
    # putouts, so park-level recording coverage cancels
    tot_obs = g.groupby("park_id")["outs"].transform("sum")
    tot_exp = g.groupby("park_id")["exp_outs"].transform("sum")
    g["conv_eff"] = g["outs"] / tot_obs - g["exp_outs"] / tot_exp
    sh = (g["exp_outs"] / tot_exp).clip(0.001, 0.999)
    g["conv_se"] = np.sqrt(sh * (1 - sh) / tot_obs)
    g["conv_z"] = g["conv_eff"] / g["conv_se"]
    g["coverage"] = tot_obs / g.groupby("park_id")["n_first"].transform("sum")
    g["err_eff"] = (g["errors"] - g["exp_errors"]) / g["chances"]
    g["err_se"] = np.sqrt(g["exp_errors"].clip(lower=1)) / g["chances"]
    g["err_z"] = g["err_eff"] / g["err_se"]
    try:
        pc = pd.read_csv(OUT_DIR.parent / "data" / "parkcode.txt", dtype=str,
                         keep_default_na=False)
        pc.columns = [c.upper() for c in pc.columns]
        g["name"] = g["park_id"].map(dict(zip(pc["PARKID"], pc["NAME"])))
    except Exception:
        g["name"] = g["park_id"]
    g.to_csv(OUT_DIR / "park_fielding_effects.csv", index=False)

    # ---- gates ----
    fen = g[(g["park_id"] == "BOS07") & (g["pos"] == 7)]
    if len(fen):
        z = float(fen["conv_z"].iloc[0])
        log("PASS" if z <= exp["fenway_lf_max_z"] else "FAIL",
            f"Fenway LF out-conversion deficit z = {z:.1f} "
            f"(paper: about -9; gate <= {exp['fenway_lf_max_z']})")
    lows = g.sort_values("conv_z").head(6)
    log("INFO", "most suppressed out-conversion (park, pos, z):\n" +
        lows[["name", "pos", "conv_z"]].round(1).to_string(index=False))
    highs = g.sort_values("conv_z", ascending=False).head(4)
    log("INFO", "most inflated:\n" +
        highs[["name", "pos", "conv_z"]].round(1).to_string(index=False))

    def pos_true_sd(pos_list):
        s = g[g["pos"].isin(pos_list) & (g["chances"] >= 2000)
              & np.isfinite(g["err_se"]) & (g["err_se"] > 0)]
        v = np.average(s["err_eff"] ** 2, weights=1 / s["err_se"] ** 2) \
            - (s["err_se"] ** 2).mean()
        return np.sqrt(max(v, 0)) * 1000
    sd_if, sd_2b, sd_of = (pos_true_sd([5, 6]), pos_true_sd([4]),
                           pos_true_sd([7, 8, 9]))
    log("PASS" if sd_if > sd_of else "FAIL",
        f"error-rate park true SD: 3B/SS {sd_if:.1f} pts, 2B {sd_2b:.1f}, "
        f"OF {sd_of:.1f} (paper: real at 3B/SS, nil in OF)")
    hot = g[g["pos"].isin([4, 5, 6])].groupby("name")["err_z"].mean()
    log("INFO", "hottest/cleanest infields (mean err z):\n" +
        hot.sort_values(ascending=False).head(4).round(1).to_string() +
        "\n...\n" + hot.sort_values().head(4).round(1).to_string())

    # conservation: attributed first-touches over non-HR BIP, per park
    pk = df.drop_duplicates(["park_id", "season"]).groupby("park_id")[
        ["park_first", "park_bip"]].sum()
    ratio = pk["park_first"] / pk["park_bip"]
    bad = int((ratio < exp["conservation_min"]).sum())
    log("PASS" if bad == 0 else "WARN",
        f"putout attribution: {bad} parks below "
        f"{exp['conservation_min']:.0%} BIP-putouts per non-HR BIP "
        f"(overall {ratio.mean():.2%})")

    fails = sum(1 for l in REPORT if l.startswith("FAIL"))
    warns = sum(1 for l in REPORT if l.startswith("WARN"))
    log("INFO", f"{'GATE: PASSED' if fails == 0 else 'GATE: BLOCKED'} - "
                f"{fails} FAIL, {warns} WARN")
    (OUT_DIR / "fielding_summary.txt").write_text("\n".join(REPORT) + "\n",
                                                  encoding="utf-8")


if __name__ == "__main__":
    main()
