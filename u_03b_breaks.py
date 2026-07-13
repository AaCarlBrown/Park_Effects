r"""Stage 3b: changepoint screen, regime means, shrinkage, variance decomposition.

Runs on the RAW hindsight park-by-season series (park_by_season_hindsight.csv).
Per the pre-registered Stage 3c rule, the raw-vs-shrunk choice for the final
catalogue is confirmed by the post-mortem (u_03c); the screen itself must run
on an unshrunk series because regime means are its output, not its input.

Method: recursive binary segmentation, PA-weighted. The sampling variance of
a park-season effect is sigma2/n_pa, with sigma2 (per-PA residual variance)
estimated robustly from first differences of the series (median-based, so the
handful of true breaks cannot inflate it). A split is accepted when the
weighted-SSE reduction exceeds sup_lr_threshold; both segments must have at
least min_segment seasons.

Writes:
    output\park_breaks.csv         park, break season, size, same-store size
    output\park_regimes.csv        park, regime start/end, mean
    output\park_by_season_shrunk.csv  EB-shrunk series (toward regime means)
    output\breaks_summary.txt      gates

Run:  py u_03b_breaks.py           (from C:\SABR_Mesoball\GitHub)
"""
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from config import EXPECTATIONS_DIR, OUT_DIR

REPORT = []


def log(level, msg):
    line = f"{level:5s} {msg}"
    REPORT.append(line)
    print(line, flush=True)


def estimate_sigma2(df):
    """Per-PA residual variance from robust first differences of the series."""
    vals = []
    for _, g in df.sort_values("season").groupby("park_id"):
        e, n = g["effect"].values, g["n_pa"].values
        if len(e) < 3:
            continue
        d = np.diff(e)
        scale = 1.0 / n[:-1] + 1.0 / n[1:]
        vals.append(d ** 2 / scale)
    v = np.concatenate(vals)
    return float(np.median(v) / 0.4549)   # median of chi-sq(1) = 0.4549


def wsse(e, w):
    m = np.average(e, weights=w)
    return float(np.sum(w * (e - m) ** 2)), m


def best_split(e, w, min_seg):
    base, _ = wsse(e, w)
    best = (None, 0.0)
    for k in range(min_seg, len(e) - min_seg + 1):
        s1, _ = wsse(e[:k], w[:k])
        s2, _ = wsse(e[k:], w[k:])
        gain = base - s1 - s2
        if gain > best[1]:
            best = (k, gain)
    return best


def segment(e, w, seasons, min_seg, thresh):
    """Recursive binary segmentation; returns list of regime slices."""
    k, gain = best_split(e, w, min_seg)
    if k is None or gain < thresh:
        return [(seasons[0], seasons[-1])]
    left = segment(e[:k], w[:k], seasons[:k], min_seg, thresh)
    right = segment(e[k:], w[k:], seasons[k:], min_seg, thresh)
    return left + right


def main():
    exp = json.loads((EXPECTATIONS_DIR / "expectations_03b.json").read_text())
    log("INFO", f"stage 3b breaks {datetime.now().isoformat(timespec='seconds')}")

    df = pd.read_csv(OUT_DIR / "park_by_season_hindsight.csv")
    tenure = df.groupby("park_id")["season"].nunique()
    keep = tenure[tenure >= exp["min_seasons"]].index
    lo, hi = exp["long_tenured_count_range"]
    log("PASS" if lo <= len(keep) <= hi else "FAIL",
        f"long-tenured parks (>= {exp['min_seasons']} seasons): {len(keep)} "
        f"(expected {lo}-{hi}; paper says 65)")
    df = df[df["park_id"].isin(keep)].sort_values(["park_id", "season"])

    sigma2 = estimate_sigma2(df)
    log("INFO", f"per-PA residual variance sigma2 = {sigma2:.4f} "
                f"(sd {np.sqrt(sigma2):.3f}); typical park-season SE "
                f"{np.sqrt(sigma2 / 6000) * 1000:.1f} pts")

    # weights are n/sigma2, so the split gain is already on the LR scale
    breaks, regimes, shrunk_rows = [], [], []
    for pid, g in df.groupby("park_id"):
        g = g.sort_values("season")
        e, n, seasons = g["effect"].values, g["n_pa"].values, g["season"].values
        w = n / 1.0
        segs = segment(e, w * (1 / sigma2), seasons, exp["min_segment"],
                       exp["sup_lr_threshold"])
        means = {}
        for (s0, s1) in segs:
            m = np.average(e[(seasons >= s0) & (seasons <= s1)],
                           weights=n[(seasons >= s0) & (seasons <= s1)])
            regimes.append({"park_id": pid, "start": s0, "end": s1,
                            "mean": m, "n_seasons": int(s1 - s0 + 1)})
            means[(s0, s1)] = m
        segs = sorted(segs)
        for i in range(1, len(segs)):
            prev, cur = segs[i - 1], segs[i]
            breaks.append({"park_id": pid, "break_season": int(cur[0]),
                           "size": means[cur] - means[prev],
                           "size_pts": round((means[cur] - means[prev]) * 1000, 1)})
        # EB shrinkage toward regime mean
        for (s0, s1) in segs:
            mask = (seasons >= s0) & (seasons <= s1)
            m = means[(s0, s1)]
            dev = e[mask] - m
            samp = sigma2 / n[mask]
            var_true = max(np.average(dev ** 2, weights=n[mask]) - samp.mean(), 0.0)
            rho = var_true / (var_true + samp)
            sh = m + rho * dev
            for s_, v_, raw_, n_ in zip(seasons[mask], sh, e[mask], n[mask]):
                shrunk_rows.append({"park_id": pid, "season": int(s_),
                                    "effect_raw": raw_, "effect_shrunk": v_,
                                    "n_pa": int(n_),
                                    "regime_start": s0, "regime_end": s1,
                                    "regime_mean": m})

    br = pd.DataFrame(breaks).sort_values("size_pts", key=abs, ascending=False)
    rg = pd.DataFrame(regimes)
    sh = pd.DataFrame(shrunk_rows)

    # ---- same-store correction for each break ----
    ss = []
    for _, b in br.iterrows():
        y = b["break_season"]
        pid = b["park_id"]
        pre = df[(df["season"] >= y - 8) & (df["season"] < y)]
        post = df[(df["season"] >= y) & (df["season"] < y + 8)]
        stable = (set(pre["park_id"].unique())
                  & set(post["park_id"].unique())) - {pid}
        def rel(win):
            own = win[win["park_id"] == pid]
            oth = win[win["park_id"].isin(stable)]
            if not len(own) or not len(oth):
                return np.nan
            return (np.average(own["effect"], weights=own["n_pa"])
                    - np.average(oth["effect"], weights=oth["n_pa"]))
        d_ss = rel(post) - rel(pre)
        ss.append(d_ss)
    br["size_same_store"] = ss
    br["composition_share"] = 1 - br["size_same_store"] / br["size"]

    br.to_csv(OUT_DIR / "park_breaks.csv", index=False)
    rg.to_csv(OUT_DIR / "park_regimes.csv", index=False)
    sh.to_csv(OUT_DIR / "park_by_season_shrunk.csv", index=False)

    # ---- gates ----
    lo, hi = exp["total_breaks_range"]
    log("PASS" if lo <= len(br) <= hi else "FAIL",
        f"total detected breaks: {len(br)} (expected {lo}-{hi}; paper says 16)")

    # ---- reconciliation of the manuscript's documented breaks ----
    # (Crosley is CIN07, not CIN08 as an earlier revision had it.)
    # Statuses: CONFIRMED (within +/-1 of the paper year), MOVED (a break at
    # the same park within +/-6 years), MISSING (nothing nearby). The gate
    # blocks only if a Wrigley break is missing (the paper's centerpiece);
    # MOVED/MISSING are findings for u_03c, and the paper is corrected to
    # the catalogue, never the reverse.
    claims = [("Fenway/Yawkey", "BOS07", 1935, 23.0),
              ("Fenway second", "BOS07", 1985, None),
              ("Wrigley up", "CHI11", 1962, None),
              ("Wrigley down", "CHI11", 1992, None),
              ("Coors humidor", "DEN02", 2002, -29.0),
              ("Cleveland Stadium", "CLE07", 1947, 24.0),
              ("Comiskey Sox Sod", "CHI10", 1969, 23.0),
              ("Comiskey turf out", "CHI10", 1976, -15.0),
              ("Crosley Goat Run", "CIN07", 1953, 19.0),
              ("Crosley restored", "CIN07", 1958, -9.0),
              ("Sportsman's expansion", "STL07", 1925, None),
              ("Ebbets", "BRO03", 1948, None),
              ("Riverfront", "CIN08", 1975, None)]
    recon = []
    for name, pid, y_paper, sz_paper in claims:
        near = br[(br["park_id"] == pid)
                  & (abs(br["break_season"] - y_paper) <= 6)]
        if len(near):
            near = near.iloc[(near["break_season"] - y_paper).abs().argsort()]
            y_got = int(near["break_season"].iloc[0])
            sz_got = float(near["size_pts"].iloc[0])
            status = "CONFIRMED" if abs(y_got - y_paper) <= 1 else "MOVED"
        else:
            y_got, sz_got, status = None, None, "MISSING"
        recon.append({"event": name, "park_id": pid, "paper_year": y_paper,
                      "paper_size_pts": sz_paper, "detected_year": y_got,
                      "detected_size_pts": sz_got, "status": status})
        lvl = "PASS" if status == "CONFIRMED" else "WARN"
        if status == "MISSING" and pid == "CHI11":
            lvl = "FAIL"
        log(lvl, f"{name}: {status}"
                 + (f" ({y_got}, {sz_got:+.1f} pts vs paper {y_paper})"
                    if y_got else f" (paper {y_paper})"))
    pd.DataFrame(recon).to_csv(OUT_DIR / "claims_vs_rerun.csv", index=False)

    # variance decomposition
    career = rg.groupby("park_id").apply(
        lambda g: np.average(g["mean"], weights=g["n_seasons"]))
    sd_between = float(np.std(career)) * 1000
    lo, hi = exp["between_park_sd_pts_range"]
    log("PASS" if lo <= sd_between <= hi else "FAIL",
        f"between-park SD of park means: {sd_between:.1f} pts "
        f"(expected {lo}-{hi}; paper says ~10)")
    # within-regime: observed variance vs sampling variance
    dfk = df.merge(rg, on="park_id")
    dfk = dfk[(dfk["season"] >= dfk["start"]) & (dfk["season"] <= dfk["end"])]
    dev2 = (dfk["effect"] - dfk["mean"]) ** 2
    ratio = float(np.average(dev2, weights=dfk["n_pa"])
                  / (sigma2 / dfk["n_pa"]).mean())
    lo, hi = exp["within_regime_var_ratio_range"]
    log("PASS" if lo <= ratio <= hi else "FAIL",
        f"within-regime observed/sampling variance ratio {ratio:.2f} "
        f"(expected {lo}-{hi}; ~1 means parks do not drift)")
    # composition share for Wrigley
    w1 = br[(br["park_id"] == "CHI11")
            & br["break_season"].isin(exp["wrigley1_years"])]
    if len(w1):
        cs = float(w1["composition_share"].iloc[0])
        lo, hi = exp["wrigley_composition_share_range"]
        log("PASS" if lo <= cs <= hi else "WARN",
            f"Wrigley break 1 composition share {cs:.2f} "
            f"(expected {lo}-{hi}; paper says quarter to a third)")

    log("INFO", "top 10 breaks by size:\n" +
        br.head(10).to_string(index=False))

    fails = sum(1 for l in REPORT if l.startswith("FAIL"))
    warns = sum(1 for l in REPORT if l.startswith("WARN"))
    log("INFO", f"{'GATE: PASSED' if fails == 0 else 'GATE: BLOCKED'} - "
                f"{fails} FAIL, {warns} WARN")
    (OUT_DIR / "breaks_summary.txt").write_text("\n".join(REPORT) + "\n",
                                                encoding="utf-8")
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
