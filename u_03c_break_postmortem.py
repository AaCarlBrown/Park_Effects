r"""Stage 3c: break-dating post-mortem for the headline breaks.

For each configured break, profiles the single-split objective by candidate
year, twice: on the RAW series and on the SHRUNK series. If the raw profile
peaks at 1934 for Fenway while the shrunk profile peaks at 1935, the original
one-year miss is diagnosed as shrinkage smoothing the evidence before the
screen saw it. Per the pre-registered rule, the catalogue's series choice is
then fixed (raw) for all parks uniformly -- which is what u_03b already does.

Writes:
    output\break_postmortem.csv    profiles (park, candidate year, gain, series)
    output\break_postmortem.txt    verdicts

Run:  py u_03c_break_postmortem.py  (from C:\SABR_Mesoball\GitHub)
"""
from datetime import datetime

import numpy as np
import pandas as pd

from config import OUT_DIR
from u_03b_breaks import estimate_sigma2, wsse

CASES = [
    ("Fenway 1934/35 (screen said 1937)", "BOS07", 1930, 1940),
    ("Fenway 1985 (screen said 1986)", "BOS07", 1980, 1990),
    ("Wrigley 1962", "CHI11", 1957, 1967),
    ("Wrigley 1992", "CHI11", 1987, 1997),
    ("Cleveland 1947 (screen said 1952)", "CLE07", 1944, 1958),
    ("Comiskey Sox Sod (screen found nothing)", "CHI10", 1964, 1980),
    ("Sportsman's mid-1920s (screen said 1919)", "STL07", 1915, 1930),
    ("Crosley restore 1958 (sub-threshold)", "CIN07", 1954, 1964),
]

LINES = []


def say(msg=""):
    LINES.append(str(msg))
    print(msg, flush=True)


def profile(g, col, sigma2, y_lo, y_hi):
    g = g.sort_values("season")
    e = g[col].values
    w = g["n_pa"].values / sigma2
    seasons = g["season"].values
    base, _ = wsse(e, w)
    out = []
    for c in range(y_lo, y_hi + 1):
        k = int(np.searchsorted(seasons, c))
        if k < 2 or k > len(e) - 2:
            continue
        s1, m1 = wsse(e[:k], w[:k])
        s2, m2 = wsse(e[k:], w[k:])
        out.append({"candidate": c, "gain": base - s1 - s2,
                    "mean_pre": m1, "mean_post": m2})
    return pd.DataFrame(out)


def main():
    say(f"stage 3c post-mortem {datetime.now().isoformat(timespec='seconds')}")
    raw = pd.read_csv(OUT_DIR / "park_by_season_hindsight.csv")
    shr = pd.read_csv(OUT_DIR / "park_by_season_shrunk.csv")
    reg = pd.read_csv(OUT_DIR / "park_regimes.csv")
    sigma2 = estimate_sigma2(raw[raw["park_id"].isin({p for _, p, _, _ in CASES})])
    say(f"sigma2 = {sigma2:.4f}; single-season SE at 6,000 PA = "
        f"{np.sqrt(sigma2 / 6000) * 1000:.1f} pts")

    all_profiles = []
    for name, pid, y_lo, y_hi in CASES:
        # bound the series by the enclosing regime boundaries so neighboring
        # breaks do not contaminate the profile
        r = reg[reg["park_id"] == pid].sort_values("start")
        lo_bound = max([s for s in r["start"] if s <= y_lo], default=y_lo - 15)
        hi_bound = min([e for e in r["end"] if e >= y_hi], default=y_hi + 15)
        say(f"\n=== {name} ({pid}), window {y_lo}-{y_hi}, "
            f"series bounded {lo_bound}-{hi_bound} ===")
        g_raw = raw[(raw["park_id"] == pid)
                    & raw["season"].between(lo_bound, hi_bound)]
        g_shr = shr[(shr["park_id"] == pid)
                    & shr["season"].between(lo_bound, hi_bound)]
        # the shrunk CSV carries no n_pa; take weights from the raw series
        g_shr = g_shr.merge(g_raw[["park_id", "season", "n_pa"]],
                            on=["park_id", "season"], how="left")
        p_raw = profile(g_raw, "effect", sigma2, y_lo, y_hi)
        p_shr = profile(g_shr.rename(columns={"effect_shrunk": "eff"}),
                        "eff", sigma2, y_lo, y_hi)
        p_raw["series"], p_shr["series"] = "raw", "shrunk"
        p_raw["case"] = p_shr["case"] = name
        all_profiles += [p_raw, p_shr]
        b_raw = int(p_raw.loc[p_raw["gain"].idxmax(), "candidate"])
        b_shr = int(p_shr.loc[p_shr["gain"].idxmax(), "candidate"])
        say(f"raw-series profile peak:    {b_raw}")
        say(f"shrunk-series profile peak: {b_shr}")
        say("raw profile (gain by candidate year):")
        say(p_raw[["candidate", "gain"]].round(1).to_string(index=False))
        # boundary-year detail
        for y in range(y_lo + 2, y_hi - 1):
            row = g_raw[g_raw["season"] == y]
            if len(row):
                say(f"  {y}: raw {row['effect'].iloc[0] * 1000:+.1f} pts "
                    f"(n={int(row['n_pa'].iloc[0]):,})")

    pd.concat(all_profiles, ignore_index=True).to_csv(
        OUT_DIR / "break_postmortem.csv", index=False)
    (OUT_DIR / "break_postmortem.txt").write_text("\n".join(LINES) + "\n",
                                                  encoding="utf-8")
    print(f"\nwrote {OUT_DIR / 'break_postmortem.txt'}")


if __name__ == "__main__":
    main()
