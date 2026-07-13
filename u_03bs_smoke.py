r"""Stage 3b smoke test: plant breaks in synthetic park series, verify the
screen finds them at the right years and sizes with no false positives.

Run:  py u_03bs_smoke.py           (from C:\SABR_Mesoball\GitHub)
"""
import sys

import numpy as np
import pandas as pd

import u_03b_breaks as m

RNG = np.random.default_rng(23)
SIGMA_PA = 0.54
N_PA = 6000


def series(pid, means_by_span, y0=1920, y1=1990):
    rows = []
    for s in range(y0, y1 + 1):
        mean = next(v for (a, b), v in means_by_span.items() if a <= s <= b)
        eff = mean + RNG.normal(0, SIGMA_PA / np.sqrt(N_PA))
        rows.append({"park_id": pid, "season": s, "effect": eff, "n_pa": N_PA})
    return rows


def main():
    rows = []
    rows += series("FEN01", {(1920, 1934): -0.005, (1935, 1990): 0.018})
    rows += series("WRI01", {(1920, 1961): 0.001, (1962, 1990): 0.0165})
    # both breaks sized above the screen's power floor (~14 pts / 8 windows)
    rows += series("TWO01", {(1920, 1952): 0.000, (1953, 1960): 0.022,
                             (1961, 1990): 0.002})
    for i in range(5):
        rows += series(f"FLAT{i}", {(1920, 1990): RNG.normal(0, 0.008)})
    df = pd.DataFrame(rows)

    sigma2 = m.estimate_sigma2(df)
    print(f"sigma2 recovered {sigma2:.4f} (true {SIGMA_PA**2:.4f})")
    ok = abs(sigma2 - SIGMA_PA ** 2) / SIGMA_PA ** 2 < 0.25

    found = {}
    for pid, g in df.groupby("park_id"):
        g = g.sort_values("season")
        segs = m.segment(g["effect"].values,
                         g["n_pa"].values / sigma2,
                         g["season"].values, 4, 12.0)
        found[pid] = sorted(s0 for s0, _ in sorted(segs)[1:])

    tests = [("FEN01", [1935], 1), ("WRI01", [1962], 1), ("TWO01", [1953, 1961], 1)]
    for pid, want, tol in tests:
        got = found[pid]
        good = len(got) == len(want) and all(abs(g - w) <= tol
                                             for g, w in zip(got, want))
        print(("PASS  " if good else "FAIL  ")
              + f"{pid}: breaks {got} (planted {want}, tol +/-{tol})")
        ok &= good
    fp = sum(len(found[f"FLAT{i}"]) for i in range(5))
    good = fp == 0
    print(("PASS  " if good else "FAIL  ") + f"false positives on flat parks: {fp}")
    ok &= good
    print("SMOKE " + ("PASSED" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
