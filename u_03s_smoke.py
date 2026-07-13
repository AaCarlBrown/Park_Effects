r"""Stage 3 smoke test: recover planted park effects and detect a planted
disconnected season, using the exact u_03_park_series code path.

Run:  py u_03s_smoke.py             (from C:\SABR_Mesoball\GitHub)
Runtime seconds.
"""
import sys

import numpy as np
import pandas as pd

import u_03_park_series as m

RNG = np.random.default_rng(7)


def main():
    parks = {"AAA01": ("AAA", 0.015), "BBB01": ("BBB", -0.010),
             "CCC01": ("CCC", 0.000), "DDD01": ("DDD", -0.005)}
    teams = [v[0] for v in parks.values()]
    rows = []
    for season in range(1990, 2000):
        for pid, (home, eff) in parks.items():
            # in the planted "pod" season 1995, AAA/BBB and CCC/DDD play
            # only within their pair
            if season == 1995:
                visitors = {"AAA": ["BBB"], "BBB": ["AAA"],
                            "CCC": ["DDD"], "DDD": ["CCC"]}[home]
            else:
                visitors = [t for t in teams if t != home]
            for vis in visitors:
                n = 400
                rows.append(pd.DataFrame({
                    "park_id": pid, "season": season, "hometeam": home,
                    "visteam": vis,
                    "bat_id": RNG.choice([f"b{i}" for i in range(60)], n),
                    "pit_id": RNG.choice([f"p{i}" for i in range(30)], n),
                    "bat_age": 27, "pit_age": 27,
                    "woba_value": eff + RNG.normal(0.32, 0.25, n)}))
    df = pd.concat(rows, ignore_index=True)

    # zero skill parts: residual reconstruction reduces to demeaned woba
    zero_mu = pd.Series(0.0, index=range(1990, 2000))
    zero_b = pd.Series(0.0, index=[f"b{i}" for i in range(60)])
    zero_p = pd.Series(0.0, index=[f"p{i}" for i in range(30)])
    zero_a = pd.Series(0.0, index=[27])
    r, ok = m.residuals_from_parts(df, zero_mu, zero_b, zero_p, zero_a, zero_a)
    pm = m.park_means(df, r, ok)

    ok_all = True
    # recovery of planted effects (non-pod seasons)
    reg = pm[pm["season"] != 1995].groupby("park_id")["effect"].mean()
    for pid, (_, eff) in parks.items():
        got = reg[pid]
        # effects are league-relative; planted mean is 0 across parks
        want = eff - np.mean([e for _, e in parks.values()])
        line = f"{pid}: recovered {got:+.4f}, planted (centered) {want:+.4f}"
        good = abs(got - want) < 0.004
        print(("PASS  " if good else "FAIL  ") + line)
        ok_all &= good
    # pod season detection
    n95 = pm.loc[pm["season"] == 1995, "n_components"].max()
    nother = pm.loc[pm["season"] != 1995, "n_components"].max()
    good = (n95 == 2) and (nother == 1)
    print(("PASS  " if good else "FAIL  ")
          + f"components: 1995 -> {n95} (want 2), others -> {nother} (want 1)")
    ok_all &= good

    print("SMOKE " + ("PASSED" if ok_all else "FAILED"))
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
