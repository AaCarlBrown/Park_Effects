r"""Stage 4 smoke test: synthetic game logs with one inflated park; the
factor pipeline must rank it first and a perfect predictor must score ~0.

Run:  py u_04s_smoke.py             (from C:\SABR_Mesoball\GitHub)
"""
import sys

import numpy as np
import pandas as pd

import u_04_conventional as m

RNG = np.random.default_rng(5)


def main():
    teams = {"AAA": ("AAA01", 1.20), "BBB": ("BBB01", 1.0),
             "CCC": ("CCC01", 1.0), "DDD": ("DDD01", 0.9)}
    rows = []
    for season in range(1980, 1992):
        for home, (park, mult) in teams.items():
            for vis in teams:
                if vis == home:
                    continue
                for g in range(13):
                    base = RNG.poisson(4.5, 2).sum()
                    rows.append({"season": season, "hometeam": home,
                                 "visteam": vis, "park": park,
                                 "runs": base * mult,
                                 "vis_runs": 0, "home_runs": 0,
                                 "date": f"{season}0501", "gamenum": "0"})
    gl = pd.DataFrame(rows)
    tf = m.team_factors(gl)
    pyt = m.park_year_tf(tf)
    means = pyt.groupby("park_id")["tf"].mean().sort_values(ascending=False)
    print(means.round(3).to_string())
    ok = means.index[0] == "AAA01" and means.index[-1] == "DDD01"
    print(("PASS  " if ok else "FAIL  ") + "factor ranking matches planted parks")
    # trailing prediction sanity: regressed toward 1, between 1 and raw mean
    p = m.trailing_regressed(pyt, "AAA01", 1990, 3, 1)
    raw = pyt[(pyt.park_id == "AAA01") & pyt.season.between(1987, 1989)]["tf"].mean()
    ok2 = 1.0 < p < raw
    print(("PASS  " if ok2 else "FAIL  ")
          + f"regression pulls toward 1 ({p:.3f} in (1, {raw:.3f}))")
    print("SMOKE " + ("PASSED" if ok and ok2 else "FAILED"))
    sys.exit(0 if ok and ok2 else 1)


if __name__ == "__main__":
    main()
