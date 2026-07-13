r"""Stage 2 smoke test: recover planted skills from synthetic data.

Builds a synthetic panel with known mu_S, tau_bat, tau_pit, and age curves,
fits it with the exact Stage 2 code path (imported from u_02_skill_model),
and verifies recovery. Run BEFORE u_02_skill_model.py touches real data.

Run:  py u_02s_smoke.py            (from C:\SABR_Mesoball\GitHub)
Runtime well under a minute.
"""
import sys

import numpy as np
import pandas as pd

import u_02_skill_model as m
from config import OUT_DIR

RNG = np.random.default_rng(11)
N_BAT, N_PIT, SEASONS = 400, 200, range(2000, 2010)
PA_PER_BAT_SEASON = 350
NOISE_SD = 0.25       # real per-PA noise is ~0.55; smaller here so the
                      # recovery gate is sharp at synthetic sample sizes


def build():
    tau_b = pd.Series(RNG.normal(0, 0.030, N_BAT),
                      index=[f"b{i:04d}" for i in range(N_BAT)])
    tau_p = pd.Series(RNG.normal(0, 0.020, N_PIT),
                      index=[f"p{i:04d}" for i in range(N_PIT)])
    mu = pd.Series(RNG.normal(0, 0.012, len(SEASONS)), index=list(SEASONS))
    mu -= mu.mean()
    ages_b = pd.Series(RNG.integers(20, 33, N_BAT), index=tau_b.index)
    ages_p = pd.Series(RNG.integers(20, 33, N_PIT), index=tau_p.index)

    def a_bat(a): return -0.0006 * (a - 27) ** 2
    def a_pit(a): return -0.0004 * (a - 26) ** 2

    rows = []
    for s in SEASONS:
        for b in tau_b.index:
            ab = int(ages_b[b] + (s - 2000))
            if not (m.AGE_LO <= ab <= m.AGE_HI):
                continue
            pits = RNG.choice(tau_p.index, PA_PER_BAT_SEASON)
            ap_ = ages_p[pits].values + (s - 2000)
            y = (mu[s] + tau_b[b] + tau_p[pits].values
                 + a_bat(ab) + a_pit(ap_)
                 + RNG.normal(0, NOISE_SD, PA_PER_BAT_SEASON))
            rows.append(pd.DataFrame({
                "season": s, "bat_id": b, "pit_id": pits,
                "bat_age": ab, "pit_age": ap_.astype(int),
                "woba_value": y, "iw": 0}))
    df = pd.concat(rows, ignore_index=True)
    df = df[(df["pit_age"] >= m.AGE_LO) & (df["pit_age"] <= m.AGE_HI)]
    return df, tau_b, tau_p, mu


def main():
    df, tau_b, tau_p, mu_true = build()
    print(f"synthetic panel: {len(df):,} rows")
    r = m.fit_twoway(df)
    lines = [f"synthetic rows {len(df):,}, fit {r['fit_seconds']:.1f}s"]
    ok = True

    def check(name, got, want, tol_corr):
        nonlocal ok
        joined = pd.concat([got, want], axis=1, keys=["hat", "true"]).dropna()
        c = joined["hat"].corr(joined["true"])
        line = f"{name}: corr(hat, true) = {c:.4f} (need >= {tol_corr})"
        lines.append(line)
        print(("PASS  " if c >= tol_corr else "FAIL  ") + line)
        if c < tol_corr:
            ok = False

    mu_hat = r["mu"]
    mu_hat.index = mu_hat.index.astype(int)
    check("mu_S", mu_hat, mu_true, 0.995)
    check("tau_bat", r["tau_bat"], tau_b, 0.95)
    check("tau_pit", r["tau_pit"], tau_p, 0.95)
    peak = int(r["alpha_bat"].idxmax())
    line = f"bat age peak {peak} (true 27; accept 26-28)"
    lines.append(line)
    print(("PASS  " if 26 <= peak <= 28 else "FAIL  ") + line)
    ok = ok and 26 <= peak <= 28

    (OUT_DIR / "smoke_02.txt").write_text("\n".join(lines) + "\n",
                                          encoding="utf-8")
    print("SMOKE " + ("PASSED" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
