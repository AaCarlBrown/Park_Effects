r"""Stage 2: the core skill model, full-sample fit.

Mirrors the omnibus estimator (two_way_pyfixest.py) exactly:

    woba_value ~ 1 | season + bat_id + pit_id + bat_age + pit_age

fitted with pyfixest on the Stage 1 panel. Filters: IBB excluded, ages 18-45,
players appearing in only one season dropped. Age curves anchored at 28;
season effects mean-anchored; player effects PA-weighted mean-anchored.

Writes:
    output\mu_S.csv            season effects with PA counts
    output\alpha_bat.csv       batter age curve
    output\alpha_pit.csv       pitcher age curve
    output\tau_bat.parquet     batter skills (bat_id, tau, n_pa, n_seasons)
    output\tau_pit.parquet     pitcher skills
    output\skill_summary.txt   PASS / WARN / FAIL lines and the stage gate

Run:  py u_02_skill_model.py       (from C:\SABR_Mesoball\GitHub)
Expect minutes of runtime and several GB of RAM.
"""
import json
import sys
import time
from datetime import datetime

import pandas as pd

from config import EXPECTATIONS_DIR, OUT_DIR

AGE_LO, AGE_HI = 18, 45
ANCHOR_AGE = 28
FORMULA = "woba_value ~ 1 | season + bat_id + pit_id + bat_age + pit_age"

REPORT = []


def log(level, msg):
    line = f"{level:5s} {msg}"
    REPORT.append(line)
    print(line, flush=True)


def load_filtered(max_season=None, path=None):
    """Panel with the omnibus filters applied; optionally truncated for
    the ex-ante expanding-window fits."""
    path = path or (OUT_DIR / "pa_derived.parquet")
    cols = ["season", "bat_id", "pit_id", "woba_value", "iw",
            "bat_age", "pit_age"]
    df = pd.read_parquet(path, columns=cols)
    df = df[df["iw"] != 1]
    df = df.dropna(subset=["woba_value", "bat_age", "pit_age",
                           "bat_id", "pit_id"])
    if max_season is not None:
        df = df[df["season"] <= max_season]
    df["bat_age"] = df["bat_age"].astype(int)
    df["pit_age"] = df["pit_age"].astype(int)
    df["season"] = df["season"].astype(int)
    df = df[(df["bat_age"] >= AGE_LO) & (df["bat_age"] <= AGE_HI)
            & (df["pit_age"] >= AGE_LO) & (df["pit_age"] <= AGE_HI)]
    bat_ns = df.groupby("bat_id")["season"].nunique()
    pit_ns = df.groupby("pit_id")["season"].nunique()
    df = df[df["bat_id"].isin(bat_ns[bat_ns >= 2].index)
            & df["pit_id"].isin(pit_ns[pit_ns >= 2].index)]
    df["bat_id"] = df["bat_id"].astype(str)
    df["pit_id"] = df["pit_id"].astype(str)
    return df


def fit_twoway(df):
    """Fit the model; return dict of anchored fixed-effect Series."""
    import pyfixest as pf
    t0 = time.time()
    mod = pf.feols(FORMULA, data=df)
    secs = time.time() - t0
    fe = {}
    for key, d in mod.fixef().items():
        name = key[2:-1] if key.startswith("C(") and key.endswith(")") else key
        fe[name] = pd.Series(d)

    mu = fe["season"].copy()
    mu.index = mu.index.astype(int)
    mu = mu.sort_index()
    mu -= mu.mean()

    def age_curve(s):
        s = s.copy()
        s.index = s.index.astype(int)
        s = s.sort_index()
        return s - (s.loc[ANCHOR_AGE] if ANCHOR_AGE in s.index else s.mean())

    ab = age_curve(fe["bat_age"])
    ap = age_curve(fe["pit_age"])

    # PA-weighted mean-anchor the player effects
    bat_pa = df.groupby("bat_id").size()
    pit_pa = df.groupby("pit_id").size()
    tb = fe["bat_id"]
    tb -= (tb * bat_pa.reindex(tb.index)).sum() / bat_pa.reindex(tb.index).sum()
    tp = fe["pit_id"]
    tp -= (tp * pit_pa.reindex(tp.index)).sum() / pit_pa.reindex(tp.index).sum()
    return {"mu": mu, "alpha_bat": ab, "alpha_pit": ap,
            "tau_bat": tb, "tau_pit": tp, "fit_seconds": secs}


def main():
    exp = json.loads((EXPECTATIONS_DIR / "expectations_02.json").read_text())
    log("INFO", f"stage 2 full-sample fit {datetime.now().isoformat(timespec='seconds')}")

    df = load_filtered()
    log("INFO", f"rows after filters: {len(df):,}  "
                f"batters {df['bat_id'].nunique():,}  "
                f"pitchers {df['pit_id'].nunique():,}  "
                f"seasons {df['season'].nunique()}")

    r = fit_twoway(df)
    log("INFO", f"pyfixest fit in {r['fit_seconds']:,.0f}s")
    mu, ab, ap = r["mu"], r["alpha_bat"], r["alpha_pit"]

    # ---- gates against the pinned Cliometrica values ----
    def gate(name, val, lo, hi):
        if lo <= val <= hi:
            log("PASS", f"{name} = {val:+.4f} (expected [{lo:+.3f}, {hi:+.3f}])")
        else:
            log("FAIL", f"{name} = {val:+.4f} outside [{lo:+.3f}, {hi:+.3f}]")

    gate("mu_1968", mu.loc[1968], *exp["mu_1968_range"])
    gate("mu_1930", mu.loc[1930], *exp["mu_1930_range"])
    gate("mu_1969_minus_1968", mu.loc[1969] - mu.loc[1968],
         *exp["mu_1969_jump_range"])
    span = mu.max() - mu.min()
    gate("mu_span", span, *exp["mu_span_range"])
    peak = int(ab.idxmax())
    if exp["bat_age_peak_range"][0] <= peak <= exp["bat_age_peak_range"][1]:
        log("PASS", f"batter age curve peaks at {peak}")
    else:
        log("FAIL", f"batter age curve peaks at {peak}, expected "
                    f"{exp['bat_age_peak_range']}")
    log("INFO", f"min mu season: {int(mu.idxmin())} ({mu.min():+.4f}); "
                f"max mu season: {int(mu.idxmax())} ({mu.max():+.4f})")

    # ---- outputs ----
    pa_per_season = df.groupby("season").size()
    pd.DataFrame({"season": mu.index, "mu": mu.values,
                  "n_pa": pa_per_season.reindex(mu.index).fillna(0).astype(int).values}
                 ).to_csv(OUT_DIR / "mu_S.csv", index=False)
    pd.DataFrame({"age": ab.index, "alpha": ab.values}).to_csv(
        OUT_DIR / "alpha_bat.csv", index=False)
    pd.DataFrame({"age": ap.index, "alpha": ap.values}).to_csv(
        OUT_DIR / "alpha_pit.csv", index=False)

    bat_pa = df.groupby("bat_id").size()
    bat_ns = df.groupby("bat_id")["season"].nunique()
    tb = r["tau_bat"]
    pd.DataFrame({"bat_id": tb.index, "tau": tb.values,
                  "n_pa": bat_pa.reindex(tb.index).values,
                  "n_seasons": bat_ns.reindex(tb.index).values}
                 ).to_parquet(OUT_DIR / "tau_bat.parquet", index=False)
    pit_pa = df.groupby("pit_id").size()
    pit_ns = df.groupby("pit_id")["season"].nunique()
    tp = r["tau_pit"]
    pd.DataFrame({"pit_id": tp.index, "tau": tp.values,
                  "n_pa": pit_pa.reindex(tp.index).values,
                  "n_seasons": pit_ns.reindex(tp.index).values}
                 ).to_parquet(OUT_DIR / "tau_pit.parquet", index=False)
    log("INFO", f"wrote mu_S.csv, alpha_*.csv, tau_bat.parquet "
                f"({len(tb):,}), tau_pit.parquet ({len(tp):,})")

    fails = sum(1 for l in REPORT if l.startswith("FAIL"))
    warns = sum(1 for l in REPORT if l.startswith("WARN"))
    log("INFO", f"{'GATE: PASSED' if fails == 0 else 'GATE: BLOCKED'} - "
                f"{fails} FAIL, {warns} WARN")
    (OUT_DIR / "skill_summary.txt").write_text("\n".join(REPORT) + "\n",
                                               encoding="utf-8")
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
