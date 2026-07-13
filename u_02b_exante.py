r"""Stage 2b: ex-ante (expanding window) skill fits, one per rating season.

For each rating season Y from 1911 to 2025, refits the Stage 2 model on
seasons <= Y only, and saves the skills as known at the end of Y. This is
the "information available at the time" series behind the paper's real-time
park ratings and the ex-ante career ledger.

Checkpointed: a season whose output already exists is skipped, so the run
can be interrupted and resumed freely. Expect an overnight run.

    py u_02b_exante.py                    (all seasons)
    py u_02b_exante.py --start 1980 --end 1990

Writes, per rating season Y:
    output\exante\tau_bat_Y.parquet   (bat_id, tau)
    output\exante\tau_pit_Y.parquet
    output\exante\mu_alpha_Y.parquet  (season/age effects as of Y)
and appends timing to output\exante\progress_log.txt.
"""
import argparse
import time
from datetime import datetime

import pandas as pd

import u_02_skill_model as m
from config import OUT_DIR

EXANTE_DIR = OUT_DIR / "exante"


def run_season(y):
    df = m.load_filtered(max_season=y)
    if df["season"].nunique() < 2:
        return f"{y}: skipped (fewer than 2 seasons of data)"
    t0 = time.time()
    r = m.fit_twoway(df)
    tb, tp = r["tau_bat"], r["tau_pit"]
    pd.DataFrame({"bat_id": tb.index, "tau": tb.values}).to_parquet(
        EXANTE_DIR / f"tau_bat_{y}.parquet", index=False)
    pd.DataFrame({"pit_id": tp.index, "tau": tp.values}).to_parquet(
        EXANTE_DIR / f"tau_pit_{y}.parquet", index=False)
    mu, ab, ap = r["mu"], r["alpha_bat"], r["alpha_pit"]
    parts = [pd.DataFrame({"kind": "mu", "key": mu.index, "value": mu.values}),
             pd.DataFrame({"kind": "alpha_bat", "key": ab.index, "value": ab.values}),
             pd.DataFrame({"kind": "alpha_pit", "key": ap.index, "value": ap.values})]
    pd.concat(parts, ignore_index=True).to_parquet(
        EXANTE_DIR / f"mu_alpha_{y}.parquet", index=False)
    return (f"{y}: {len(df):,} rows, {df['bat_id'].nunique():,} batters, "
            f"fit {time.time() - t0:,.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1911)
    ap.add_argument("--end", type=int, default=2025)
    args = ap.parse_args()
    EXANTE_DIR.mkdir(parents=True, exist_ok=True)
    logf = EXANTE_DIR / "progress_log.txt"

    for y in range(args.start, args.end + 1):
        if (EXANTE_DIR / f"tau_bat_{y}.parquet").exists():
            print(f"{y}: exists, skipped")
            continue
        msg = run_season(y)
        stamp = datetime.now().isoformat(timespec="seconds")
        with open(logf, "a", encoding="utf-8") as f:
            f.write(f"{stamp} {msg}\n")
        print(msg, flush=True)
    print("done")


if __name__ == "__main__":
    main()
