r"""Stage 1b: diagnose the three Stage 1 anomalies before re-gating.

1. Profile the 99,027 null woba_value rows: which outcome flags do they carry?
2. Wrigley game counts under every candidate definition, to find the paper's 9,442.
3. bat_hand semantics: value counts and switch-hitter crosstabs.

Read-only; writes output\diagnose_01b.txt.

Run:  py u_01b_diagnose.py         (from C:\SABR_Mesoball\GitHub)
"""
from datetime import datetime

import pandas as pd

from config import OUT_DIR

LINES = []


def say(msg=""):
    LINES.append(str(msg))
    print(msg)


def main():
    say(f"stage 1b diagnostics {datetime.now().isoformat(timespec='seconds')}")
    panel = pd.read_parquet(OUT_DIR / "pa_derived.parquet")

    # ---- 1. null woba_value profile ----
    nul = panel[panel["woba_value"].isna()]
    say(f"\n--- null woba_value rows: {len(nul):,} ---")
    flags = ["pa", "single", "double", "triple", "hr", "sh", "sf", "hbp",
             "walk", "k", "xi", "roe", "fc", "othout", "iw"]
    say("flag sums among null rows (vs full panel):")
    for f in flags:
        if f in nul.columns:
            say(f"  {f:7s} null-rows {int(nul[f].sum()):>9,}   "
                f"panel {int(panel[f].sum()):>10,}")
    say("null rows by decade:")
    say((nul["season"] // 10 * 10).value_counts().sort_index().to_string())

    # ---- 2. Wrigley counts, every plausible definition ----
    say("\n--- Wrigley (CHI11) game counts ---")
    wr = panel[panel["park_id"] == "CHI11"]
    combos = [
        ("all games, all seasons", None, None),
        ("all games since 1950", 1950, None),
        ("day games, all seasons", None, "D"),
        ("day games since 1950", 1950, "D"),
        ("night games, all seasons", None, "N"),
    ]
    for label, y0, dn in combos:
        m = wr
        if y0:
            m = m[m["season"] >= y0]
        if dn:
            m = m[m["daynight"] == dn]
        say(f"  {label:28s} {m['game_id'].nunique():>7,}")
    say("Wrigley games by decade and day/night:")
    g = (wr.drop_duplicates("game_id")
           .assign(decade=lambda d: d["season"] // 10 * 10)
           .groupby(["decade", "daynight"])["game_id"].count().unstack(fill_value=0))
    say(g.to_string())

    # ---- 3. bat_hand semantics ----
    say("\n--- bat_hand semantics ---")
    say("bat_bats value counts:")
    say(panel["bat_bats"].value_counts(dropna=False).to_string())
    say("bat_hand value counts:")
    say(panel["bat_hand"].value_counts(dropna=False).to_string())
    sw = panel[panel["bat_bats"] == "B"]
    say(f"switch-hitter (bat_bats==B) PA: {len(sw):,}")
    say("crosstab bat_hand x pit_hand for switch hitters:")
    say(pd.crosstab(sw["bat_hand"], sw["pit_hand"]).to_string())
    say("crosstab bat_hand x bat_bats (full panel, top hands):")
    say(pd.crosstab(panel["bat_hand"], panel["bat_bats"]).to_string())
    # a famous switch hitter as a spot check, if present
    for pid in ["mantm101", "rosep001", "smito001"]:
        s = panel[panel["bat_id"] == pid]
        if len(s):
            say(f"spot check {pid}: bat_bats={s['bat_bats'].mode().iat[0]}, "
                f"bat_hand vs pit_hand:")
            say(pd.crosstab(s["bat_hand"], s["pit_hand"]).to_string())
            break

    (OUT_DIR / "diagnose_01b.txt").write_text("\n".join(LINES) + "\n",
                                              encoding="utf-8")
    print(f"\nwrote {OUT_DIR / 'diagnose_01b.txt'}")


if __name__ == "__main__":
    main()
