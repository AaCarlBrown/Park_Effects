r"""Stage 9: validation consolidation.

Collects every stage's gate outcomes and every documented expectation
revision into one report - the evidence behind the manuscript's Validation
section - and enumerates the three error models used across the pipeline
(filling the appendix TK):

  1. analytic sampling variance: per-PA residual variance / n, used for
     park-season effects, splits, and fielding shares (binomial form);
  2. serial-scatter variance: robust (median-based) first-difference and
     season-to-season scatter estimates, immune to the true breaks, used
     by the changepoint screen, the wind regime SEs, and the conversion
     diagnostic;
  3. cluster-robust sandwich (clustered by park), used for the lineup-fit
     regression.

Writes output\validation_report.txt.
Run:  py u_09_validate.py           (from C:\SABR_Mesoball\GitHub)
"""
import json
from datetime import datetime

from config import EXPECTATIONS_DIR, OUT_DIR

SUMMARIES = [
    ("Stage 0 inputs", "setup_report.txt"),
    ("Stage 1 derivations", "derive_summary.txt"),
    ("Stage 2 skill model", "skill_summary.txt"),
    ("Stage 3 park series", "park_series_summary.txt"),
    ("Stage 3b breaks", "breaks_summary.txt"),
    ("Stage 4 conventional", "conventional_summary.txt"),
    ("Stage 4b contest", "contest_summary.txt"),
    ("Stage 5 wind", "wind_summary.txt"),
    ("Stage 6 splits", "splits_summary.txt"),
    ("Stage 7 ledger", "ledger_summary.txt"),
    ("Stage 8 fielding", "fielding_summary.txt"),
    ("Stage 8b lineup fit", "alloc_summary.txt"),
]

ARTIFACTS = """Artifacts caught by their signatures during the rerun
(each documented in the Expectations revision blocks):
- IBB null wOBA values: nulls coincided exactly with the iw flag.
- Roster-hand semantics: switch hitters carried 'B' on every PA (Mantle
  spot check); per-PA side derived from opposing pitcher's hand.
- League components: two disconnected leagues before interleague play;
  three 2020 pods; caught by union-find, encoded as league-relative
  effects.
- Draft ledger/handedness magnitudes inflated by low-K shrinkage in the
  original analysis; rerun scales are roughly half to two-thirds.
- Own-park absorption in fielding shift-share (Rogers Centre cells
  cancelling to exactly zero); fixed leave-own-park-out.
- firstf sparsity (46% coverage) degenerating out-conversion to 1.0;
  metric replaced with putout shares.
- Park-level putout-attribution coverage (deduced game accounts;
  Sportsman's suppressed at five positions at once); fixed with
  compositional shares.
- Wind-station availability split (pre-1973 files under USAF 999999).
"""


def main():
    lines = [f"validation report {datetime.now().isoformat(timespec='seconds')}",
             ""]
    stale = []
    for label, fn in SUMMARIES:
        p = OUT_DIR / fn
        if not p.exists():
            lines.append(f"{label:24s} MISSING ({fn})")
            stale.append(label)
            continue
        txt = p.read_text(encoding="utf-8").splitlines()
        gate = next((l for l in reversed(txt) if "GATE:" in l), "no gate line")
        fails = sum(1 for l in txt if l.startswith("FAIL"))
        warns = sum(1 for l in txt if l.startswith("WARN"))
        status = gate.split("GATE:")[-1].strip()
        lines.append(f"{label:24s} {status:32s} ({fails} FAIL / {warns} WARN)")
        if "BLOCKED" in gate:
            stale.append(label)
    lines.append("")
    lines.append("Expectation revisions on record:")
    for f in sorted(EXPECTATIONS_DIR.glob("expectations_*.json")):
        try:
            j = json.loads(f.read_text())
        except Exception:
            continue
        for k, v in j.get("revisions", {}).items():
            lines.append(f"  [{f.stem.replace('expectations_', 'stage ')}/"
                         f"{k}] {v[:140]}...")
    lines.append("")
    lines.append(ARTIFACTS)
    lines.append("Error models used across the pipeline (appendix TK fill):")
    lines.append("  1. analytic sampling variance (per-PA residual variance/n)")
    lines.append("  2. robust serial-scatter variance (median first-difference"
                 " / season scatter; break-immune)")
    lines.append("  3. cluster-robust sandwich, clustered by park")
    if stale:
        lines.append("")
        lines.append(f"STALE/BLOCKED summaries needing a rerun for a green "
                     f"record: {', '.join(stale)}")
    report = "\n".join(lines) + "\n"
    (OUT_DIR / "validation_report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
