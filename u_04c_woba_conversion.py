r"""Stage 4c (diagnostic): park-lifetime wOBA factors vs run factors, by
regime - does each park convert expected runs into actual runs at its own
stable rate?

Pre-registered hypothesis (Aaron, 2026-07-13): yes - mesoball predicts
wOBA better, yet actual-runs methods keep an edge in the run-factor
currency, which requires park-specific run conversion that expected runs
miss. Test: per park-regime (regimes from the Stage 3b screen), compute

  woba_eff : PA-weighted mean raw park wOBA differential (FG-style
             multi-year average of the raw differential, scaled-wOBA units)
  exp_run  : woba_eff converted to actual-runs units by the season's
             wOBAScale (wOBA weights are inflated by ~1.25 so league wOBA
             reads like OBP; expected runs = wOBA diff / scale)
  run_eff  : mean of (TF - 1) x league runs/PA over the same seasons
             (actual-runs units)
  conv     : run_eff - exp_run   (the park's run-conversion residual)

REVISION (2026-07-13, author-caught): the first run computed
conv = run_eff - woba_eff, mixing scaled-wOBA units with actual-run
units; that embeds a mechanical -(1 - 1/scale) x woba_eff term
(about -20% of the park's wOBA effect), inflating residuals for strong
parks and the true-SD estimate. Fixed by dividing each season's wOBA
differential by that season's wOBAScale before differencing.

If conv has positive true variance across park-regimes (sampling variance
subtracted), the conversion trait is real and a conversion-augmented
mesoball predictor is the natural follow-up.

Writes output\park_run_conversion.csv and output\conversion_summary.txt.
Run:  py u_04c_woba_conversion.py    (from C:\SABR_Mesoball\GitHub)
"""
from datetime import datetime

import numpy as np
import pandas as pd

from config import DATA_DIR, OUT_DIR

MIN_SEASONS = 8
LINES = []


def say(msg=""):
    LINES.append(str(msg))
    print(msg, flush=True)


def main():
    say(f"stage 4c conversion {datetime.now().isoformat(timespec='seconds')}")

    # raw park wOBA differentials (as in the Stage 4b target)
    pa = pd.read_parquet(OUT_DIR / "pa_derived.parquet",
                         columns=["park_id", "season", "woba_value"])
    hind = pd.read_csv(OUT_DIR / "park_by_season_hindsight.csv")
    raw = (pa.dropna(subset=["woba_value"])
           .groupby(["park_id", "season"])["woba_value"]
           .agg(["mean", "size"]).reset_index()
           .rename(columns={"mean": "raw_woba", "size": "n"}))
    raw = raw.merge(hind[["park_id", "season", "component"]],
                    on=["park_id", "season"], how="inner")
    cmean = (raw.groupby(["season", "component"])
             .apply(lambda d: np.average(d["raw_woba"], weights=d["n"]),
                    include_groups=False).rename("cmean").reset_index())
    raw = raw.merge(cmean, on=["season", "component"])
    raw["woba_diff"] = raw["raw_woba"] - raw["cmean"]

    # era-specific wOBA scale: expected actual runs per point of wOBA
    ww = pd.read_csv(DATA_DIR / "wOBA_weights.csv", encoding="utf-8-sig")
    scale = ww.set_index("Season")["wOBAScale"]
    raw["exp_run"] = raw["woba_diff"] / raw["season"].map(scale)

    # run effects from the game-log team factors
    tf = pd.read_csv(OUT_DIR / "team_season_factors.csv")
    pyt = (tf.groupby(["park_id", "season"])
           .apply(lambda d: np.average(d["tf"], weights=d["home_g"]),
                  include_groups=False).rename("tf").reset_index())
    pred = pd.read_csv(OUT_DIR / "park_factor_predictions.csv")
    rpp = pred.drop_duplicates("season").set_index("season")["league_rpp"]
    pyt["run_eff"] = (pyt["tf"] - 1) * pyt["season"].map(rpp)

    d = raw.merge(pyt, on=["park_id", "season"], how="inner")
    reg = pd.read_csv(OUT_DIR / "park_regimes.csv")
    d = d.merge(reg, on="park_id")
    d = d[(d["season"] >= d["start"]) & (d["season"] <= d["end"])]

    rows = []
    for (pid, s0, e0), g in d.groupby(["park_id", "start", "end"]):
        if g["season"].nunique() < MIN_SEASONS:
            continue
        woba_eff = np.average(g["woba_diff"], weights=g["n"])
        exp_run = np.average(g["exp_run"], weights=g["n"])
        run_eff = g["run_eff"].mean()
        # season-to-season scatter gives honest SEs for both
        se_w = g["exp_run"].std() / np.sqrt(len(g))
        se_r = g["run_eff"].std() / np.sqrt(len(g))
        rows.append({"park_id": pid, "start": int(s0), "end": int(e0),
                     "n_seasons": g["season"].nunique(),
                     "woba_eff_pts": woba_eff * 1000,
                     "exp_run_eff_pts": exp_run * 1000,
                     "run_eff_pts": run_eff * 1000,
                     "conv_pts": (run_eff - exp_run) * 1000,
                     "se_pts": np.hypot(se_w, se_r) * 1000})
    out = pd.DataFrame(rows)
    try:
        pc = pd.read_csv(OUT_DIR.parent / "data" / "parkcode.txt", dtype=str,
                         keep_default_na=False)
        pc.columns = [c.upper() for c in pc.columns]
        out["name"] = out["park_id"].map(dict(zip(pc["PARKID"], pc["NAME"])))
    except Exception:
        out["name"] = out["park_id"]
    out.to_csv(OUT_DIR / "park_run_conversion.csv", index=False)

    say(f"park-regimes with >= {MIN_SEASONS} seasons: {len(out)}")
    say(f"corr(run_eff, woba_eff) across park-regimes: "
        f"{out['run_eff_pts'].corr(out['woba_eff_pts']):.3f}")
    v_obs = np.average(out["conv_pts"] ** 2, weights=1 / out["se_pts"] ** 2)
    v_samp = (out["se_pts"] ** 2).mean()
    sd_true = np.sqrt(max(v_obs - v_samp, 0.0))
    say(f"conversion residual: observed SD {out['conv_pts'].std():.1f} pts, "
        f"mean sampling SE {np.sqrt(v_samp):.1f}, TRUE SD {sd_true:.1f} pts")
    say(f"parks-regimes with |conv|/se >= 2: "
        f"{(out['conv_pts'].abs() / out['se_pts'] >= 2).sum()} of {len(out)}"
        f" (chance ~{0.046 * len(out):.0f})")
    out["z"] = out["conv_pts"] / out["se_pts"]
    top = out.reindex(out["z"].abs().sort_values(ascending=False).index)
    say("\nlargest conversion residuals (runs materialize above/below "
        "expectation):")
    say(top.head(8)[["name", "start", "end", "woba_eff_pts",
                     "exp_run_eff_pts", "run_eff_pts",
                     "conv_pts", "z"]].round(1).to_string(index=False))
    say("\nverdict: positive TRUE SD supports the hypothesis that parks "
        "convert wOBA to runs at stable idiosyncratic rates; a "
        "conversion-augmented mesoball run-factor predictor is then the "
        "natural upgrade (not applied retroactively to the locked Stage 4 "
        "contest).")
    (OUT_DIR / "conversion_summary.txt").write_text("\n".join(LINES) + "\n",
                                                    encoding="utf-8")


if __name__ == "__main__":
    main()
