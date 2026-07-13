r"""Stage 3f: regenerate manuscript Figures 1-3 from rerun output, 300 dpi.

Dots  : conventional 1970s-method within-team differential, computed from
        the Retrosheet game logs (tenant's runs+runs-against per PA at home
        minus the same tenant on the road, home-game weighted).
Line  : the real-time mesoball series (skills known through the rated
        season), each season EB-shrunk toward the mean of the regime
        detectable at that time (changepoint screen on the series through
        that season only) - fully ex-ante, no future information.
Dashes: regime means from the changepoint screen (Fenway 1937/1986,
        Wrigley 1962/1992).

Writes to output\figures\: fig1_fenway.png, fig2_wrigley.png,
fig3_eleven_year.png, each also as *_gray.png for BRJ print, and
figure_notes.txt recording the regimes drawn.

Run:  py u_03f_figures.py           (from C:\SABR_Mesoball\GitHub)
Requires matplotlib (pip install matplotlib if missing).
"""
import re
from datetime import datetime

import numpy as np
import pandas as pd

from config import GAMELOG_DIR, OUT_DIR
from u_03b_breaks import estimate_sigma2

FIG_DIR = OUT_DIR / "figures"
PARKS = {"BOS07": ("Fenway Park", "fig1_fenway"),
         "CHI11": ("Wrigley Field", "fig2_wrigley")}


def conventional_series(park_id):
    """1970s-method dots: tenant runs/PA at home minus tenant on the road."""
    pa_per_game = (pd.read_parquet(OUT_DIR / "pa_derived.parquet",
                                   columns=["game_id"])
                   .groupby("game_id").size().rename("pa"))
    rows = []
    for f in sorted(GAMELOG_DIR.iterdir()):
        m = re.match(r"(?i)gl(\d{4})\.txt$", f.name)
        if not m:
            continue
        gl = pd.read_csv(f, header=None, usecols=[0, 1, 3, 6, 9, 10, 16],
                         names=["date", "gamenum", "visteam", "hometeam",
                                "vis_runs", "home_runs", "park"],
                         dtype={"date": str, "gamenum": str, "visteam": str,
                                "hometeam": str, "park": str},
                         keep_default_na=False)
        gl["season"] = int(m.group(1))
        rows.append(gl)
    gl = pd.concat(rows, ignore_index=True)
    gl["game_id"] = gl["hometeam"] + gl["date"] + gl["gamenum"]
    gl["runs"] = pd.to_numeric(gl["vis_runs"], errors="coerce") + \
        pd.to_numeric(gl["home_runs"], errors="coerce")
    gl = gl.merge(pa_per_game, left_on="game_id", right_index=True, how="inner")

    out = []
    tenants = gl[gl["park"] == park_id].groupby(
        ["season", "hometeam"]).size().reset_index()[["season", "hometeam"]]
    for season, team in tenants.itertuples(index=False):
        g = gl[gl["season"] == season]
        home = g[(g["park"] == park_id) & (g["hometeam"] == team)]
        road = g[(g["visteam"] == team) & (g["park"] != park_id)]
        if len(home) < 10 or len(road) < 10:
            continue
        d = (home["runs"].sum() / home["pa"].sum()
             - road["runs"].sum() / road["pa"].sum())
        out.append({"season": season, "diff": d, "games": len(home)})
    o = pd.DataFrame(out)
    return (o.groupby("season")
            .apply(lambda d: np.average(d["diff"], weights=d["games"]),
                   include_groups=False).rename("diff").reset_index())


def exante_shrunk(park_id, sigma2, regimes, min_seg=4, thresh=12.0):
    """Ex-ante line: season t's real-time estimate shrunk toward the mean
    of the current regime as detectable AT THE TIME - changepoint screen on
    the series through t only, then EB shrinkage toward the mean since the
    last break found. No information from later seasons enters season t's
    plotted value. min_seg/thresh match the Stage 4 real-time predictor
    (expectations_04.json). The dashed regime means remain the full-sample
    (hindsight) regimes from u_03b, drawn as separate elements.
    """
    from u_03b_breaks import segment
    rt = pd.read_csv(OUT_DIR / "park_by_season_realtime.csv")
    rt = rt[rt["park_id"] == park_id].sort_values("season")
    e_all = rt["effect"].values
    n_all = rt["n_pa"].values
    seasons = rt["season"].values
    rows = []
    for i in range(len(rt)):
        e, n, ss = e_all[:i + 1], n_all[:i + 1], seasons[:i + 1]
        if len(ss) >= 2 * min_seg:
            segs = sorted(segment(e, n / sigma2, ss, min_seg, thresh))
        else:
            segs = [(ss[0], ss[-1])]
        # within-regime true variance, estimated from data through t only
        dev2, samp = [], []
        for s0, s1 in segs:
            m0 = (ss >= s0) & (ss <= s1)
            mu = np.average(e[m0], weights=n[m0])
            dev2.append((e[m0] - mu) ** 2 * n[m0])
            samp.append(sigma2 / n[m0])
        var_true = max(np.concatenate(dev2).sum() / n.sum()
                       - np.concatenate(samp).mean(), 0.0)
        s0 = segs[-1][0]
        mask = ss >= s0
        m = np.average(e[mask], weights=n[mask])
        w = var_true / (var_true + sigma2 / n_all[i])
        rows.append({"season": seasons[i], "shrunk": m + w * (e_all[i] - m)})
    line = pd.DataFrame(rows)
    rg = regimes[regimes["park_id"] == park_id]
    spans = pd.concat([pd.DataFrame(
        {"season": range(int(r["start"]), int(r["end"]) + 1),
         "regime_mean": r["mean"], "start": r["start"], "end": r["end"]})
        for _, r in rg.iterrows()], ignore_index=True)
    return line.merge(spans, on="season", how="left").dropna(subset=["start"])


def draw(park_id, name, stem, conv, line, gray=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dot_c = "0.45" if gray else "#4d7f96"
    line_c = "0.05" if gray else "#c8404f"
    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=300)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.scatter(conv["season"], conv["diff"], s=12, color=dot_c, zorder=3,
               label="Conventional single-season differential (game logs)")
    ax.plot(line["season"], line["shrunk"], color=line_c, lw=1.0, alpha=0.85,
            zorder=4, label="Mesoball real-time estimate (no future information)")
    for (s0, e0, m0) in line[["start", "end", "regime_mean"]].drop_duplicates().itertuples(index=False):
        ax.hlines(m0, s0, e0, color="0.1", ls="--", lw=1.2, zorder=5)
    y0, y1 = int(line["season"].min()), int(line["season"].max())
    ax.set_title(f"{name}, {y0}–{y1}: conventional differential vs "
                 f"the plate-appearance model", fontsize=9)
    ax.set_ylabel("runs/PA vs league-average other parks", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{stem}{'_gray' if gray else ''}.png")
    plt.close(fig)


def draw_fig3(data, gray=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ma_c = "0.55" if gray else "#7a8fa6"
    line_c = "0.05" if gray else "#c8404f"
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 7.6), dpi=300, sharex=True)
    for ax, (pid, (name, _)) in zip(axes, PARKS.items()):
        conv, line = data[pid]
        ma = conv.set_index("season")["diff"].rolling(11, center=True,
                                                      min_periods=6).mean()
        ax.axhline(0, color="0.6", lw=0.8)
        ax.plot(ma.index, ma.values, color=ma_c, lw=1.5,
                label="11-yr centered moving average, conventional")
        ax.plot(line["season"], line["shrunk"], color=line_c, lw=1.0,
                alpha=0.85, label="Mesoball real-time estimate (no future information)")
        for (s0, e0, m0) in line[["start", "end", "regime_mean"]].drop_duplicates().itertuples(index=False):
            ax.hlines(m0, s0, e0, color="0.1", ls="--", lw=1.2)
        ax.set_title(name, fontsize=10, loc="left")
        ax.set_ylabel("runs/PA vs league-average other parks", fontsize=8)
        ax.legend(fontsize=7, loc="lower right")
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"fig3_eleven_year{'_gray' if gray else ''}.png")
    plt.close(fig)


def draw_wrigley_11yr(data, gray=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ma_c = "0.55" if gray else "#7a8fa6"
    line_c = "0.05" if gray else "#c8404f"
    conv, line = data["CHI11"]
    ma = conv.set_index("season")["diff"].rolling(11, center=True,
                                                  min_periods=6).mean()
    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=300)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.plot(ma.index, ma.values, color=ma_c, lw=1.5,
            label="11-yr centered moving average, conventional")
    ax.plot(line["season"], line["shrunk"], color=line_c, lw=1.0,
            alpha=0.85, label="Mesoball real-time estimate (no future information)")
    for (s0, e0, m0) in line[["start", "end", "regime_mean"]].drop_duplicates().itertuples(index=False):
        ax.hlines(m0, s0, e0, color="0.1", ls="--", lw=1.7)
    ax.set_title("Wrigley Field: 11-year averaging vs the "
                 "plate-appearance model", fontsize=9)
    ax.set_ylabel("runs/PA vs league-average other parks", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"fig_wrigley_11yr{'_gray' if gray else ''}.png")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    regimes = pd.read_csv(OUT_DIR / "park_regimes.csv")
    hind = pd.read_csv(OUT_DIR / "park_by_season_hindsight.csv")
    sigma2 = estimate_sigma2(hind[hind["park_id"].isin(PARKS)])
    notes = [f"stage 3f {datetime.now().isoformat(timespec='seconds')}",
             f"sigma2 {sigma2:.4f}"]
    data = {}
    for pid, (name, stem) in PARKS.items():
        conv = conventional_series(pid)
        line = exante_shrunk(pid, sigma2, regimes)
        data[pid] = (conv, line)
        for gray in (False, True):
            draw(pid, name, stem, conv, line, gray)
        rg = regimes[regimes["park_id"] == pid]
        notes.append(f"{pid} regimes: " + "; ".join(
            f"{int(r.start)}-{int(r.end)} {r['mean']*1000:+.1f} pts"
            for _, r in rg.iterrows()))
        notes.append(f"{pid} conv-vs-model corr: "
                     f"{conv.merge(line, on='season')[['diff', 'shrunk']].corr().iloc[0, 1]:.3f}")
    for gray in (False, True):
        draw_fig3(data, gray)
        draw_wrigley_11yr(data, gray)
    (FIG_DIR / "figure_notes.txt").write_text("\n".join(notes) + "\n",
                                              encoding="utf-8")
    print("\n".join(notes))
    print(f"wrote 6 figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
