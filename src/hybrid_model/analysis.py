"""Scoring the hybrid runs against ground truth and BCDJ's own simulated baseline.

    python -m src.hybrid_model.analysis                 # writes all three figures
    python -m src.hybrid_model.analysis --view auc       # just one

Three figures, one question each:

**adoption_by_design** -- does the hybrid's overall adoption rate land anywhere
near the village's real take-up? Every design so far overshoots it, some by
4x, and the instruction axis (D) is what mostly decides by how much.

**adoption_leader_split** -- ground truth has leaders adopt *less* than
non-leaders (13.6% vs 25.9%, because leaders are BCDJ's covariate-only logit
population and the MFI's direct pitch is a poor predictor of who they picked).
Does any design reproduce that reversal, or does every one of them get it
backward?

**household_auc** -- collapsed to a single population-level rate, a design can
look equally wrong for two very different reasons: uniform noise, or the
right *ranking* of households at the wrong *level*. AUC on each household's
share-of-replicates-adopted, against its true outcome, tells the two apart.

What "ground truth" and "BCDJ baseline" mean here
---------------------------------------------------
Ground truth is `ground_truth_rates` -- BCDJ's own published take-up, on the
giant-component denominator every run shares. It is one number, not a
distribution: village 6 only happened once.

The BCDJ baseline is `bcdj_run` under the pooled 43-village logit
(`fit_betas`), replicated `BASELINE_SEEDS` times on village 6's real network.
Unlike ground truth it *is* a distribution -- the logit is stochastic -- so it
plots as a mean and a std band, the same shape a design with enough
replicates would have. It is the number the hybrid is actually meant to beat:
matching ground truth better than the logit does is the only way the LLM
substitution could be said to add anything the identity model does not
already have identified from BCDJ's fitted covariates alone.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are written, not shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

try:
    from .. import data_loader as dl
    from .game_master import (
        LLMs,
        OUTPUT_DIR,
        VILLAGE,
        adjacency_matrix,
        auc,
        bcdj_run,
        build_village,
        covariates,
        fit_betas,
        ground_truth_rates,
        logit_p,
        population,
    )
except ImportError:  # running as a script, not a package
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import data_loader as dl  # type: ignore[no-redef]
    from game_master import (  # type: ignore[no-redef]
        LLMs,
        OUTPUT_DIR,
        VILLAGE,
        adjacency_matrix,
        auc,
        bcdj_run,
        build_village,
        covariates,
        fit_betas,
        ground_truth_rates,
        logit_p,
        population,
    )

# House palette (`src/plots.py`, `src/pilot/adoption_rate_pilot.py`) -- kept
# identical rather than re-derived, so a figure from this module sits next to
# theirs without a visible seam.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
HAIRLINE = "#e1e0d9"
PARTICIPANT = "#eb6834"  # the hybrid's own result
INFO = "#2a78d6"  # the BCDJ simulated baseline
BASELINE = "#c3c2b7"

# Free to run -- no API call -- so a tight reference distribution costs
# nothing. Ground truth is one village realised once; this is what "once"
# would look like if BCDJ's fitted logit, not an LLM, had generated it.
BASELINE_SEEDS = 30

_REP = re.compile(r"^v(?P<village>\d+)_rep(?P<rep>\d+)\.csv$")


# --------------------------------------------------------------------------
# Loading the hybrid runs
# --------------------------------------------------------------------------


def load_design_rates(
    model: LLMs = LLMs.GPT_5_4_NANO,
    village: int = VILLAGE,
    output_dir: Path | str = OUTPUT_DIR,
    root: Path | str | None = None,
) -> pd.DataFrame:
    """One row per (design, replicate): adoption rate on the full-population denominator.

    **The denominator is `ground_truth_rates`'s n, not the count of rows in the
    CSV.** A run's log holds one row per household actually asked, which by
    round T is usually most but never all of the village -- so `df.joined.mean()`
    over the file answers "rate among the decided", a different and larger
    number than "rate among the village", which is what ground truth and the
    BCDJ baseline are both stated on. A household absent from the log was never
    asked and never adopted, exactly as `RunResult.adopted` leaves it, so
    `joined.sum() / n` reconstructs the same population rate the run itself
    would report without re-simulating anything.
    """
    gt = ground_truth_rates(village, root=root)
    n, n_lead, n_non = gt["n"], gt["n_leaders"], gt["n_non_leaders"]

    model_dir = Path(output_dir) / model.name.lower()
    rows = []
    for design_dir in sorted(model_dir.glob("*")):
        if not design_dir.is_dir():
            continue
        for csv_path in sorted(design_dir.glob(f"v{village}_rep*.csv")):
            match = _REP.match(csv_path.name)
            if not match:
                continue
            df = pd.read_csv(csv_path, usecols=["is_leader", "joined"])
            leader = df["is_leader"].astype(bool)
            rows.append(
                {
                    "design": design_dir.name,
                    "replicate": int(match["rep"]),
                    "n": n,
                    "adopted": int(df["joined"].sum()),
                    "adopted_leaders": int(df.loc[leader, "joined"].sum()),
                    "adopted_non_leaders": int(df.loc[~leader, "joined"].sum()),
                    "rate": df["joined"].sum() / n,
                    "leader_rate": df.loc[leader, "joined"].sum() / n_lead,
                    "non_leader_rate": df.loc[~leader, "joined"].sum() / n_non,
                }
            )
    if not rows:
        raise FileNotFoundError(f"no hybrid logs under {model_dir} for village {village}")
    return pd.DataFrame(rows).sort_values(["design", "replicate"], ignore_index=True)


# --------------------------------------------------------------------------
# The BCDJ simulated baseline: the pooled logit, run on village 6's real network
# --------------------------------------------------------------------------


def bcdj_baseline(
    village: int = VILLAGE,
    seeds: int = BASELINE_SEEDS,
    root: Path | str | None = None,
) -> pd.DataFrame:
    """`bcdj_run` under the pooled-43-village logit, replicated `seeds` times.

    One row per seed, same three rates `load_design_rates` reports, so the two
    frames plot on the same axes without conversion. The population and
    network are built once and reused across seeds -- nothing about the
    village changes between replicates, only the logit's own coin flips.
    """
    leaders, households = build_village(village, root=root)
    pop = population(leaders, households)
    A = adjacency_matrix(pop)
    beta, _ = fit_betas(root=root)
    p = logit_p(covariates(village, root=root), beta)

    n = len(pop)
    is_leader = np.array([a.is_leader for a in pop], dtype=bool)
    n_lead, n_non = int(is_leader.sum()), int((~is_leader).sum())

    rows = []
    for s in range(seeds):
        r = bcdj_run(pop, A, p, village=village, seed=s, replicate=s)
        rows.append(
            {
                "seed": s,
                "n": n,
                "rate": r.adopted.sum() / n,
                "leader_rate": r.adopted[is_leader].sum() / n_lead,
                "non_leader_rate": r.adopted[~is_leader].sum() / n_non,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Shared drawing bits
# --------------------------------------------------------------------------


def _style_axis(ax, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_ylabel(ylabel, fontsize=10, color=INK_2)
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(axis="y", labelsize=9, colors=INK_2)
    ax.grid(axis="y", color=HAIRLINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(HAIRLINE)


def _reference_band(ax, x0: float, x1: float, mean: float, std: float, color: str, label: str) -> None:
    ax.fill_between([x0, x1], mean - std, mean + std, color=color, alpha=0.12, zorder=1, linewidth=0)
    ax.plot([x0, x1], [mean, mean], color=color, linewidth=1.6, linestyle="--", zorder=2, label=label)


def _dot_strip(ax, x: float, values: pd.Series, color: str, width: float = 0.16) -> None:
    """Every replicate as its own dot, jittered, plus a bar at the mean.

    Individual dots rather than a formula-derived error bar: a design run at
    1-2 replicates should look sparse, not confidently narrow. `np.std` on a
    single value is 0, which would otherwise draw a whisker with nothing
    behind it.
    """
    values = values.to_numpy(dtype=float)
    mean = values.mean()
    ax.bar(x, mean, width=width * 2.2, color=color, alpha=0.30, edgecolor=color, linewidth=1.0, zorder=2)
    if len(values) > 1:
        rng = np.random.default_rng(abs(hash(("jitter", x))) % (2**32))
        jitter = rng.uniform(-width * 0.5, width * 0.5, size=len(values))
    else:
        jitter = np.zeros(1)
    ax.scatter(x + jitter, values, s=22, color=color, edgecolor=SURFACE, linewidth=0.6, zorder=3)


# --------------------------------------------------------------------------
# Plot 1: overall adoption rate by design
# --------------------------------------------------------------------------


def plot_adoption_by_design(
    rates: pd.DataFrame | None = None,
    baseline: pd.DataFrame | None = None,
    village: int = VILLAGE,
    root: Path | str | None = None,
    outfile: Path | None = None,
) -> plt.Figure:
    """Population adoption rate per design against ground truth and the BCDJ baseline."""
    rates = rates if rates is not None else load_design_rates(village=village, root=root)
    baseline = baseline if baseline is not None else bcdj_baseline(village=village, root=root)
    gt = ground_truth_rates(village, root=root)

    designs = sorted(rates["design"].unique())
    fig, ax = plt.subplots(figsize=(max(9.0, 0.9 * len(designs) + 2.0), 6.0), facecolor=SURFACE)

    for i, design in enumerate(designs):
        _dot_strip(ax, i, rates.loc[rates["design"] == design, "rate"], PARTICIPANT)

    _reference_band(ax, -0.5, len(designs) - 0.5, baseline["rate"].mean(), baseline["rate"].std(), INFO,
                     f"BCDJ simulated baseline (n={len(baseline)} seeds)")
    ax.plot([-0.5, len(designs) - 0.5], [gt["all"], gt["all"]], color=INK, linewidth=1.8, zorder=2,
            label=f"ground truth (empirical, n={gt['n']})")

    ax.set_xlim(-0.5, len(designs) - 0.5)
    ax.set_xticks(range(len(designs)))
    reps_per = rates.groupby("design")["replicate"].nunique()
    ticks = [f"{d}\n(n={reps_per[d]})" for d in designs]
    ax.set_xticklabels(ticks, fontsize=8, family="monospace", color=INK_2)
    _style_axis(ax, "adoption rate (whole population)")
    ax.set_title(f"Village {village} -- adoption rate by design", fontsize=13, color=INK, loc="left", pad=10)

    handles = [
        Line2D([], [], marker="o", ls="", ms=7, mfc=PARTICIPANT, mec=SURFACE, label="hybrid run (each dot: one replicate)"),
        Line2D([], [], color=INFO, lw=1.6, ls="--", label=f"BCDJ simulated baseline (mean +/- std, n={len(baseline)} seeds)"),
        Line2D([], [], color=INK, lw=1.8, label=f"ground truth (empirical, n={gt['n']})"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK_2, borderaxespad=0.3)
    fig.text(0.005, 0.005,
              "denominator: full giant-component village (households never asked count as not adopted)",
              fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 1.0))

    if outfile is not None:
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=200, facecolor=SURFACE, bbox_inches="tight")
        print(f"wrote {outfile}")
    return fig


# --------------------------------------------------------------------------
# Plot 2: leader vs non-leader split
# --------------------------------------------------------------------------


def plot_adoption_leader_split(
    rates: pd.DataFrame | None = None,
    baseline: pd.DataFrame | None = None,
    village: int = VILLAGE,
    root: Path | str | None = None,
    outfile: Path | None = None,
) -> plt.Figure:
    """Leader and non-leader adoption rate per design, same two baselines each.

    Ground truth has leaders adopt *less* than non-leaders (13.6% vs 25.9% on
    village 6) -- the MFI's direct pitch to a leader is not the same as a
    neighbour's endorsement, and BCDJ's own logit is fit on leaders only, so it
    reproduces the reversal by construction. Whether any hybrid design does is
    the question this figure is for.
    """
    rates = rates if rates is not None else load_design_rates(village=village, root=root)
    baseline = baseline if baseline is not None else bcdj_baseline(village=village, root=root)
    gt = ground_truth_rates(village, root=root)

    designs = sorted(rates["design"].unique())
    width = 0.32
    fig, ax = plt.subplots(figsize=(max(9.5, 1.1 * len(designs) + 2.0), 6.4), facecolor=SURFACE)

    for i, design in enumerate(designs):
        sub = rates.loc[rates["design"] == design]
        _dot_strip(ax, i - width, sub["leader_rate"], PARTICIPANT, width=width * 0.8)
        _dot_strip(ax, i + width, sub["non_leader_rate"], "#c2410c", width=width * 0.8)  # darker orange, same family

    span = (-0.5, len(designs) - 0.5)
    _reference_band(ax, *span, baseline["leader_rate"].mean(), baseline["leader_rate"].std(), INFO,
                     "BCDJ baseline, leaders")
    _reference_band(ax, *span, baseline["non_leader_rate"].mean(), baseline["non_leader_rate"].std(), "#7fb2e8",
                     "BCDJ baseline, non-leaders")
    ax.plot(span, [gt["leaders"], gt["leaders"]], color=INK, linewidth=1.8, zorder=2)
    ax.plot(span, [gt["non_leaders"], gt["non_leaders"]], color=INK, linewidth=1.8, linestyle=":", zorder=2)

    ax.set_xlim(*span)
    ax.set_xticks(range(len(designs)))
    reps_per = rates.groupby("design")["replicate"].nunique()
    ax.set_xticklabels([f"{d}\n(n={reps_per[d]})" for d in designs], fontsize=8, family="monospace", color=INK_2)
    _style_axis(ax, "adoption rate")
    ax.set_title(f"Village {village} -- adoption rate by design, leaders vs non-leaders", fontsize=13, color=INK,
                 loc="left", pad=10)

    handles = [
        Line2D([], [], marker="o", ls="", ms=7, mfc=PARTICIPANT, mec=SURFACE, label="hybrid: leaders"),
        Line2D([], [], marker="o", ls="", ms=7, mfc="#c2410c", mec=SURFACE, label="hybrid: non-leaders"),
        Line2D([], [], color=INFO, lw=1.6, ls="--", label="BCDJ baseline: leaders"),
        Line2D([], [], color="#7fb2e8", lw=1.6, ls="--", label="BCDJ baseline: non-leaders"),
        Line2D([], [], color=INK, lw=1.8, label=f"ground truth: leaders ({gt['leaders']:.1%})"),
        Line2D([], [], color=INK, lw=1.8, ls=":", label=f"ground truth: non-leaders ({gt['non_leaders']:.1%})"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8, labelcolor=INK_2, ncols=2,
              borderaxespad=0.3)
    fig.text(0.005, 0.005,
              f"n_leaders={gt['n_leaders']}, n_non_leaders={gt['n_non_leaders']} (giant component); "
              "households never asked count as not adopted",
              fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 1.0))

    if outfile is not None:
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=200, facecolor=SURFACE, bbox_inches="tight")
        print(f"wrote {outfile}")
    return fig


# --------------------------------------------------------------------------
# Plot 6: per-household AUC
# --------------------------------------------------------------------------


def household_auc(
    model: LLMs = LLMs.GPT_5_4_NANO,
    village: int = VILLAGE,
    output_dir: Path | str = OUTPUT_DIR,
    root: Path | str | None = None,
    n_boot: int = 500,
) -> pd.DataFrame:
    """One row per design: rank-AUC of each household's share-of-replicates-adopted vs. its true outcome.

    Collapsed to a population rate, a design that is uniformly too eager and a
    design that ranks households correctly at the wrong level look identical.
    This tells them apart: a household's score here is the fraction of that
    design's replicates in which it ended up adopted (0 for a replicate where
    it was never asked, same convention as `load_design_rates`), scored
    against `v.mf` -- ground truth, read here and nowhere near a prompt
    (module docstring, "the ground truth is read separately, by the scorer").

    A design run at few replicates gets a coarse score (2 replicates: only
    0, .5, 1 are possible), so the bootstrap CI is over households, not
    replicates -- it says how much the *AUC estimate* would move under a
    different draw of village 6's households, not how much more data the
    design itself needs.
    """
    leaders, households = build_village(village, root=root)
    pop = population(leaders, households)
    hh_ids = np.array([a.hh_id for a in pop])
    v = dl.load_village(village, root=root if root is not None else dl.DEFAULT_ROOT)
    keep = v.in_giant.astype(bool)
    truth = v.mf[keep].astype(bool)
    if len(truth) != len(hh_ids):
        raise RuntimeError(f"v{village}: {len(truth)} ground-truth rows but {len(hh_ids)} agents; giant-only mismatch")
    index_of = {hh: i for i, hh in enumerate(hh_ids)}

    model_dir = Path(output_dir) / model.name.lower()
    rng = np.random.default_rng(0)
    rows = []
    for design_dir in sorted(model_dir.glob("*")):
        if not design_dir.is_dir():
            continue
        csvs = sorted(design_dir.glob(f"v{village}_rep*.csv"))
        if not csvs:
            continue
        adopted_sum = np.zeros(len(hh_ids))
        for csv_path in csvs:
            df = pd.read_csv(csv_path, usecols=["hh_id", "joined"])
            for hh, joined in zip(df["hh_id"], df["joined"]):
                idx = index_of.get(int(hh))
                if idx is not None and joined:
                    adopted_sum[idx] += 1
        score = adopted_sum / len(csvs)
        estimate = auc(score, truth)

        boot = np.empty(n_boot)
        n = len(score)
        for b in range(n_boot):
            sample = rng.integers(0, n, size=n)
            boot[b] = auc(score[sample], truth[sample])
        lo, hi = np.nanpercentile(boot, [5, 95])

        rows.append({
            "design": design_dir.name,
            "n_replicates": len(csvs),
            "auc": estimate,
            "ci_lo": lo,
            "ci_hi": hi,
        })
    if not rows:
        raise FileNotFoundError(f"no hybrid logs under {model_dir} for village {village}")
    return pd.DataFrame(rows).sort_values("design", ignore_index=True)


def plot_household_auc(
    table: pd.DataFrame | None = None,
    village: int = VILLAGE,
    root: Path | str | None = None,
    outfile: Path | None = None,
) -> plt.Figure:
    table = table if table is not None else household_auc(village=village, root=root)
    designs = list(table["design"])

    fig, ax = plt.subplots(figsize=(max(8.0, 0.9 * len(designs) + 2.0), 5.6), facecolor=SURFACE)
    x = np.arange(len(designs))
    err = np.vstack([table["auc"] - table["ci_lo"], table["ci_hi"] - table["auc"]])
    ax.bar(x, table["auc"], width=0.55, color=PARTICIPANT, alpha=0.75, edgecolor=PARTICIPANT, zorder=3)
    ax.errorbar(x, table["auc"], yerr=err, fmt="none", ecolor=INK_2, elinewidth=1.1, capsize=3, zorder=4)
    ax.axhline(0.5, color=INK, linewidth=1.4, linestyle="--", zorder=2, label="chance (AUC = 0.5)")

    ax.set_xticks(x)
    ticks = [f"{d}\n(n={r})" for d, r in zip(table["design"], table["n_replicates"])]
    ax.set_xticklabels(ticks, fontsize=8, family="monospace", color=INK_2)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUC: household's share-of-replicates-adopted vs. true adoption", fontsize=9.5, color=INK_2)
    ax.set_title(f"Village {village} -- does each design rank households correctly, even where the level is wrong?",
                 fontsize=12.5, color=INK, loc="left", pad=10)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=HAIRLINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=9, colors=INK_2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(HAIRLINE)
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_2)
    fig.text(0.005, 0.005,
              "whiskers: 90% bootstrap CI over households (500 resamples), not over replicates",
              fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 1.0))

    if outfile is not None:
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=200, facecolor=SURFACE, bbox_inches="tight")
        print(f"wrote {outfile}")
    return fig


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

FIGURE_DIR = Path("figures/hybrid")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--village", type=int, default=VILLAGE)
    p.add_argument("--model", default=LLMs.GPT_5_4_NANO.value, choices=[m.value for m in LLMs])
    p.add_argument("--view", default="all", choices=("all", "adoption", "leader-split", "auc"))
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=FIGURE_DIR)
    p.add_argument("--dpi", type=int, default=200)
    a = p.parse_args(argv)

    model = LLMs(a.model)
    rates = load_design_rates(model=model, village=a.village, output_dir=a.output_dir, root=a.root)
    baseline = bcdj_baseline(village=a.village, root=a.root)

    if a.view in ("all", "adoption"):
        fig = plot_adoption_by_design(rates, baseline, village=a.village, root=a.root,
                                       outfile=a.out_dir / "adoption_by_design.png")
        plt.close(fig)
    if a.view in ("all", "leader-split"):
        fig = plot_adoption_leader_split(rates, baseline, village=a.village, root=a.root,
                                          outfile=a.out_dir / "adoption_leader_split.png")
        plt.close(fig)
    if a.view in ("all", "auc"):
        table = household_auc(model=model, village=a.village, output_dir=a.output_dir, root=a.root)
        fig = plot_household_auc(table, village=a.village, root=a.root, outfile=a.out_dir / "household_auc.png")
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
