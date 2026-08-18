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
import textwrap
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
#
# The grammar here is `src/full_llm_model/analysis.py`'s, and the pilots' before
# it, kept rather than re-invented so the three sets of figures read as one
# report: a bar is a mean over replicates, its whisker is +/-1 SE of that mean,
# the value is printed above it, the key sits above the axes where it cannot
# land on a bar, and the caption is wrapped to the figure. The rate axis stays
# 0-100% instead of zooming to whatever this run happened to do -- a design's
# panel is meant to be readable next to the full-LLM one, and a rescaled axis
# would make an over-eager model look moderate.

# The two subpopulations get the pilots' two-arm slots, since that is what they
# are here: one bar per group, side by side, on every column.
LEADER, NON_LEADER = PARTICIPANT, INFO
WARNING = "#e34948"  # kept for the one thing that is a warning, not a category

# Data units of label space kept clear of bars on the right, so a reference line
# can be named where it is rather than in a legend, and the name never has to be
# read against a bar. `bbox_inches="tight"` absorbs whatever the text overruns.
MARGIN = 1.9


def _style_axis(ax, ylabel: str, ymax: float = 1.0) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_ylabel(ylabel, fontsize=9.5, color=INK_2, labelpad=8)
    ax.set_ylim(0.0, ymax + 0.06)  # headroom so a bar near 100% keeps its label and whisker
    ticks = np.arange(0.0, ymax + 1e-9, 0.2)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{tick:.0%}" for tick in ticks])
    ax.tick_params(axis="y", labelsize=9, colors=INK_2, length=0)
    ax.grid(axis="y", color=HAIRLINE, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(HAIRLINE)


def _footnote(fig: plt.Figure, text: str, width: int = 118) -> float:
    """Wrap a caption to the figure rather than letting it stretch the canvas.

    `bbox_inches="tight"` grows the saved image to whatever the widest artist
    needs, so an unwrapped one-line footnote silently doubles the figure width.
    Returns the bottom margin `tight_layout` should leave for it.
    """
    lines = textwrap.wrap(text, width=width)
    fig.text(0.005, 0.005, "\n".join(lines), fontsize=7.5, color=MUTED, va="bottom")
    return 0.03 + 0.022 * (len(lines) - 1)


def _save(fig: plt.Figure, outfile: Path | None, dpi: int = 200) -> None:
    if outfile is None:
        return
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {outfile}")


def mean_se(values) -> tuple[float, float]:
    """Mean and standard error of the mean over replicates.

    The SE is `std(ddof=1) / sqrt(S)`: how precisely S replicates pin down
    *this design's* mean, and nothing more. It is not a claim about village 6,
    which happened once. A design at one replicate has no SE at all and gets no
    whisker rather than a whisker of zero -- a single draw should not look like a
    converged one, which is the same reason the jittered dot strip that used to
    stand here was worth losing: it spent five marks per design on scatter the
    whisker states in one.
    """
    values = pd.Series(values).to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    if len(values) < 2:
        return float(values[0]), float("nan")
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def mean_sd(values) -> tuple[float, float]:
    """Mean and standard deviation -- what the BCDJ baseline's spread means.

    Not the SE of its mean: 30 seeds pin that mean down to nothing, and the
    number the baseline is standing in for is *one* realisation of village 6.
    The SD is how far a single run of the fitted logit lands from its own centre,
    which is the quantity a design's single village is being compared against.
    """
    values = pd.Series(values).to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std(ddof=1) if len(values) > 1 else 0.0)


def _bar(
    ax,
    x: float,
    mean: float,
    spread: float,
    colour: str,
    width: float,
    *,
    alpha: float = 0.9,
    label: str | None = None,
    label_colour: str = INK_2,
    zorder: int = 2,
) -> float:
    """One bar: the mean, a whisker if there is a spread to draw, the value above it.

    Returns the top of whatever was drawn, so a caller can stack a second
    annotation over it without measuring the bar itself.
    """
    if not np.isfinite(mean):
        return float("nan")
    ax.bar(x, mean, width=width, color=colour, alpha=alpha, linewidth=0, zorder=zorder)
    top = mean
    if np.isfinite(spread) and spread > 0:
        ax.errorbar(x, mean, yerr=spread, fmt="none", ecolor=INK_2, elinewidth=1.1, capsize=3.0,
                    capthick=1.1, zorder=zorder + 2)
        top = mean + spread
    # Relief rule: the value is always legible as text, never by bar height alone.
    ax.annotate(label if label is not None else f"{mean:.0%}", (x, top), textcoords="offset points",
                xytext=(0, 4), ha="center", fontsize=8, color=label_colour, zorder=zorder + 3)
    return top


def _asymmetric_bar(ax, x: float, mean: float, lo: float, hi: float, colour: str, width: float) -> float:
    """`_bar` with an interval that is not symmetric about the mean (a bootstrap CI)."""
    ax.bar(x, mean, width=width, color=colour, alpha=0.9, linewidth=0, zorder=2)
    ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], fmt="none", ecolor=INK_2, elinewidth=1.1,
                capsize=3.0, capthick=1.1, zorder=4)
    return hi


def _column_ticks(
    ax,
    labels: list[str],
    counts: list[str | None],
    n_reference: int = 0,
    flagged: set[str] | None = None,
) -> None:
    """Design labels in monospace, with the replicate count on a second line.

    The count belongs on the tick and not in the legend the way the full-LLM
    figures carry it: there the design pairs *are* the legend's series, here they
    are the x axis. Reference columns keep the proportional face, which is the
    quiet way of saying they are not one of the runs.

    `flagged` marks a column the way the pilots mark a design that came out the
    wrong way round -- on its tick, in the warning colour, rather than with a
    glyph floating over the bars.
    """
    flagged = flagged or set()
    ax.set_xticks(range(len(labels)))
    ticks = []
    for lab, count in zip(labels, counts):
        head = f"{lab} !" if lab in flagged else lab
        ticks.append(head if count is None else f"{head}\n{count}")
    ax.set_xticklabels(ticks, fontsize=8, color=INK_2, linespacing=1.5)
    for i, tick in enumerate(ax.get_xticklabels()):
        tick.set_family("monospace" if i >= n_reference else "sans-serif")
        if labels[i] in flagged:
            tick.set_color(WARNING)
    ax.tick_params(axis="x", length=0, pad=7)


def _legend(ax, handles: list[Line2D]) -> None:
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.005), frameon=False, fontsize=8.5,
              labelcolor=INK_2, ncols=len(handles) if len(handles) <= 4 else 3, columnspacing=1.8,
              handletextpad=0.6, borderaxespad=0.0)


def _swatch(colour: str, label: str, alpha: float = 0.9) -> Line2D:
    return Line2D([], [], marker="s", ls="", ms=8, mfc=colour, mec=colour, alpha=alpha, label=label)


# --------------------------------------------------------------------------
# Plot 1: overall adoption rate by design
# --------------------------------------------------------------------------


def plot_adoption_by_design(
    rates: pd.DataFrame | None = None,
    baseline: pd.DataFrame | None = None,
    village: int = VILLAGE,
    root: Path | str | None = None,
    outfile: Path | None = None,
    dpi: int = 200,
) -> plt.Figure:
    """Population adoption rate per design against ground truth and the BCDJ baseline."""
    rates = rates if rates is not None else load_design_rates(village=village, root=root)
    baseline = baseline if baseline is not None else bcdj_baseline(village=village, root=root)
    gt = ground_truth_rates(village, root=root)
    truth = gt["all"]

    designs = sorted(rates["design"].unique())
    reps_per = rates.groupby("design")["replicate"].nunique()
    b_mean, b_sd = mean_sd(baseline["rate"])
    span = (-0.62, len(designs) - 0.38)
    label_x = span[1] + 0.30

    fig, ax = plt.subplots(figsize=(max(9.0, 0.95 * len(designs) + 4.4), 5.4), facecolor=SURFACE)

    multiples = []
    for i, design in enumerate(designs):
        mean, se = mean_se(rates.loc[rates["design"] == design, "rate"])
        _bar(ax, i, mean, se, PARTICIPANT, width=0.62)
        multiples.append(mean / truth)

    # Both references over the bars they refer to, rather than under them: the
    # comparison is the point of the figure, and a bar drawn from zero would
    # otherwise bury the two levels it is being read against. Their names and the
    # baseline's spread go in the right margin, clear of every bar.
    ax.plot(span, [b_mean, b_mean], color=INFO, linewidth=1.4, linestyle=(0, (5, 4)), zorder=6)
    ax.errorbar(label_x, b_mean, yerr=b_sd, fmt="none", ecolor=INFO, elinewidth=1.1, capsize=3.0, capthick=1.1,
                zorder=6)
    ax.annotate(f"BCDJ logit {b_mean:.1%} +/-{b_sd:.1%}", (label_x + 0.16, b_mean), va="center", ha="left",
                fontsize=8, color=INFO, zorder=7)
    ax.plot(span, [truth, truth], color=INK, linewidth=1.6, zorder=6)
    ax.annotate(f"ground truth {truth:.1%}", (label_x, truth), va="center", ha="left", fontsize=8, color=INK,
                zorder=7)

    ax.set_xlim(span[0], span[1] + MARGIN)
    _column_ticks(ax, designs, [f"S={reps_per[d]}" for d in designs])
    _style_axis(ax, "adoption rate (whole population)")
    ax.set_title(f"Village {village} -- adoption rate by design, against the take-up it is aiming at",
                 fontsize=13, color=INK, loc="left", pad=38)
    _legend(ax, [
        _swatch(PARTICIPANT, "design mean over replicates (whisker: +/-1 SE)"),
        Line2D([], [], color=INK, lw=1.6, label=f"ground truth (empirical, n={gt['n']})"),
        Line2D([], [], color=INFO, lw=1.4, ls=(0, (5, 4)), label=f"BCDJ simulated baseline (mean +/-1 SD, "
                                                                f"{len(baseline)} seeds)"),
    ])
    bottom = _footnote(fig,
        f"denominator: full giant-component village, n={gt['n']} (a household never asked was never informed and "
        "counts as not adopted). every design overshoots the village's real take-up, by "
        f"{min(multiples):.1f}x to {max(multiples):.1f}x. the baseline's spread is +/-1 SD over its seeds, not "
        "the SE of its mean: it stands in for one realisation of village 6, which is what a design is compared "
        "against.")
    fig.tight_layout(rect=(0.0, bottom, 1.0, 1.0))
    _save(fig, outfile, dpi)
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
    dpi: int = 200,
) -> plt.Figure:
    """Leader and non-leader adoption rate per design, with both references as their own columns.

    Ground truth has leaders adopt *less* than non-leaders (13.6% vs 25.9% on
    village 6) -- the MFI's direct pitch to a leader is not the same as a
    neighbour's endorsement, and BCDJ's own logit is fit on leaders only. Whether
    any hybrid design reproduces the reversal is the question this figure is for,
    so the two references are drawn as columns in the same two-bar mark as the
    designs rather than as horizontal lines: four levels crammed into the
    0.13-0.30 band were unreadable, and a column puts the reference's own
    leader/non-leader gap in exactly the form the designs' gaps are in.
    """
    rates = rates if rates is not None else load_design_rates(village=village, root=root)
    baseline = baseline if baseline is not None else bcdj_baseline(village=village, root=root)
    gt = ground_truth_rates(village, root=root)

    designs = sorted(rates["design"].unique())
    reps_per = rates.groupby("design")["replicate"].nunique()

    columns: list[dict] = [
        {"label": "ground truth", "count": f"n={gt['n']}", "alpha": 0.45,
         "leader": (gt["leaders"], float("nan")), "non_leader": (gt["non_leaders"], float("nan"))},
        {"label": "BCDJ logit", "count": f"{len(baseline)} seeds", "alpha": 0.45,
         "leader": mean_sd(baseline["leader_rate"]), "non_leader": mean_sd(baseline["non_leader_rate"])},
    ]
    for design in designs:
        sub = rates.loc[rates["design"] == design]
        columns.append({"label": design, "count": f"S={reps_per[design]}", "alpha": 0.9,
                        "leader": mean_se(sub["leader_rate"]), "non_leader": mean_se(sub["non_leader_rate"])})

    width = 0.38
    span = (-0.62, len(columns) - 0.38)
    fig, ax = plt.subplots(figsize=(max(9.5, 1.15 * len(columns) + 2.6), 5.4), facecolor=SURFACE)

    # A column whose two bars come out the wrong way round is flagged on its tick,
    # the way the pilots flag a design -- but only while the flag discriminates.
    # Every design reversing the order is the footnote's finding, not nine red
    # ticks': painting the whole axis red would say it once per column and point
    # at nothing.
    flagged = {col["label"] for col in columns if col["leader"][0] >= col["non_leader"][0]}
    for i, col in enumerate(columns):
        lead_mean, lead_spread = col["leader"]
        non_mean, non_spread = col["non_leader"]
        _bar(ax, i - width / 2, lead_mean, lead_spread, LEADER, width, alpha=col["alpha"])
        _bar(ax, i + width / 2, non_mean, non_spread, NON_LEADER, width, alpha=col["alpha"])
    reversed_count = len(flagged & set(designs))
    if reversed_count == len(designs):
        flagged = set()

    ax.axvline(1.5, color=HAIRLINE, linewidth=1.0, zorder=1)  # references left of it, runs right

    ax.set_xlim(*span)
    _column_ticks(ax, [c["label"] for c in columns], [c["count"] for c in columns], n_reference=2,
                  flagged=flagged)
    _style_axis(ax, "adoption rate")
    ax.set_title(f"Village {village} -- leaders vs non-leaders, by design",
                 fontsize=13, color=INK, loc="left", pad=38)
    handles = [
        _swatch(LEADER, f"leaders (n={gt['n_leaders']})"),
        _swatch(NON_LEADER, f"non-leaders (n={gt['n_non_leaders']})"),
        _swatch(MUTED, "reference column (paler)", alpha=0.45),
    ]
    if flagged:
        handles.append(Line2D([], [], marker="$!$", ls="", ms=7, mfc=WARNING, mec=WARNING,
                              label="flagged tick: leaders adopt more, reversing ground truth's order"))
    _legend(ax, handles)
    bottom = _footnote(fig,
        f"ground truth: {gt['leaders']:.1%} of leaders against {gt['non_leaders']:.1%} of non-leaders -- "
        f"{reversed_count} of {len(designs)} designs put the two the other way round. giant component, households "
        "never asked count as not adopted. whiskers: +/-1 SE of the mean over a design's replicates, +/-1 SD over "
        "the baseline's seeds; ground truth is one realisation and has none.")
    fig.tight_layout(rect=(0.0, bottom, 1.0, 1.0))
    _save(fig, outfile, dpi)
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
    dpi: int = 200,
) -> plt.Figure:
    """Each design's ranking AUC, on the same 0-100% frame as the other two panels.

    The frame is not zoomed to the designs even though they all sit near the
    middle of it, for the same reason the other two panels are not: a reader
    stepping between the three figures should not have to re-read the axis. What
    the zoom would have bought is bought instead by the chance line drawn over
    the bars and the estimate printed above each whisker.
    """
    table = table if table is not None else household_auc(village=village, root=root)
    designs = list(table["design"])
    span = (-0.62, len(designs) - 0.38)

    fig, ax = plt.subplots(figsize=(max(9.0, 0.95 * len(designs) + 4.4), 4.8), facecolor=SURFACE)

    for i, row in enumerate(table.itertuples()):
        top = _asymmetric_bar(ax, i, row.auc, row.ci_lo, row.ci_hi, PARTICIPANT, width=0.62)
        clears = row.ci_lo > 0.5
        ax.annotate(f"{row.auc:.2f}", (i, top), textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=8, color=INK if clears else INK_2, zorder=5)

    # Chance over the bars, not behind them: 0.5 is this scale's null, and it is
    # the only line on the panel a bar can be on the wrong side of.
    ax.plot(span, [0.5, 0.5], color=INK, linewidth=1.6, zorder=6)
    ax.annotate("chance (AUC = 0.5)", (span[1] + 0.30, 0.5), va="center", ha="left", fontsize=8, color=INK,
                zorder=7)

    ax.set_xlim(span[0], span[1] + MARGIN)
    _column_ticks(ax, designs, [f"S={n}" for n in table["n_replicates"]])
    _style_axis(ax, "AUC: share of replicates adopted vs. true adoption")
    ax.set_title(f"Village {village} -- does a design rank households correctly, even at the wrong level?",
                 fontsize=13, color=INK, loc="left", pad=38)
    _legend(ax, [
        _swatch(PARTICIPANT, "AUC estimate (whisker: 90% bootstrap CI over households)"),
        Line2D([], [], color=INK, lw=1.6, label="chance (AUC = 0.5)"),
    ])

    n_clear = int((table["ci_lo"] > 0.5).sum())
    bottom = _footnote(fig,
        f"{n_clear} of {len(designs)} designs keep their whole interval above chance (those estimates are printed "
        "in black). the CI is 500 bootstrap resamples over households, not over replicates: it says how far the "
        "AUC estimate would move under a different draw of village 6's households, not how many more replicates "
        "the design needs. a design at S replicates can only score a household at multiples of 1/S, so a small S "
        "coarsens the ranking it is being credited with.")
    fig.tight_layout(rect=(0.0, bottom, 1.0, 1.0))
    _save(fig, outfile, dpi)
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
                                       outfile=a.out_dir / "adoption_by_design.png", dpi=a.dpi)
        plt.close(fig)
    if a.view in ("all", "leader-split"):
        fig = plot_adoption_leader_split(rates, baseline, village=a.village, root=a.root,
                                          outfile=a.out_dir / "adoption_leader_split.png", dpi=a.dpi)
        plt.close(fig)
    if a.view in ("all", "auc"):
        table = household_auc(model=model, village=a.village, output_dir=a.output_dir, root=a.root)
        fig = plot_household_auc(table, village=a.village, root=a.root, outfile=a.out_dir / "household_auc.png",
                                  dpi=a.dpi)
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
