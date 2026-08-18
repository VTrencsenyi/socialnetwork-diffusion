"""Plot a village network in the style of Figure 1 of Banerjee et al. (2013).

Figure 1 in the paper is a *schematic* drawn on a toy ~20-node graph: grey ties,
"L" leaders, "?" informed-but-undecided nodes, gold ticks for participants, dark
crosses for non-participants, and curved arrows for information passing. This
module renders the same visual grammar on the real village networks.

One thing the paper's figure can do that the data cannot
--------------------------------------------------------
The arrows in Figure 1 are *model output*, not observation. This dataset records
network ties, who the leaders were, and who eventually participated. It contains
no record of who actually told whom, and no household-level adoption date. So:

- `view="outcome"` shows only what was observed: ties, leaders, participants,
  non-participants. Nothing here is inferred.
- `view="hops"` shades households by network distance to the nearest leader.
  This is the paper's own stated proxy -- "network distance to these leaders
  therefore offers a proxy for access to information" -- and no more than that.
- `view="panels"` draws the Figure-1 A-E sequence. It needs a `DiffusionTrace`.
  If you do not supply one it falls back to `DiffusionTrace.from_bfs()`, a pure
  breadth-first reachability wave from the leaders, which is a null model and is
  labelled as such on the figure. Once the LLM simulation exists, pass its real
  trace here and the same code renders it.

A non-participant in this data is ambiguous in a way the plot must not hide: it
may be a household that never heard about microfinance, or one that heard and
declined. Nothing in the bundle distinguishes them, so both render as one
"did not participate" category rather than as Figure 1's separate pale and
crossed nodes.

Palette: dataviz reference palette, validated with
`validate_palette.js "#eb6834,#2a78d6" --mode light --pairs all` (all checks
pass). Figures commit to the light surface only -- they are destined for a
write-up PDF, not a themed web page.

Usage
-----
    python3 src/plots.py --village 1
    python3 src/plots.py --village 1 --view hops
    python3 src/plots.py --village 1 --view panels
    python3 src/plots.py --village 1 --view all --out figures/village_1.png
"""

from __future__ import annotations

import argparse
import textwrap
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

try:
    from .data_loader import (
        DEFAULT_ROOT,
        EXTRA_MF_VILLAGES,
        NETWORK_TYPES,
        DataError,
        Village,
        _panel_all,
        analysis_villages,
        available_villages,
        load_village,
    )
except ImportError:  # running as a script, not a package
    from data_loader import (  # type: ignore[no-redef]
        DEFAULT_ROOT,
        EXTRA_MF_VILLAGES,
        NETWORK_TYPES,
        DataError,
        Village,
        _panel_all,
        analysis_villages,
        available_villages,
        load_village,
    )

# --------------------------------------------------------------------------
# Palette (dataviz reference instance, light surface)
# --------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
HAIRLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

PARTICIPANT = "#eb6834"  # categorical slot 2 (orange) -- the paper's gold
NEUTRAL_FILL = "#f0efec"  # diverging-midpoint gray: "did not participate"
INFORMED_FILL = "#d5d4ce"  # a step darker: "informed but did not participate"
INFO = "#2a78d6"  # categorical slot 1 (blue) -- information passing

# Sequential blue, ordinal-safe on the light surface (nothing lighter than
# step 250, per the ordinal floor).
HOP_RAMP = ["#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef"]
UNREACHED = "#e1e0d9"

# Diverging pair for correlation heatmaps: blue <-> red, neutral gray midpoint
# (dataviz reference palette's documented diverging pair; midpoint reuses
# NEUTRAL_FILL). Equal step count per arm, built as a 3-stop interpolation.
CORR_NEG = "#104281"  # sequential-blue darkest step -- r = -1
CORR_MID = NEUTRAL_FILL  # r = 0
CORR_POS = "#e34948"  # categorical slot 8 (red) -- r = +1
CORR_CMAP = LinearSegmentedColormap.from_list("corr_diverging", [CORR_NEG, CORR_MID, CORR_POS], N=256)

FIGSIZE = (11.0, 9.0)


# --------------------------------------------------------------------------
# Diffusion trace: what the simulator will hand us later
# --------------------------------------------------------------------------


@dataclass
class DiffusionTrace:
    """A record of one diffusion run, in adjacency-row index space.

    informed_at   (n,) int, period each household first heard; -1 = never
    adopted_at    (n,) int, period each household adopted;     -1 = never
    transmissions list of (t, source, target) information-passing events
    label         short description, printed on the figure
    is_observed   True only if this came from data. Always False here: the
                  bundle contains no transmission record, so any trace is
                  either simulated or a null model.
    """

    informed_at: np.ndarray
    adopted_at: np.ndarray
    transmissions: list[tuple[int, int, int]] = field(default_factory=list)
    label: str = "simulated"
    is_observed: bool = False

    @property
    def n_periods(self) -> int:
        return int(max(self.informed_at.max(), self.adopted_at.max(), 0)) + 1

    @classmethod
    def from_bfs(cls, v: Village) -> "DiffusionTrace":
        """Null model: information spreads one hop per period, nobody blocks it.

        Adoption is taken from ground truth and pinned to the period the
        household is first reached. So the *timing* is hypothetical while the
        *set* of adopters is real. This is deliberately a straw man -- it is
        what a network with no behaviour in it would predict, and it is the
        thing an LLM agent model has to beat.
        """
        n = v.n
        informed_at = np.full(n, -1, dtype=int)
        transmissions: list[tuple[int, int, int]] = []

        frontier = deque()
        for i in np.flatnonzero(v.leader == 1):
            informed_at[i] = 0
            frontier.append(i)

        while frontier:
            i = frontier.popleft()
            for j in v.neighbours(i):
                if informed_at[j] == -1:
                    informed_at[j] = informed_at[i] + 1
                    transmissions.append((int(informed_at[j]), int(i), int(j)))
                    frontier.append(j)

        adopted_at = np.where((v.mf == 1) & (informed_at >= 0), informed_at, -1)
        return cls(
            informed_at=informed_at,
            adopted_at=adopted_at,
            transmissions=transmissions,
            label="BFS reachability (null model): information spreads one hop per period, nobody blocks it",
            is_observed=False,
        )


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------


def build_graph(v: Village) -> nx.Graph:
    g = nx.from_numpy_array(v.adjacency)
    return g


def layout(v: Village, seed: int = 7, iterations: int = 300) -> np.ndarray:
    """Positions for every household, as an (n, 2) array.

    The giant component gets a spring layout of its own; the small fragments
    and isolates are parked in a strip underneath rather than being allowed to
    push the main structure around, which is what a whole-graph spring layout
    does to a village with 7 isolates.
    """
    comps = v.components()
    pos = np.zeros((v.n, 2), dtype=float)

    giant = comps[0]
    sub = nx.from_numpy_array(v.adjacency[np.ix_(giant, giant)])
    k = 1.6 / np.sqrt(len(giant))
    gp = nx.spring_layout(sub, seed=seed, iterations=iterations, k=k)
    xy = np.array([gp[i] for i in range(len(giant))])
    xy -= xy.min(axis=0)
    span = xy.max(axis=0)
    span[span == 0] = 1.0
    xy /= span.max()
    pos[giant] = xy

    # Fragments in a strip below the giant component.
    rest = [i for c in comps[1:] for i in c]
    if rest:
        y0 = -0.14
        for slot, i in enumerate(rest):
            pos[i] = (0.02 + 0.055 * (slot % 18), y0 - 0.075 * (slot // 18))
    return pos


# --------------------------------------------------------------------------
# Drawing primitives
# --------------------------------------------------------------------------


def _node_sizes(v: Village, lo: float = 34.0, hi: float = 210.0) -> np.ndarray:
    d = v.degree.astype(float)
    if d.max() == d.min():
        return np.full(v.n, lo)
    return lo + (hi - lo) * np.sqrt(d / d.max())


def _draw_edges(ax, v: Village, pos: np.ndarray, alpha: float = 0.45, lw: float = 0.45) -> None:
    iu, ju = np.triu_indices(v.n, k=1)
    m = v.adjacency[iu, ju] == 1
    segs = np.stack([pos[iu[m]], pos[ju[m]]], axis=1)
    ax.add_collection(
        matplotlib.collections.LineCollection(segs, colors=BASELINE, linewidths=lw, alpha=alpha, zorder=1)
    )


def _draw_leader_rings(ax, v: Village, pos: np.ndarray, sizes: np.ndarray, scale: float = 1.9) -> None:
    """Mark leaders with a ring rather than the paper's 'L' glyph.

    Figure 1 labels its ~4 leaders 'L'. Real villages seed 20-40 of them, and
    they sit in the dense core by construction (they were chosen for being
    well connected), so glyphs there collide into an unreadable smudge. The
    ring is the same information without the collision.
    """
    lead = v.leader == 1
    ax.scatter(
        pos[lead, 0],
        pos[lead, 1],
        s=sizes[lead] * scale,
        facecolors="none",
        edgecolors=INK,
        linewidths=1.15,
        zorder=4,
    )


def _draw_arrows(ax, pos: np.ndarray, events, color: str = INFO, rad: float = 0.22, alpha: float = 0.85) -> None:
    for _, s, t in events:
        ax.add_patch(
            FancyArrowPatch(
                pos[s],
                pos[t],
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=1.1,
                color=color,
                alpha=alpha,
                zorder=5,
                shrinkA=4,
                shrinkB=5,
            )
        )


def _clean(ax) -> None:
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.margins(0.04)


# --------------------------------------------------------------------------
# Views
# --------------------------------------------------------------------------


def _annotate_fragments(ax, v: Village, pos: np.ndarray) -> None:
    """Label the strip of disconnected households parked below the main graph."""
    comps = v.components()
    rest = [i for c in comps[1:] for i in c]
    if not rest:
        return
    rest_arr = np.array(rest)
    n_iso = int((v.degree[rest_arr] == 0).sum())
    n_lead = int(v.leader[rest_arr].sum())
    n_took = int(v.mf[rest_arr].sum())

    # An isolated leader is still seeded -- it just has nobody to tell. Say so
    # rather than lumping it in with the households no information can reach.
    bits = [f"outside the giant component: {len(rest)} households ({n_iso} with no ties at all)"]
    if n_lead:
        bits.append(f"{n_lead} of them leaders, seeded but unable to pass information on")
    bits.append(f"{n_took} participated")
    ax.text(
        pos[rest, 0].min() - 0.02,
        pos[rest, 1].max() + 0.035,
        " — ".join(bits),
        fontsize=8,
        color=MUTED,
        ha="left",
        va="bottom",
    )


def plot_outcome(ax, v: Village, pos: np.ndarray) -> None:
    """Observed data only: ties, leaders, participants, non-participants."""
    sizes = _node_sizes(v)
    _draw_edges(ax, v, pos)

    took = v.mf == 1
    ax.scatter(
        pos[~took, 0],
        pos[~took, 1],
        s=sizes[~took],
        c=NEUTRAL_FILL,
        edgecolors=MUTED,
        linewidths=0.6,
        zorder=2,
        label="Did not participate",
    )
    ax.scatter(
        pos[took, 0],
        pos[took, 1],
        s=sizes[took],
        c=PARTICIPANT,
        edgecolors=SURFACE,
        linewidths=0.9,
        zorder=3,
        label="Participated in microfinance",
    )
    _draw_leader_rings(ax, v, pos, sizes)
    _annotate_fragments(ax, v, pos)
    _clean(ax)

    handles = [
        Line2D([], [], marker="o", ls="", ms=8, mfc=PARTICIPANT, mec=SURFACE, label="Participated"),
        Line2D([], [], marker="o", ls="", ms=8, mfc=NEUTRAL_FILL, mec=MUTED, label="Did not participate"),
        Line2D([], [], marker="o", ls="", ms=11, mfc="none", mec=INK, label="Leader (injection point)"),
        Line2D([], [], color=BASELINE, lw=1.0, label="Social tie"),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        frameon=False,
        fontsize=8.5,
        labelcolor=INK_2,
        handletextpad=0.6,
        borderaxespad=0.2,
    )
    ax.set_title("Observed outcome", fontsize=11, color=INK, loc="left", pad=6)


def hops_from_leaders(v: Village) -> np.ndarray:
    """Shortest-path distance to the nearest leader; -1 if unreachable."""
    n = v.n
    dist = np.full(n, -1, dtype=int)
    q = deque()
    for i in np.flatnonzero(v.leader == 1):
        dist[i] = 0
        q.append(i)
    while q:
        i = q.popleft()
        for j in v.neighbours(i):
            if dist[j] == -1:
                dist[j] = dist[i] + 1
                q.append(j)
    return dist


def plot_hops(ax, v: Village, pos: np.ndarray) -> None:
    """Network distance to the nearest leader -- the paper's information proxy."""
    sizes = _node_sizes(v)
    _draw_edges(ax, v, pos)
    d = hops_from_leaders(v)

    # These villages are dense and heavily seeded, so the realised hop range is
    # usually 0-2. Spread the ramp across however many levels actually occur
    # instead of taking its first k steps, which would leave near-identical
    # blues sitting next to each other.
    levels = sorted(set(int(x) for x in np.unique(d) if x >= 0))
    if len(levels) > len(HOP_RAMP):
        levels = levels[: len(HOP_RAMP) - 1] + [levels[len(HOP_RAMP) - 1]]
    idx = np.linspace(0, len(HOP_RAMP) - 1, num=max(len(levels), 2)).round().astype(int)
    top = max(levels) if levels else 0

    handles = []
    for slot, h in enumerate(levels):
        m = (d == h) if h < top else (d >= h)
        colour = HOP_RAMP[idx[slot]]
        if h == 0:
            lbl = "leader (0 hops)"
        elif h < top:
            lbl = f"{h} hop" + ("s" if h > 1 else "")
        else:
            lbl = f"{h}+ hops"
        ax.scatter(pos[m, 0], pos[m, 1], s=sizes[m], c=colour, edgecolors=SURFACE, linewidths=0.7, zorder=3)
        handles.append(Line2D([], [], marker="o", ls="", ms=8, mfc=colour, mec=SURFACE, label=lbl))
    m = d == -1
    if m.any():
        ax.scatter(pos[m, 0], pos[m, 1], s=sizes[m], c=UNREACHED, edgecolors=MUTED, linewidths=0.6, zorder=2)
        handles.append(
            Line2D([], [], marker="o", ls="", ms=8, mfc=UNREACHED, mec=MUTED, label=f"unreachable ({int(m.sum())})")
        )

    _annotate_fragments(ax, v, pos)
    _clean(ax)
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK_2, borderaxespad=0.2)
    reach = d[d >= 0]
    ax.set_title(
        f"Network distance to nearest leader  (max {int(reach.max()) if len(reach) else 0} hops)",
        fontsize=11,
        color=INK,
        loc="left",
        pad=6,
    )

    # State the empirical gradient on the figure, so the proxy is judged, not assumed.
    nonlead = v.leader == 0
    lines = []
    for h in levels:
        if h == 0:
            continue
        sel = nonlead & ((d == h) if h < top else (d >= h))
        if sel.sum() >= 5:
            lbl = f"{h} hop" + ("s" if h > 1 else "") if h < top else f"{h}+ hops"
            lines.append(f"{lbl}: {v.mf[sel].mean():.0%} (n={int(sel.sum())})")
    if lines:
        ax.text(
            0.0,
            -0.04,
            "Take-up by distance (non-leaders) — " + "   ".join(lines),
            transform=ax.transAxes,
            fontsize=8.5,
            color=INK_2,
        )


def plot_panels(
    v: Village,
    trace: DiffusionTrace | None = None,
    max_panels: int = 6,
    seed: int = 7,
    pos: np.ndarray | None = None,
) -> plt.Figure:
    """Figure-1-style A-E sequence: the information front advancing period by period."""
    trace = trace or DiffusionTrace.from_bfs(v)
    pos = layout(v, seed=seed) if pos is None else pos
    sizes = _node_sizes(v, lo=18.0, hi=110.0)

    periods = list(range(min(trace.n_periods, max_panels)))
    ncols = min(3, len(periods))
    nrows = int(np.ceil(len(periods) / ncols))

    # Reserve fixed inches for the header and the legend strip, so a 1-row
    # figure does not have its title land on top of the note.
    head_in, foot_in = 1.15, 0.75
    height = 4.2 * nrows + head_in + foot_in
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.6 * ncols, height), facecolor=SURFACE, squeeze=False
    )
    axes = axes.ravel()

    for ax in axes[len(periods) :]:
        ax.set_visible(False)

    for panel, t in enumerate(periods):
        ax = axes[panel]
        ax.set_facecolor(SURFACE)
        _draw_edges(ax, v, pos, alpha=0.35, lw=0.35)

        informed = (trace.informed_at >= 0) & (trace.informed_at <= t)
        adopted = (trace.adopted_at >= 0) & (trace.adopted_at <= t)

        # uninformed
        m = ~informed
        ax.scatter(pos[m, 0], pos[m, 1], s=sizes[m], c=SURFACE, edgecolors=BASELINE, linewidths=0.6, zorder=2)
        # informed, did not participate
        m = informed & ~adopted
        ax.scatter(pos[m, 0], pos[m, 1], s=sizes[m], c=INFORMED_FILL, edgecolors=INK_2, linewidths=0.7, zorder=3)
        # participated
        m = adopted
        ax.scatter(pos[m, 0], pos[m, 1], s=sizes[m], c=PARTICIPANT, edgecolors=SURFACE, linewidths=0.8, zorder=4)

        # A dense period can carry >100 transmissions; fade them so the arrows
        # read as a flood rather than an unreadable mat of blue.
        events = [e for e in trace.transmissions if e[0] == t]
        _draw_arrows(ax, pos, events, alpha=0.85 if len(events) < 40 else 0.45)
        lead = v.leader == 1
        _draw_leader_rings(ax, v, pos, sizes, scale=2.0)
        _clean(ax)
        ax.set_title(
            f"{chr(65 + panel)}   period {t}  —  informed {informed.mean():.0%},  participating {adopted.mean():.0%}",
            fontsize=10,
            color=INK,
            loc="left",
            pad=4,
        )

    handles = [
        Line2D([], [], marker="o", ls="", ms=8, mfc=SURFACE, mec=BASELINE, label="Not yet informed"),
        Line2D([], [], marker="o", ls="", ms=8, mfc=INFORMED_FILL, mec=INK_2, label="Informed, did not participate"),
        Line2D([], [], marker="o", ls="", ms=8, mfc=PARTICIPANT, mec=SURFACE, label="Participated"),
        Line2D([], [], marker="o", ls="", ms=11, mfc="none", mec=INK, label="Leader (injection point)"),
        Line2D([], [], color=INFO, lw=1.4, label="Information passed this period"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9,
        labelcolor=INK_2,
        bbox_to_anchor=(0.5, 0.16 / height),
    )
    fig.text(
        0.012,
        1 - 0.30 / height,
        f"Village {v.village} — diffusion of information and participation",
        fontsize=13,
        color=INK,
        ha="left",
        va="center",
    )
    note = trace.label
    if not trace.is_observed:
        note += (
            "\nNOT OBSERVED: the bundle records no transmissions and no household adoption dates. "
            "Timing here is hypothetical; only the set of participants is ground truth."
        )
    fig.text(0.012, 1 - 0.52 / height, note, fontsize=8.5, color=INK_2, ha="left", va="top")
    fig.tight_layout(rect=(0, foot_in / height, 1, 1 - head_in / height))
    return fig


# --------------------------------------------------------------------------
# Top-level figures
# --------------------------------------------------------------------------


def plot_village(v: Village, view: str = "outcome", seed: int = 7) -> plt.Figure:
    pos = layout(v, seed=seed)

    if view == "panels":
        return plot_panels(v, seed=seed, pos=pos)

    if view == "all":
        fig, axes = plt.subplots(1, 2, figsize=(16.0, 8.0), facecolor=SURFACE)
        for ax in axes:
            ax.set_facecolor(SURFACE)
        plot_outcome(axes[0], v, pos)
        plot_hops(axes[1], v, pos)
    else:
        fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        if view == "outcome":
            plot_outcome(ax, v, pos)
        elif view == "hops":
            plot_hops(ax, v, pos)
        else:
            raise ValueError(f"unknown view {view!r}")

    fig.suptitle(
        f"Village {v.village} — {v.n} households, {v.n_edges} ties "
        f"({v.network_type})",
        fontsize=13,
        color=INK,
        x=0.012,
        ha="left",
        y=0.985,
    )
    fig.text(
        0.012,
        0.952,
        f"{int(v.leader.sum())} leaders seeded  ·  {int(v.mf.sum())} of {v.n} households participated "
        f"({v.takeup_all:.1%}; {v.takeup_nonleader:.1%} of non-leaders)  ·  node size ∝ degree",
        fontsize=9,
        color=INK_2,
        ha="left",
        va="top",
    )
    fig.text(
        0.012,
        0.022,
        "A household that did not participate may never have heard about microfinance, or may have heard and declined — "
        "the data cannot tell these apart.",
        fontsize=8.5,
        color=MUTED,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    return fig


def plot_takeup_by_village(root: Path = DEFAULT_ROOT, horizon: int | None = None) -> plt.Figure:
    """Empirical microfinance take-up over time, one line per village plus the mean.

    x = t (trimester since the panel's baseline), y = dynamicMF_empirical --
    the observed participation rate among all households in the village, as
    published in panel.dta. Villages run to different final t, so the mean
    line at each t is taken over whichever villages have data there rather
    than requiring all 43 to reach the same horizon.

    If `horizon` is given, only villages whose observation horizon (the max t
    with a non-missing dynamicMF_empirical value) equals it are plotted. That
    subset is small enough to label individually rather than lumping every
    line into one grey band.
    """
    panel = _panel_all(str(root)).dropna(subset=["dynamicMF_empirical"])
    if horizon is not None:
        full_horizon = panel.groupby("village")["t"].max()
        panel = panel[panel["village"].isin(full_horizon[full_horizon == horizon].index)]
    villages = sorted(panel["village"].unique())

    fig, ax = plt.subplots(figsize=(10.0, 7.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    if horizon is not None:
        cmap = plt.get_cmap("tab20")
        colors = {vil: cmap(i % 20) for i, vil in enumerate(villages)}
        for vil in villages:
            g = panel[panel["village"] == vil].sort_values("t")
            ax.plot(g["t"], g["dynamicMF_empirical"], color=colors[vil], alpha=0.9, linewidth=1.5, zorder=2)
    else:
        for _, g in panel.groupby("village"):
            g = g.sort_values("t")
            ax.plot(g["t"], g["dynamicMF_empirical"], color=BASELINE, alpha=0.55, linewidth=1.0, zorder=2)

    mean = panel.groupby("t")["dynamicMF_empirical"].mean().sort_index()
    mean_color = INK if horizon is not None else PARTICIPANT
    ax.plot(
        mean.index,
        mean.values,
        color=mean_color,
        linewidth=3.0,
        linestyle="--" if horizon is not None else "-",
        zorder=4,
    )

    ax.set_xlabel("Trimester (t)", fontsize=10, color=INK_2)
    ax.set_ylabel("Empirical participation rate (dynamicMF_empirical)", fontsize=10, color=INK_2)
    title = "Microfinance take-up per village over time"
    if horizon is not None:
        title += f"  —  horizon-{horizon} villages only (n={len(villages)})"
    ax.set_title(title, fontsize=13, color=INK, loc="left", pad=10)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.set_ylim(0, None)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(HAIRLINE)
    ax.grid(True, color=HAIRLINE, linewidth=0.6, zorder=0)

    if horizon is not None:
        handles = [Line2D([], [], color=colors[vil], lw=1.8, label=f"Village {vil}") for vil in villages]
        handles.append(Line2D([], [], color=INK, lw=2.5, ls="--", label="Mean"))
        ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8, labelcolor=INK_2, ncol=2)
    else:
        handles = [
            Line2D([], [], color=BASELINE, lw=1.2, alpha=0.8, label=f"Village ({len(villages)} total)"),
            Line2D([], [], color=PARTICIPANT, lw=2.5, label="Mean across villages"),
        ]
        ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    return fig


def plot_takeup_horizon_histogram(root: Path = DEFAULT_ROOT) -> plt.Figure:
    """How many trimesters of observed take-up data each village has.

    A village's horizon is the largest t with a non-missing dynamicMF_empirical
    value in panel.dta -- the same cutoff plot_takeup_by_village uses to trim
    each village's line, so the two figures agree on how far a village's data
    goes.
    """
    panel = _panel_all(str(root)).dropna(subset=["dynamicMF_empirical"])
    horizon = panel.groupby("village")["t"].max().astype(int)
    counts = horizon.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8.0, 6.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.bar(counts.index, counts.values, color=INFO, edgecolor=SURFACE, linewidth=0.8, width=0.7, zorder=3)

    for x, y in zip(counts.index, counts.values):
        ax.text(x, y + 0.15, str(int(y)), ha="center", va="bottom", fontsize=9, color=INK_2)

    ax.set_xlabel("Max trimester with observed take-up data", fontsize=10, color=INK_2)
    ax.set_ylabel("Number of villages", fontsize=10, color=INK_2)
    ax.set_title(
        f"Distribution of village observation horizons  (n={int(counts.sum())} villages)",
        fontsize=13,
        color=INK,
        loc="left",
        pad=10,
    )
    ax.set_xticks(counts.index)
    ax.tick_params(colors=INK_2, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(HAIRLINE)
    ax.grid(True, axis="y", color=HAIRLINE, linewidth=0.6, zorder=0)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Feature correlation heatmap
# --------------------------------------------------------------------------

CORR_FEATURES = [
    "religion",
    "rooms",
    "beds",
    "capita",
    "rooms_per_capita",
    "beds_per_capita",
    "electricity",
    "own_latrine",
    "has_shg",
    "has_savings",
    "_adopted",
]


def household_feature_table(v: Village) -> pd.DataFrame:
    """One row per household: the raw and derived attributes behind the heatmap.

    `rooms`, `beds`, `religion`, `electricity`, `own_latrine` and `_adopted`
    come from household_characteristics / MF<V>.csv and are populated for
    every household. `capita`, `has_shg`, `has_savings` come from
    individual_characteristics, which only covers the surveyed-individual
    subset (see data_loader's module docstring); households outside that
    subset get NaN there, and NaN propagates into `rooms_per_capita` /
    `beds_per_capita`. `.corr()` drops those pairwise rather than imputing.

    `religion` is hohreligion turned into category codes (Hinduism/Islam/
    Christianity) purely so it has a numeric column to sit in a Pearson
    correlation matrix -- it is nominal, not ordinal, so treat that row/column
    as "does this correlate with the boundary between religion groups", not
    as a meaningful magnitude.
    """
    hh = v.households

    religion_str = hh.hohreligion.astype(str).str.strip()
    religion = pd.Categorical(religion_str).codes.astype(float)
    religion[religion_str.isin(["nan", ""])] = np.nan

    electricity = hh.electricity.astype(str).str.startswith("Yes").astype(float)
    electricity[hh.electricity.isna()] = np.nan

    own_latrine = (hh.latrine.astype(str).str.strip() == "Owned").astype(float)
    own_latrine[hh.latrine.isna()] = np.nan

    ind = v.individuals
    capita_s = ind.groupby("hhid").size()
    shg_s = ind.groupby("hhid")["shgparticipate"].apply(lambda s: float((s == "Yes").any()))
    savings_s = ind.groupby("hhid")["savings"].apply(lambda s: float((s == "Yes").any()))

    hhid = hh.hhid.to_numpy()
    capita = capita_s.reindex(hhid).to_numpy(dtype=float)
    has_shg = shg_s.reindex(hhid).to_numpy(dtype=float)
    has_savings = savings_s.reindex(hhid).to_numpy(dtype=float)

    rooms = hh.room_no.to_numpy(dtype=float)
    beds = hh.bed_no.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rooms_per_capita = np.where(capita > 0, rooms / capita, np.nan)
        beds_per_capita = np.where(capita > 0, beds / capita, np.nan)

    return pd.DataFrame(
        {
            "village": v.village,
            "religion": religion,
            "rooms": rooms,
            "beds": beds,
            "capita": capita,
            "rooms_per_capita": rooms_per_capita,
            "beds_per_capita": beds_per_capita,
            "electricity": electricity,
            "own_latrine": own_latrine,
            "has_shg": has_shg,
            "has_savings": has_savings,
            "_adopted": v.mf.astype(float),
        }
    )


def _fmt_r(val: float) -> str:
    """`0.43` -> `.43`, `-0.02` -> `-.02` -- drops the leading zero so an 11x11
    grid of two-decimal values has a chance of fitting inside its cell."""
    if np.isnan(val):
        return "–"
    s = f"{val:.2f}"
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s


def _draw_corr_heatmap(ax, corr: pd.DataFrame, title: str, note: str = "") -> None:
    labels = list(corr.columns)
    m = corr.to_numpy()
    n = len(labels)
    im = ax.imshow(m, cmap=CORR_CMAP, vmin=-1, vmax=1, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=42, ha="right", rotation_mode="anchor", fontsize=8.5, color=INK_2)
    ax.set_yticklabels(labels, fontsize=8.5, color=INK_2)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)

    cell_fontsize = 6.3 if n > 9 else 7.5
    for i in range(n):
        for j in range(n):
            val = m[i, j]
            txt_color = MUTED if np.isnan(val) else (SURFACE if abs(val) > 0.55 else INK)
            ax.text(j, i, _fmt_r(val), ha="center", va="center", fontsize=cell_fontsize, color=txt_color)

    ax.set_title(title, fontsize=11.5, color=INK, loc="left", pad=8)
    if note:
        wrapped = "\n".join(textwrap.wrap(note, width=78))
        ax.text(0.0, -0.30, wrapped, transform=ax.transAxes, fontsize=7.8, color=MUTED, ha="left", va="top")

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=INK_2, labelsize=8, length=0)
    cbar.set_label("Pearson r", fontsize=8.5, color=INK_2)


_CORR_NOTE = (
    "capita / has_shg / has_savings / *_per_capita limited to the individual-survey subset; "
    "religion is nominal (category codes, not a magnitude); pairwise-complete correlations."
)


def plot_feature_correlation_village(v: Village) -> plt.Figure:
    """Feature correlation heatmap for one village's households."""
    df = household_feature_table(v)[CORR_FEATURES]
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(9.6, 9.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    note = f"n={v.n} households.  " + _CORR_NOTE
    _draw_corr_heatmap(ax, corr, f"Village {v.village} — feature correlations", note)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    return fig


def plot_feature_correlation_all_villages(
    root: Path = DEFAULT_ROOT,
    network_type: str = "allVillageRelationships",
    villages: list[int] | None = None,
) -> plt.Figure:
    """Feature correlation heatmap pooled over every household in the 43-village analysis sample.

    Defaults to `analysis_villages()`, not `available_villages()`: the 6 extra
    villages with an MF outcome but no BSS program entry are outside the
    paper's analysis scope and would otherwise be mixed in silently.
    """
    villages = list(villages) if villages is not None else analysis_villages(root)
    tables = []
    skipped = []
    for vil in villages:
        try:
            v = load_village(vil, root=root, network_type=network_type)
        except DataError:
            skipped.append(vil)
            continue
        tables.append(household_feature_table(v))
    if skipped:
        print(f"  ! skipped {len(skipped)} village(s) with no usable data: {skipped}")

    pooled = pd.concat(tables, ignore_index=True)
    corr = pooled[CORR_FEATURES].corr()

    extras = sorted(set(villages) & set(EXTRA_MF_VILLAGES))
    extras_note = f"  Includes {len(extras)} village(s) outside the 43-village analysis sample: {extras}." if extras else ""
    fig, ax = plt.subplots(figsize=(9.6, 9.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    note = f"n={len(pooled)} households across {pooled['village'].nunique()} villages.{extras_note}  " + _CORR_NOTE
    _draw_corr_heatmap(ax, corr, "43-village analysis sample — pooled feature correlations", note)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    return fig


def plot_variance_explained_all_villages(
    root: Path = DEFAULT_ROOT,
    network_type: str = "allVillageRelationships",
    villages: list[int] | None = None,
) -> plt.Figure:
    """Bar plot: how much of `_adopted`'s variance each heatmap feature explains alone, pooled over the

    43-village analysis sample. With a single regressor, R² is just the
    squared Pearson r -- the same pairwise-complete correlations
    `plot_feature_correlation_all_villages` already puts in its `_adopted`
    row/column, reread here as "variance explained" and ranked instead of laid
    out as a matrix. Defaults to `analysis_villages()`, not
    `available_villages()` -- see that function's docstring.
    """
    villages = list(villages) if villages is not None else analysis_villages(root)
    tables = []
    skipped = []
    for vil in villages:
        try:
            v = load_village(vil, root=root, network_type=network_type)
        except DataError:
            skipped.append(vil)
            continue
        tables.append(household_feature_table(v))
    if skipped:
        print(f"  ! skipped {len(skipped)} village(s) with no usable data: {skipped}")

    pooled = pd.concat(tables, ignore_index=True)
    corr = pooled[CORR_FEATURES].corr()
    r2 = (corr["_adopted"].drop("_adopted") ** 2).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9.0, 6.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = np.arange(len(r2))
    ax.barh(y, r2.values, color=INFO, edgecolor=SURFACE, linewidth=0.8, height=0.65, zorder=3)

    xmax = np.nanmax(r2.values) if not np.all(np.isnan(r2.values)) else 0.0
    for yi, val in zip(y, r2.values):
        label = "–" if np.isnan(val) else f"{val:.3f}"
        xpos = 0.0 if np.isnan(val) else val
        ax.text(xpos + xmax * 0.02, yi, label, va="center", ha="left", fontsize=8.5, color=INK_2)

    ax.set_yticks(y)
    ax.set_yticklabels(r2.index, fontsize=9, color=INK_2)
    ax.invert_yaxis()  # largest R^2 at top
    ax.set_xlim(0, max(0.05, xmax * 1.25))
    ax.set_xlabel("R² with _adopted  (squared pairwise-complete Pearson r)", fontsize=10, color=INK_2)
    ax.set_title(
        "43-village analysis sample — variance in adoption explained by each feature",
        fontsize=13,
        color=INK,
        loc="left",
        pad=10,
    )
    ax.tick_params(colors=INK_2, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(HAIRLINE)
    ax.grid(True, axis="x", color=HAIRLINE, linewidth=0.6, zorder=0)

    extras = sorted(set(villages) & set(EXTRA_MF_VILLAGES))
    extras_note = f"  Includes {len(extras)} village(s) outside the 43-village analysis sample: {extras}." if extras else ""
    note = f"n={len(pooled)} households across {pooled['village'].nunique()} villages.{extras_note}  " + _CORR_NOTE
    ax.text(
        0.0,
        -0.20,
        "\n".join(textwrap.wrap(note, width=95)),
        transform=ax.transAxes,
        fontsize=7.8,
        color=MUTED,
        ha="left",
        va="top",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    return fig


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--village", type=int, default=1)
    p.add_argument(
        "--view",
        default="outcome",
        choices=("outcome", "hops", "panels", "all", "takeup", "horizon", "corr", "corr-all", "var-explained"),
    )
    p.add_argument("--network", default="allVillageRelationships", choices=NETWORK_TYPES)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--out", type=Path, default=None, help="output path (default figures/village_<v>_<view>.png)")
    p.add_argument("--seed", type=int, default=7, help="layout seed")
    p.add_argument("--dpi", type=int, default=170)
    p.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="with --view takeup, restrict to villages whose observation horizon equals this trimester",
    )
    a = p.parse_args(argv)

    if a.view in ("takeup", "horizon", "corr-all", "var-explained"):
        if a.view == "takeup":
            fig = plot_takeup_by_village(root=a.root, horizon=a.horizon)
            name = "takeup_by_village" if a.horizon is None else f"takeup_by_village_horizon{a.horizon}"
        elif a.view == "horizon":
            fig = plot_takeup_horizon_histogram(root=a.root)
            name = "takeup_horizon_histogram"
        elif a.view == "corr-all":
            fig = plot_feature_correlation_all_villages(root=a.root, network_type=a.network)
            name = "feature_correlation_all_villages"
        else:
            fig = plot_variance_explained_all_villages(root=a.root, network_type=a.network)
            name = "variance_explained_all_villages"
        out = a.out or Path("figures") / f"{name}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=a.dpi, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")
        return 0

    v = load_village(a.village, root=a.root, network_type=a.network)
    for w in v.warnings:
        print(f"  ! {w}")

    if a.view == "corr":
        fig = plot_feature_correlation_village(v)
    else:
        fig = plot_village(v, view=a.view, seed=a.seed)
    out = a.out or Path("figures") / f"village_{a.village}_{a.view}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=a.dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
