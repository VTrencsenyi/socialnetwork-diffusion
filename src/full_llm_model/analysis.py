"""Reading the full-LLM runs back: timing, level, and who talks to whom.

	python -m src.full_llm_model.analysis                          # every folder, all three views
	python -m src.full_llm_model.analysis --view timing            # just one view
	python -m src.full_llm_model.analysis --pair A1B0C0D2-A0B0D2   # just one design pair

Figures mirror the logs: ``figures/full_llm/<agent>/<adoption>-<transmission>/``
against ``output/full_llm/<agent>/<adoption>-<transmission>/``, the same three
components in the same order, so a folder of figures is findable from the run
that produced it and vice versa.

The agent is the outer split because every number here is a property of the
model that produced it: two agents run on the same designs are two populations,
not two replicates, and averaging across them would hide the only comparison
the split makes available. The design pair is the inner split because it is the
unit that was run -- a transmission design under two adoption designs is two
runs sharing nothing but a name.

Each pair folder holds that pair alone. ``figures/full_llm/<agent>/`` also
holds one set drawn over every pair that agent has been run under, one hue
each, which is the cross-design read the per-pair folders cannot give: same
figure, same axes, so the designs are compared and not just displayed.

Three figures, one question each:

**adoption_over_time** -- the full model decides *when* as well as whether:
adoption can only reach a household after transmission has, so a design that
lands the right final level by informing everyone in round 1 is not the same
model as one that reproduces the empirical ramp. Cumulative adoption is drawn
against cumulative informed (dotted, same colour) so an early plateau can be
read for which of the two steps stalled.

**adoption_rates** -- the final level, split three ways the population rate
alone cannot separate: leaders against non-leaders (ground truth has leaders
adopt *less*), and the two conditional accuracies. A design that adopts
almost everyone scores a near-perfect true-positive rate and a near-zero
true-negative rate, so the pair is only meaningful read together: they are
sensitivity and specificity against ``v.mf``, not two independent scores.

**transmission_rates** -- the edge-level step, which has no ground truth at
all. The bundle records ties, leaders and final take-up; it contains no record
of who actually told whom (`src/plots.py`, module docstring). So this figure
is face validity only: does the agent talk more when it has joined than when
it has not, and do leaders -- who were asked by the MFI to spread the word --
talk more than everyone else? Both orderings are what the instrument implies;
neither is a fit to data.

Denominators
------------
Adoption rates are on the whole giant component, the denominator
``ground_truth_rates`` states -- a household never informed was never asked,
never appears in the log, and counts as not adopted, exactly as the run's own
``RunResult.adopted`` leaves it.

Transmission rates are on *eligible directed edges*: informed sender, never-
informed target, which is the only edge the model ever elicits. An edge whose
call errored or whose answer would not parse stays in the denominator as a
non-transmission, because that is what the run itself did with it.
"""

from __future__ import annotations

import argparse
import re
import textwrap
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are written, not shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

try:
	from .. import data_loader as dl
	from ..hybrid_model.game_master import (
		LLMs,
		PARSING_ERROR,
		VILLAGE,
		adjacency_matrix,
		bcdj_run,
		build_village,
		covariates,
		fit_betas,
		ground_truth_rates,
		logit_p,
		population,
	)
	from .game_master import OUTPUT_DIR, parse_pair_slug
except ImportError:  # running as a script, not a package
	import sys

	sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
	import data_loader as dl  # type: ignore[no-redef]
	from full_llm_model.game_master import OUTPUT_DIR, parse_pair_slug  # type: ignore[no-redef]
	from hybrid_model.game_master import (  # type: ignore[no-redef]
		LLMs,
		PARSING_ERROR,
		VILLAGE,
		adjacency_matrix,
		bcdj_run,
		build_village,
		covariates,
		fit_betas,
		ground_truth_rates,
		logit_p,
		population,
	)

# House palette (`src/plots.py`, `src/hybrid_model/analysis.py`) -- kept
# identical rather than re-derived, so a figure from this module sits next to
# theirs without a visible seam.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
HAIRLINE = "#e1e0d9"

# One hue per design pair, assigned in this fixed order and never cycled. The
# first two are the house PARTICIPANT/INFO; the rest continue the reference
# categorical order. Validated as a set for the light surface:
#   node validate_palette.js "#eb6834,#2a78d6,#1baf7a,#eda100,#e87ba4,#008300" \
#       --mode light --surface "#fcfcfb"   -> all checks pass
# Three slots WARN on contrast against the light surface, so the relief rule
# applies and every bar carries a visible value label.
#
# The sixth slot is the reference order's green, added when a sixth *model* was
# run under the shared design pair. Appending rather than re-stepping is what
# keeps the first five where they were: `model_colours` hands hues out by index,
# so no figure already drawn changes colour. The grey below is the wrong answer
# for a model -- `BASELINE` is also grey, and the paper's bar sits in the same
# adoption figure, so a sixth model in `OVERFLOW` would read as a second
# reference rather than as a result.
SERIES = ("#eb6834", "#2a78d6", "#1baf7a", "#eda100", "#e87ba4", "#008300")
OVERFLOW = MUTED  # a 7th design onward: one grey, labelled as unresolved

# The paper's own model gets the palette's neutral slot rather than a hue out of
# SERIES: it is not one more design competing for a colour, it is the thing the
# designs are measured against, and a grey bar next to an ink ground-truth line
# reads as reference on sight. `src/hybrid_model/analysis.py` and the pilots'
# `SAMPLE_COLOURS["none"]` use the same grey.
BASELINE = "#c3c2b7"
# Free to run -- no API call -- so a tight reference distribution costs nothing.
BASELINE_SEEDS = 30
# Its name on the axis and in the key. Not a design label: nothing parses it,
# and no `A.B.C.D.` slug can collide with it.
BASELINE_PAIR = "BCDJ logit"

FIGURE_DIR = Path("figures/full_llm")

_REP = re.compile(r"^v(?P<village>\d+)_rep(?P<rep>\d+)_adoption\.csv$")


# --------------------------------------------------------------------------
# Where an agent's runs and its figures live
# --------------------------------------------------------------------------


def agent_slug(model: LLMs) -> str:
	"""The folder name for one agent, e.g. ``gpt_5.4_nano``.

	`write_result` names the run directory from the model *value*, so this
	mirrors that rather than `LLMs.name`, which differs (`gpt_5_4_nano`).
	"""
	return model.value.replace("-", "_")


def agent_root(model: LLMs, output_dir: Path | str = OUTPUT_DIR) -> Path:
	"""The run directory for one agent, tolerating the older `LLMs.name` layout."""
	root = Path(output_dir) / agent_slug(model)
	if not root.is_dir():
		legacy = Path(output_dir) / model.name.lower()
		if legacy.is_dir():
			return legacy
	return root


# --------------------------------------------------------------------------
# The population every rate is stated on
# --------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _village_frame(village: int, root: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""`(hh_ids, is_leader, truth)` in run index order, giant component only.

	`truth` is `v.mf` -- read here, by the scorer, and nowhere near a prompt.
	The length check is the guard that the log's population and the ground
	truth's population are the same pruning of the same village.
	"""
	root_path = Path(root) if root is not None else None
	leaders, households = build_village(village, root=root_path)
	pop = population(leaders, households)
	hh_ids = np.array([a.hh_id for a in pop], dtype=int)
	is_leader = np.array([a.is_leader for a in pop], dtype=bool)

	v = dl.load_village(village, root=root_path if root_path is not None else dl.DEFAULT_ROOT)
	keep = v.in_giant.astype(bool)
	truth = v.mf[keep].astype(bool)
	if len(truth) != len(hh_ids):
		raise RuntimeError(f"v{village}: {len(truth)} ground-truth rows but {len(hh_ids)} agents; giant-only mismatch")
	return hh_ids, is_leader, truth


# --------------------------------------------------------------------------
# Loading the full-LLM runs
# --------------------------------------------------------------------------


def _rate(mask: np.ndarray, values: np.ndarray) -> float:
	"""`values[mask].mean()`, or NaN where the subgroup is empty."""
	return float(values[mask].mean()) if mask.any() else float("nan")


def load_full_runs(
	model: LLMs = LLMs.GPT_5_4_NANO,
	village: int = VILLAGE,
	output_dir: Path | str = OUTPUT_DIR,
	root: Path | str | None = None,
	pairs: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Every replicate of every design pair for one agent, as `(rates, curves)`.

	`rates` is one row per (adoption design, transmission design, replicate):
	the final adoption level three ways, the two conditional accuracies, and
	the edge-level transmission rates. `curves` is one row per (design pair,
	replicate, round): cumulative adoption and cumulative informed, both on
	the whole-population denominator.

	`pairs` restricts the read to those folder slugs (`A1B0C0D2-A0B0D2`);
	the default reads every pair the agent has been run under.

	A replicate's two logs are read together and a missing transmission log
	leaves that replicate's transmission columns NaN rather than dropping the
	adoption side, which is still a complete answer to the level question.
	"""
	hh_ids, is_leader, truth = _village_frame(village, str(root) if root is not None else None)
	index_of = {int(hh): i for i, hh in enumerate(hh_ids)}
	n = len(hh_ids)

	rate_rows: list[dict[str, object]] = []
	curve_rows: list[dict[str, object]] = []
	model_root = agent_root(model, output_dir)
	wanted = {p.upper() for p in pairs} if pairs is not None else None
	for pair_dir in sorted(p for p in model_root.glob("*") if p.is_dir()):
		try:
			adoption_design, transmission_design = parse_pair_slug(pair_dir.name)
		except ValueError:  # not a design-pair folder; leave it alone
			continue
		if wanted is not None and pair_dir.name.upper() not in wanted:
			continue
		for adoption_path in sorted(pair_dir.glob(f"v{village}_rep*_adoption.csv")):
			match = _REP.match(adoption_path.name)
			if not match:
				continue
			replicate = int(match["rep"])
			pair = f"{adoption_design} x {transmission_design}"

			adoption = pd.read_csv(adoption_path, usecols=["round", "hh_id", "joined"])
			adopted = np.zeros(n, dtype=bool)
			adopted_round = np.full(n, -1, dtype=int)
			asked_round = np.full(n, -1, dtype=int)
			for round_r, hh, joined in zip(adoption["round"], adoption["hh_id"], adoption["joined"]):
				idx = index_of.get(int(hh))
				if idx is None:  # a household outside this village's giant component
					continue
				asked_round[idx] = int(round_r)
				if joined:
					adopted[idx], adopted_round[idx] = True, int(round_r)

			transmission_path = adoption_path.with_name(adoption_path.name.replace("_adoption.csv", "_transmission.csv"))
			tx = None
			if transmission_path.exists():
				tx = pd.read_csv(
					transmission_path,
					usecols=["round", "sender_hh_id", "target_hh_id", "sender_adopted", "transmitted",
							 "landed", "decision", "error"],
				)

			# Informed comes from the *transmission* log, not from who was
			# asked: a household reached in the last round is informed and
			# never gets asked, so the adoption log alone would drop it.
			# Leaders are informed by the MFI before round 1.
			informed_round = np.where(is_leader, 0, -1)
			if tx is not None and len(tx):
				for round_r, hh, landed in zip(tx["round"], tx["target_hh_id"], tx["landed"]):
					idx = index_of.get(int(hh))
					if idx is not None and landed and informed_round[idx] < 0:
						informed_round[idx] = int(round_r)
			else:  # no edge log: fall back to the round each household was asked
				informed_round = np.where(is_leader, 0, asked_round)

			rounds = max(
				int(adoption["round"].max()) if len(adoption) else 0,
				int(tx["round"].max()) if tx is not None and len(tx) else 0,
			)
			for round_r in range(1, rounds + 1):
				reached = (informed_round >= 0) & (informed_round <= round_r)
				joined_by_now = adopted & (adopted_round <= round_r)
				curve_rows.append({
					"pair": pair,
					"adoption_design": adoption_design,
					"transmission_design": transmission_design,
					"replicate": replicate,
					"round": round_r,
					"informed": float(reached.sum()) / n,
					"adopted": float(joined_by_now.sum()) / n,
				})

			row: dict[str, object] = {
				"pair": pair,
				"adoption_design": adoption_design,
				"transmission_design": transmission_design,
				"replicate": replicate,
				"n": n,
				"rate": float(adopted.mean()),
				"leader_rate": _rate(is_leader, adopted),
				"non_leader_rate": _rate(~is_leader, adopted),
				"tpr": _rate(truth, adopted),
				"tnr": _rate(~truth, ~adopted),
			}

			tx_columns = {
				"tx_rate": np.nan, "tx_leader_rate": np.nan, "tx_non_leader_rate": np.nan,
				"tx_adopter_rate": np.nan, "tx_non_adopter_rate": np.nan,
				"tx_landed_rate": np.nan, "tx_edges": 0, "tx_unusable": 0,
			}
			if tx is not None and len(tx):
				transmitted = tx["transmitted"].astype(bool).to_numpy()
				landed = tx["landed"].astype(bool).to_numpy()
				sender_adopted = tx["sender_adopted"].astype(bool).to_numpy()
				sender_leader = np.array(
					[bool(is_leader[index_of[int(hh)]]) if int(hh) in index_of else False for hh in tx["sender_hh_id"]]
				)
				# What the run itself could not use: a failed call, or an
				# answer that never parsed. Both were counted as "did not
				# transmit" by the run, so they stay in the denominator
				# here too -- but the reader is told how many there were.
				unusable = (
					(tx["error"].notna() & tx["error"].astype(str).str.strip().ne(""))
					| tx["decision"].astype(str).str.strip().eq(PARSING_ERROR)
				)
				tx_columns = {
					"tx_rate": float(transmitted.mean()),
					"tx_leader_rate": _rate(sender_leader, transmitted),
					"tx_non_leader_rate": _rate(~sender_leader, transmitted),
					"tx_adopter_rate": _rate(sender_adopted, transmitted),
					"tx_non_adopter_rate": _rate(~sender_adopted, transmitted),
					"tx_landed_rate": float(landed.mean()),
					"tx_edges": int(len(tx)),
					"tx_unusable": int(unusable.sum()),
				}
			row.update(tx_columns)
			rate_rows.append(row)

	if not rate_rows:
		raise FileNotFoundError(f"no full-LLM logs under {model_root} for village {village}")
	rates = pd.DataFrame(rate_rows).sort_values(["pair", "replicate"], ignore_index=True)
	curves = pd.DataFrame(curve_rows).sort_values(["pair", "replicate", "round"], ignore_index=True)
	return rates, curves



# --------------------------------------------------------------------------
# Shared drawing bits
# --------------------------------------------------------------------------


def design_colours(pairs: list[str]) -> dict[str, str]:
	"""One hue per design pair, in `SERIES` order, never cycled.

	Past the palette's six slots the extra designs all take one grey. That is
	deliberately unhelpful: the fix is to plot fewer design pairs at a time,
	not to invent a seventh hue that no longer separates under CVD.
	"""
	return {pair: (SERIES[i] if i < len(SERIES) else OVERFLOW) for i, pair in enumerate(pairs)}


def _title(village: int, question: str, pairs: list[str]) -> str:
	"""The title, naming the design pair when the axes carry only one.

	In a per-pair folder the hue no longer distinguishes anything, so the pair
	moves into the title where it still identifies the figure once the file has
	been pulled out of its folder and into the write-up.
	"""
	subject = f"Village {village}" if len(pairs) != 1 else f"Village {village}, {pairs[0]}"
	return f"{subject} -- {question}"


def _style_axis(ax, ylabel: str, ymax: float = 1.0) -> None:
	ax.set_facecolor(SURFACE)
	ax.set_ylabel(ylabel, fontsize=9.5, color=INK_2, labelpad=8)
	ax.set_ylim(0.0, ymax + 0.06)  # headroom so a bar near 100% keeps its label and whisker
	ax.set_yticks(np.arange(0.0, ymax + 1e-9, 0.2))
	ax.set_yticklabels([f"{tick:.0%}" for tick in np.arange(0.0, ymax + 1e-9, 0.2)])
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
	which happened once. A design at one replicate has no SE at all and gets
	no whisker rather than a whisker of zero -- a single draw should not look
	like a converged one.
	"""
	values = pd.Series(values).to_numpy(dtype=float)
	values = values[np.isfinite(values)]
	if not len(values):
		return float("nan"), float("nan")
	if len(values) < 2:
		return float(values[0]), float("nan")
	return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def mean_sd(values) -> tuple[float, float]:
	"""Mean and standard deviation -- what the paper's model's spread means here.

	Not the SE of its mean: 30 seeds pin that mean down to nothing, and the
	number the baseline stands in for is *one* realisation of village 6. The SD
	is how far a single run of the fitted logit lands from its own centre, which
	is the quantity a design's village is being compared against.
	"""
	values = pd.Series(values).to_numpy(dtype=float)
	values = values[np.isfinite(values)]
	if not len(values):
		return float("nan"), float("nan")
	return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _bar_with_se(ax, x: float, values, colour: str, width: float, spread: str = "se") -> float:
	"""One design's bar in one group: the mean, with a +/-1 SE whisker on top.

	`spread="sd"` draws +/-1 SD instead, which is what the paper's model's bar
	carries -- see `mean_sd`. The whisker is labelled in the footnote either way,
	because the two are not the same claim.
	"""
	mean, se = mean_se(values) if spread == "se" else mean_sd(values)
	if not np.isfinite(mean):
		return float("nan")
	ax.bar(x, mean, width=width, color=colour, alpha=0.9, linewidth=0, zorder=2)
	top = mean
	if np.isfinite(se) and se > 0:
		ax.errorbar(x, mean, yerr=se, fmt="none", ecolor=INK_2, elinewidth=1.1, capsize=3.0,
					capthick=1.1, zorder=4)
		top = mean + se
	# Relief rule: three of the five hues sit below 3:1 on this surface, so the
	# value is always legible as text and never by colour alone. It is knocked out
	# of whatever is behind it, because a ground-truth line at the same height as
	# a bar's label would otherwise be struck straight through the digits.
	ax.annotate(f"{mean:.0%}", (x, top), textcoords="offset points", xytext=(0, 4), ha="center",
				fontsize=8, color=INK_2, zorder=7,
				bbox=dict(boxstyle="square,pad=0.12", facecolor=SURFACE, edgecolor="none"))
	return mean


def _grouped_bars(
	ax,
	rates: pd.DataFrame,
	groups: list[tuple[str, str]],
	colours: dict[str, str],
	spreads: dict[str, str] | None = None,
) -> float:
	"""`groups` is `(column, x-label)`; one bar per design pair inside each group.

	`spreads` overrides which spread a given pair's whisker carries (`"se"` by
	default, `"sd"` for the paper's model). Returns the half-width the bars of one
	group span, so a reference line can be drawn over exactly the bars it refers to.
	"""
	pairs = list(colours)
	spreads = spreads or {}
	step = min(0.72 / max(len(pairs), 1), 0.30)  # one design should read as a bar, not a wall
	for g, (column, _) in enumerate(groups):
		for d, pair in enumerate(pairs):
			offset = (d - (len(pairs) - 1) / 2) * step
			_bar_with_se(ax, g + offset, rates.loc[rates["pair"] == pair, column], colours[pair],
						 step * 0.86, spread=spreads.get(pair, "se"))
	ax.set_xlim(-0.62, len(groups) - 0.38)
	ax.set_xticks(range(len(groups)))
	ax.set_xticklabels([label for _, label in groups], fontsize=9.5, color=INK_2, linespacing=1.5)
	ax.tick_params(axis="x", length=0, pad=7)
	return max(step * len(pairs) / 2, step * 0.7)


def _pair_legend(ax, colours: dict[str, str], rates: pd.DataFrame, extra: list[Line2D] | None = None) -> None:
	"""The design-pair key, above the axes where it can never sit on a bar."""
	reps = rates.groupby("pair")["replicate"].nunique()
	handles = [
		Line2D([], [], marker="s", ls="", ms=8, mfc=colour, mec=colour,
			   label=f"{pair}  (S={int(reps.get(pair, 0))})")
		for pair, colour in colours.items()
	]
	handles += extra or []
	ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.005), frameon=False, fontsize=8.5,
			  labelcolor=INK_2, ncols=len(handles) if len(handles) <= 4 else 3, columnspacing=1.8,
			  handletextpad=0.6,
			  borderaxespad=0.0)


# --------------------------------------------------------------------------
# The paper's own model, on the same population and the same denominators
# --------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _bcdj_baseline(village: int, seeds: int, root: str | None) -> pd.DataFrame:
	"""BCDJ's information model under the pooled 43-village logit, `seeds` times.

	This is the paper's prediction, not a second LLM design: `bcdj_run` is
	`diffusion_model.m` unchanged, its transmission step is BCDJ's own qN/qP, and
	step 1 is their fitted logit rather than a model call. The betas come from
	`fit_betas` -- pooled over all 43 villages, so village 6 is not fitted to
	itself -- and nothing here reads a run log.

	One row per seed, in the same columns and on the same denominators
	`load_full_runs` reports, so the two frames concatenate and plot on one axis
	without conversion. Unlike ground truth it *is* a distribution: the logit is
	stochastic, and 30 draws of it are what "the paper's model on this village"
	looks like as a spread rather than a point.
	"""
	root_path = Path(root) if root is not None else None
	hh_ids, is_leader, truth = _village_frame(village, root)
	leaders, households = build_village(village, root=root_path)
	pop = population(leaders, households)
	A = adjacency_matrix(pop)
	beta, _ = fit_betas(root=root_path)
	p_join = logit_p(covariates(village, root=root_path), beta)

	rows = []
	for seed in range(seeds):
		result = bcdj_run(pop, A, p_join, village=village, seed=seed, replicate=seed)
		adopted = result.adopted
		rows.append({
			"pair": BASELINE_PAIR,
			"adoption_design": BASELINE_PAIR,
			"transmission_design": BASELINE_PAIR,
			"replicate": seed,
			"n": len(hh_ids),
			"rate": float(adopted.mean()),
			"leader_rate": _rate(is_leader, adopted),
			"non_leader_rate": _rate(~is_leader, adopted),
			"tpr": _rate(truth, adopted),
			"tnr": _rate(~truth, ~adopted),
		})
	return pd.DataFrame(rows)


def bcdj_baseline(
	village: int = VILLAGE,
	seeds: int = BASELINE_SEEDS,
	root: Path | str | None = None,
) -> pd.DataFrame:
	"""`_bcdj_baseline`, cached across the many folders one CLI run plots."""
	return _bcdj_baseline(village, seeds, str(root) if root is not None else None).copy()


# --------------------------------------------------------------------------
# Plot 1: adoption over time
# --------------------------------------------------------------------------


def plot_adoption_over_time(
	curves: pd.DataFrame,
	rates: pd.DataFrame | None = None,
	village: int = VILLAGE,
	root: Path | str | None = None,
	outfile: Path | None = None,
	dpi: int = 200,
) -> plt.Figure:
	"""Cumulative adoption per round, with cumulative informed dotted behind it."""
	gt = ground_truth_rates(village, root=root)
	v = dl.load_village(village, root=root if root is not None else dl.DEFAULT_ROOT)
	empirical = v.adoption_curve()

	pairs = sorted(curves["pair"].unique())
	colours = design_colours(pairs)
	rounds = int(curves["round"].max())
	fig, ax = plt.subplots(figsize=(max(8.0, 0.9 * rounds + 4.6), 5.4), facecolor=SURFACE)

	for pair in pairs:
		sub = curves.loc[curves["pair"] == pair]
		adopted = sub.groupby("round")["adopted"].apply(lambda column: pd.Series(mean_se(column), index=["mean", "se"]))
		adopted = adopted.unstack()
		informed = sub.groupby("round")["informed"].mean()
		x = adopted.index.to_numpy(dtype=float)
		colour = colours[pair]
		if adopted["se"].notna().any():
			se = adopted["se"].fillna(0.0).to_numpy()
			ax.fill_between(x, adopted["mean"] - se, adopted["mean"] + se, color=colour, alpha=0.16,
							linewidth=0, zorder=2)
		ax.plot(x, informed.to_numpy(), color=colour, linewidth=1.3, linestyle=(0, (1, 2)), alpha=0.85, zorder=3)
		ax.plot(x, adopted["mean"].to_numpy(), color=colour, linewidth=2.0, marker="o", markersize=5.0,
				markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=4)
		for series, style in (("informed", informed.to_numpy()), ("adopted", adopted["mean"].to_numpy())):
			ax.annotate(f"{style[-1]:.0%}", (x[-1], style[-1]), textcoords="offset points", xytext=(7, 0),
						va="center", fontsize=8, color=INK_2 if series == "adopted" else MUTED, zorder=5)

	if empirical is not None:
		emp = empirical.dropna(subset=["empirical_rescaled"])
		emp = emp[emp["t"] >= 1]
		ax.plot(emp["t"], emp["empirical_rescaled"], color=INK, linewidth=1.5, marker="s", markersize=4.5,
				markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=5)
	ax.axhline(gt["all"], color=INK_2, linewidth=1.2, linestyle=(0, (5, 4)), zorder=3)
	ax.annotate(f"ground truth {gt['all']:.1%}", (rounds + 0.42, gt["all"]), textcoords="offset points",
				xytext=(0, 5), ha="right", fontsize=8, color=INK_2, zorder=5)

	ax.set_xlim(0.7, rounds + 0.45)
	ax.set_xticks(range(1, rounds + 1))
	ax.set_xlabel("round (one trimester of the BCDJ panel)", fontsize=9.5, color=INK_2, labelpad=6)
	ax.tick_params(axis="x", labelsize=9, colors=INK_2, length=0, pad=6)
	_style_axis(ax, "share of the village (cumulative)")
	ax.set_title(_title(village, "adoption per round, against how far information had reached", pairs),
				 fontsize=13, color=INK, loc="left", pad=38)

	extra = [
		Line2D([], [], color=INK_2, lw=2.0, label="adopted (solid)"),
		Line2D([], [], color=INK_2, lw=1.3, ls=(0, (1, 2)), label="informed (dotted)"),
		Line2D([], [], color=INK, lw=1.5, marker="s", ms=4.5, label="empirical (panel.dta, rescaled)"),
	]
	_pair_legend(ax, colours, rates if rates is not None else curves, extra)
	bottom = _footnote(fig,
		f"denominator: giant component, n={gt['n']} (a household never informed was never asked and counts as "
		"not adopted). informed is reconstructed from the edges that landed, so it includes the last round's "
		"targets, who are informed but never asked. empirical is panel.dta on the full census, rescaled to this "
		"village's final take-up. bands are +/-1 SE of the mean across replicates.")
	fig.tight_layout(rect=(0.0, bottom, 1.0, 1.0))
	_save(fig, outfile, dpi)
	return fig


# --------------------------------------------------------------------------
# Plot 2: adoption level, split, and the two conditional accuracies
# --------------------------------------------------------------------------


def plot_adoption_rates(
	rates: pd.DataFrame,
	village: int = VILLAGE,
	root: Path | str | None = None,
	outfile: Path | None = None,
	dpi: int = 200,
	baseline: pd.DataFrame | None = None,
) -> plt.Figure:
	"""Overall / leader / non-leader adoption, plus true-positive and true-negative rate.

	Three things on one axis: this agent's designs, the paper's own model, and the
	human ground truth. The paper's model is a bar like the designs -- it is a
	simulation with a spread, and comparing it to them as a line would flatten
	that -- while ground truth stays a line, because village 6 happened once.

	The middle bar is the whole point of the panel. A design that lands closer to
	the ink line than the grey bar does has added something the fitted logit did
	not already have from BCDJ's covariates alone; one that does not has only
	reproduced it more expensively.
	"""
	gt = ground_truth_rates(village, root=root)
	pairs = sorted(rates["pair"].unique())
	if baseline is None:
		baseline = bcdj_baseline(village=village, root=root)
	if BASELINE_PAIR not in set(rates["pair"]):  # never add it twice
		rates = pd.concat([baseline, rates], ignore_index=True)
	# Reference first, so it is the leftmost bar of every group and the first
	# entry of the key: the datum is read before the thing being measured.
	colours = {BASELINE_PAIR: BASELINE, **design_colours(pairs)}

	groups = [
		("rate", f"overall\nn={gt['n']}"),
		("leader_rate", f"leaders\nn={gt['n_leaders']}"),
		("non_leader_rate", f"non-leaders\nn={gt['n_non_leaders']}"),
		("tpr", "true positive rate\nof real adopters"),
		("tnr", "true negative rate\nof real non-adopters"),
	]
	fig, ax = plt.subplots(figsize=(max(9.5, 1.3 * len(colours) + 7.0), 5.4), facecolor=SURFACE)
	half = _grouped_bars(ax, rates, groups, colours, spreads={BASELINE_PAIR: "sd"})

	for g, level in {0: gt["all"], 1: gt["leaders"], 2: gt["non_leaders"]}.items():
		ax.plot([g - half, g + half], [level, level], color=INK, linewidth=1.6, zorder=6)
		ax.annotate(f"{level:.1%}", (g + half, level), textcoords="offset points", xytext=(4, 0),
					va="center", fontsize=7.5, color=INK, zorder=6)

	_style_axis(ax, "rate")
	ax.set_title(_title(village, "final adoption level and its two conditional accuracies", pairs),
				 fontsize=13, color=INK, loc="left", pad=38)
	extra = [Line2D([], [], color=INK, lw=1.6, label="human ground truth (empirical)")]
	_pair_legend(ax, colours, rates, extra)
	bottom = _footnote(fig,
		"bars are the mean over replicates, whiskers +/-1 SE of that mean (a design at one replicate gets none). "
		f"{BASELINE_PAIR} is the paper's own prediction -- BCDJ's information model, diffusion_model.m unchanged, "
		"under their pooled 43-village logit, simulated on this village's real network; its whisker is +/-1 SD "
		"over seeds, the spread of one realisation rather than the precision of a mean. TPR and TNR are "
		"sensitivity and specificity against v.mf on the same population -- a design that adopts everyone scores "
		"TPR 1.0 and TNR 0.0, so the pair only means something read together.")
	fig.tight_layout(rect=(0.0, bottom, 1.0, 1.0))
	_save(fig, outfile, dpi)
	return fig


# --------------------------------------------------------------------------
# Plot 3: the edge-level transmission step
# --------------------------------------------------------------------------


def plot_transmission_rates(
	rates: pd.DataFrame,
	village: int = VILLAGE,
	outfile: Path | None = None,
	dpi: int = 200,
) -> plt.Figure:
	"""Transmission rate over eligible edges, by the sender's role and its own decision."""
	pairs = sorted(rates["pair"].unique())
	colours = design_colours(pairs)
	groups = [
		("tx_rate", "all eligible\nedges"),
		("tx_leader_rate", "leader\nsenders"),
		("tx_non_leader_rate", "non-leader\nsenders"),
		("tx_adopter_rate", "senders who\njoined"),
		("tx_non_adopter_rate", "senders who\ndid not"),
	]
	# Shorter than the adoption panel: the frame stays 0-100% (see below) and a
	# tall empty panel would read as a missing series rather than a low rate.
	fig, ax = plt.subplots(figsize=(max(9.5, 1.3 * len(pairs) + 7.0), 4.6), facecolor=SURFACE)
	_grouped_bars(ax, rates, groups, colours)

	# A rate axis, so the frame stays 0-100% rather than zooming to whatever
	# this agent happened to do: two agents' folders are meant to be readable
	# side by side, and a rescaled axis would make a quiet model look loud.
	_style_axis(ax, "transmission rate (of eligible edges)")
	ax.set_title(_title(village, "who tells whom, by the sender's role and its own decision", pairs),
				 fontsize=13, color=INK, loc="left", pad=38)
	_pair_legend(ax, colours, rates)

	logged = rates.loc[rates["tx_edges"] > 0]  # a replicate with no edge log is not a zero-edge replicate
	edges = logged.groupby("pair")["tx_edges"].mean()
	landed = logged.groupby("pair")["tx_landed_rate"].mean()
	volume = "; ".join(
		f"{pair}: {edges[pair]:.0f} edges/replicate, {landed[pair]:.0%} of them landed" for pair in edges.index
	)
	unusable = int(rates["tx_unusable"].sum())
	bottom = _footnote(fig,
		"eligible edge: informed sender, never-informed target -- the only edge the model is ever asked about. "
		f"errored or unparseable calls stay in the denominator as non-transmissions ({unusable} of "
		f"{int(rates['tx_edges'].sum())} calls). {volume}. no ground truth exists for this step: the bundle "
		"records no who-told-whom, so this is face validity only.")
	fig.tight_layout(rect=(0.0, bottom, 1.0, 1.0))
	_save(fig, outfile, dpi)
	return fig


# --------------------------------------------------------------------------
# The entry point: one folder of figures per design pair, plus the comparison
# --------------------------------------------------------------------------


def pair_dir_name(pair: str) -> str:
	"""``"A1B0C0D2 x A0B0D2" -> "A1B0C0D2-A0B0D2"`` -- back to the log's folder name."""
	adoption, transmission = (part.strip() for part in pair.split(" x "))
	return f"{adoption}-{transmission}"


def _write_views(
	rates: pd.DataFrame,
	curves: pd.DataFrame,
	out_dir: Path,
	village: int,
	root: Path | str | None,
	views: tuple[str, ...],
	dpi: int,
) -> list[Path]:
	"""The three figures for whatever design pairs `rates` and `curves` carry."""
	written: list[Path] = []
	if "timing" in views:
		path = out_dir / f"v{village}_adoption_over_time.png"
		plt.close(plot_adoption_over_time(curves, rates, village=village, root=root, outfile=path, dpi=dpi))
		written.append(path)
	if "adoption" in views:
		path = out_dir / f"v{village}_adoption_rates.png"
		plt.close(plot_adoption_rates(rates, village=village, root=root, outfile=path, dpi=dpi))
		written.append(path)
	if "transmission" in views:
		path = out_dir / f"v{village}_transmission_rates.png"
		plt.close(plot_transmission_rates(rates, village=village, outfile=path, dpi=dpi))
		written.append(path)
	return written


def write_agent_figures(
	model: LLMs = LLMs.GPT_5_4_NANO,
	village: int = VILLAGE,
	output_dir: Path | str = OUTPUT_DIR,
	root: Path | str | None = None,
	figure_dir: Path | str = FIGURE_DIR,
	views: tuple[str, ...] = ("timing", "adoption", "transmission"),
	pairs: tuple[str, ...] | None = None,
	scope: tuple[str, ...] = ("pair", "agent"),
	dpi: int = 200,
) -> list[Path]:
	"""Write one agent's figures under ``figures/full_llm/<agent>/``.

	Two scopes, and by default both. ``pair`` writes one folder per design
	pair, named as its log folder is, holding that pair alone -- which is what
	a write-up cites, and what a single terminated run can produce on its own.
	``agent`` writes one set over every pair the agent has been run under, one
	hue each, which is the cross-design comparison a per-pair folder cannot
	make; a model comparison stays a between-folder read either way.

	`pairs` restricts both scopes to those folder slugs. Returns the paths
	written.
	"""
	rates, curves = load_full_runs(
		model=model, village=village, output_dir=output_dir, root=root, pairs=pairs
	)
	agent_dir = Path(figure_dir) / agent_slug(model)
	written: list[Path] = []

	if "pair" in scope:
		for pair in sorted(rates["pair"].unique()):
			written += _write_views(
				rates.loc[rates["pair"] == pair], curves.loc[curves["pair"] == pair],
				agent_dir / pair_dir_name(pair), village, root, views, dpi,
			)
	# One pair is one pair whichever folder it is drawn in, so the agent-level
	# set is only a different figure when there is more than one to compare.
	if "agent" in scope and rates["pair"].nunique() > 1:
		written += _write_views(rates, curves, agent_dir, village, root, views, dpi)
	return written


def main(argv: list[str] | None = None) -> int:
	p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	p.add_argument("--village", type=int, default=VILLAGE)
	p.add_argument("--model", default=LLMs.GPT_5_4_NANO.value, choices=[m.value for m in LLMs])
	p.add_argument("--view", default="all", choices=("all", "timing", "adoption", "transmission"))
	p.add_argument("--pair", action="append", default=None,
				   help="design-pair folder, e.g. A1B0C0D2-A0B0D2; repeatable, default every pair")
	p.add_argument("--scope", default="all", choices=("all", "pair", "agent"),
				   help="per-pair folders, the agent-level comparison, or both (default)")
	p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
	p.add_argument("--root", type=Path, default=None)
	p.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
	p.add_argument("--dpi", type=int, default=200)
	a = p.parse_args(argv)

	views = ("timing", "adoption", "transmission") if a.view == "all" else (a.view,)
	scope = ("pair", "agent") if a.scope == "all" else (a.scope,)
	write_agent_figures(
		model=LLMs(a.model), village=a.village, output_dir=a.output_dir, root=a.root,
		figure_dir=a.figure_dir, views=views, pairs=tuple(a.pair) if a.pair else None,
		scope=scope, dpi=a.dpi,
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
