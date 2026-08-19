"""One design pair, every model that has been run under it, on one pair of axes.

	python -m src.full_llm_model.model_comparison                      # both figures
	python -m src.full_llm_model.model_comparison --view adoption      # just one
	python -m src.full_llm_model.model_comparison --pair A1B0C0D2-A0B0D2

`analysis.py` splits on the agent first and compares designs inside a folder,
which is the right split when the question is what a design does. These two
figures ask the other question -- what the *model* does -- and so hold the
design pair fixed and put one hue per model on one axis. A model comparison is
only a comparison at all when the designs are identical, which is why the pair
is a single argument here and not a loop.

They land in ``figures/full_llm/`` directly rather than in an agent folder: a
figure that spans every agent belongs to none of them.

**adoption_rates** -- the same five splits `analysis.plot_adoption_rates`
draws, with the same two references: the human ground truth as an ink line
(village 6 happened once, so it is a datum and not a distribution) and BCDJ's
own model as a grey bar (a simulation with a spread, so it is a bar like the
models it is measured against). The read is whether any model lands nearer the
ink line than the grey bar does; a model that does not has reproduced the
fitted logit at the price of an API bill.

**transmission_rates** -- the edge-level step, against BCDJ's published qN and
qP (`Main_models_1_3.m`, pooled over 43 villages). These are the paper's point
estimates for exactly two of the five groups: qP is the probability an adopter
speaks over an eligible edge in a round, qN the probability a non-adopter does.
So they are drawn solid over those two groups and dashed across the rest, where
the paper's model makes no separate claim -- a leader and a non-leader transmit
alike under BCDJ, and the all-edges bar is a mixture whose weights are the
model's own adoption level. There is still no ground truth for this step: the
bundle records no who-told-whom, so qN/qP are the paper's estimate to be
compared against, not a measurement to be scored on.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are written, not shown
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

try:
	from ..hybrid_model.game_master import (
		LLMs,
		QN,
		QP,
		VILLAGE,
		ground_truth_rates,
	)
	from .analysis import (
		BASELINE,
		BASELINE_PAIR,
		FIGURE_DIR,
		INK,
		INK_2,
		OVERFLOW,
		SERIES,
		SURFACE,
		_footnote,
		_grouped_bars,
		_pair_legend,
		_save,
		_style_axis,
		agent_root,
		bcdj_baseline,
		load_full_runs,
	)
	from .game_master import OUTPUT_DIR, parse_pair_slug
except ImportError:  # running as a script, not a package
	import sys

	sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
	from full_llm_model.analysis import (  # type: ignore[no-redef]
		BASELINE,
		BASELINE_PAIR,
		FIGURE_DIR,
		INK,
		INK_2,
		OVERFLOW,
		SERIES,
		SURFACE,
		_footnote,
		_grouped_bars,
		_pair_legend,
		_save,
		_style_axis,
		agent_root,
		bcdj_baseline,
		load_full_runs,
	)
	from full_llm_model.game_master import OUTPUT_DIR, parse_pair_slug  # type: ignore[no-redef]
	from hybrid_model.game_master import (  # type: ignore[no-redef]
		LLMs,
		QN,
		QP,
		VILLAGE,
		ground_truth_rates,
	)

# The design pair these two figures are about. Every model on the axis was run
# under this and nothing else -- see the module docstring.
DEFAULT_PAIR = "A1B0C1D0-A0B0D2"


# --------------------------------------------------------------------------
# Which models have actually been run under this pair
# --------------------------------------------------------------------------


def available_models(
	pair: str = DEFAULT_PAIR,
	village: int = VILLAGE,
	output_dir: Path | str = OUTPUT_DIR,
) -> list[LLMs]:
	"""Every `LLMs` member with at least one replicate logged under `pair`.

	In declaration order, which is roughly generation order and is the order the
	hues are handed out in, so adding a model later does not recolour the ones
	already in a write-up. The check is for an adoption log rather than for the
	folder: a directory created by a run that died before its first write is not
	a model that has been run.
	"""
	found = []
	for model in LLMs:
		pair_dir = agent_root(model, output_dir) / pair
		if pair_dir.is_dir() and any(pair_dir.glob(f"v{village}_rep*_adoption.csv")):
			found.append(model)
	return found


def load_models(
	models: list[LLMs] | None = None,
	pair: str = DEFAULT_PAIR,
	village: int = VILLAGE,
	output_dir: Path | str = OUTPUT_DIR,
	root: Path | str | None = None,
) -> pd.DataFrame:
	"""One `rates` frame over every model, with `pair` relabelled to the model.

	`load_full_runs` already reports every column these two figures need, on the
	denominators its own docstring states; the only change is what the `pair`
	column carries. Downstream (`_grouped_bars`, `_pair_legend`) groups on that
	column and does not care what it names, so relabelling it is what turns a
	design comparison into a model comparison -- there is no second loader.
	"""
	parse_pair_slug(pair)  # a typo in the slug should fail here, not silently find nothing
	models = models if models is not None else available_models(pair, village, output_dir)
	if not models:
		raise FileNotFoundError(f"no full-LLM logs for {pair} under {output_dir} (village {village})")

	frames = []
	for model in models:
		rates, _ = load_full_runs(
			model=model, village=village, output_dir=output_dir, root=root, pairs=(pair,)
		)
		rates = rates.copy()
		rates["model"] = model.value
		rates["pair"] = model.value  # the axis groups on `pair`; here one pair is one model
		frames.append(rates)
	return pd.concat(frames, ignore_index=True)


def model_colours(models: list[str]) -> dict[str, str]:
	"""One hue per model, in `SERIES` order, never cycled.

	Past the palette's five slots every extra model takes the same grey, exactly
	as `analysis.design_colours` does: six lines that no longer separate under
	CVD is a worse figure than two figures of five.
	"""
	return {model: (SERIES[i] if i < len(SERIES) else OVERFLOW) for i, model in enumerate(models)}


def _model_order(rates: pd.DataFrame) -> list[str]:
	"""The models present, in `LLMs` declaration order rather than alphabetical."""
	present = set(rates["pair"])
	return [m.value for m in LLMs if m.value in present]


def _title(village: int, pair: str, question: str) -> str:
	return f"Village {village}, {pair} -- {question}"


# --------------------------------------------------------------------------
# Plot 1: adoption, against the humans and against the paper's own model
# --------------------------------------------------------------------------


def plot_model_adoption_rates(
	rates: pd.DataFrame,
	pair: str = DEFAULT_PAIR,
	village: int = VILLAGE,
	root: Path | str | None = None,
	outfile: Path | None = None,
	dpi: int = 200,
	baseline: pd.DataFrame | None = None,
	footnote: bool = True,
) -> plt.Figure:
	"""Overall / leader / non-leader adoption and the two conditional accuracies, per model."""
	gt = ground_truth_rates(village, root=root)
	models = _model_order(rates)
	if baseline is None:
		baseline = bcdj_baseline(village=village, root=root)
	if BASELINE_PAIR not in set(rates["pair"]):  # never add it twice
		rates = pd.concat([baseline, rates], ignore_index=True)
	# Reference first, so it is the leftmost bar of every group and the first
	# entry of the key: the datum is read before the thing being measured.
	colours = {BASELINE_PAIR: BASELINE, **model_colours(models)}

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
	ax.set_title(_title(village, pair, "final adoption level by model, against the humans and the paper"),
				 fontsize=13, color=INK, loc="left", pad=38)
	extra = [Line2D([], [], color=INK, lw=1.6, label="human ground truth (empirical)")]
	_pair_legend(ax, colours, rates, extra)
	caption = (
		f"every model on this axis was run under {pair} and nothing else, which is what makes the comparison one. "
		"bars are the mean over replicates, whiskers +/-1 SE of that mean (a model at one replicate gets none). "
		f"{BASELINE_PAIR} is the paper's own prediction -- BCDJ's information model, diffusion_model.m unchanged, "
		"under their pooled 43-village logit, simulated on this village's real network; its whisker is +/-1 SD "
		"over seeds, the spread of one realisation rather than the precision of a mean. denominator: giant "
		f"component, n={gt['n']}. TPR and TNR are sensitivity and specificity against v.mf on the same "
		"population -- a model that adopts everyone scores TPR 1.0 and TNR 0.0, so the pair only means "
		"something read together.")
	bottom = _footnote(fig, caption) if footnote else 0.0
	fig.tight_layout(rect=(0.0, bottom, 1.0, 1.0))
	_save(fig, outfile, dpi)
	return fig


# --------------------------------------------------------------------------
# Plot 2: transmission, against BCDJ's published qN and qP
# --------------------------------------------------------------------------


def plot_model_transmission_rates(
	rates: pd.DataFrame,
	pair: str = DEFAULT_PAIR,
	village: int = VILLAGE,
	outfile: Path | None = None,
	dpi: int = 200,
	qN: float = QN,
	qP: float = QP,
	footnote: bool = True,
) -> plt.Figure:
	"""Transmission rate over eligible edges, per model, against the paper's qN and qP."""
	models = _model_order(rates)
	colours = model_colours(models)
	groups = [
		("tx_rate", "all eligible\nedges"),
		("tx_leader_rate", "leader\nsenders"),
		("tx_non_leader_rate", "non-leader\nsenders"),
		("tx_adopter_rate", "senders who\njoined"),
		("tx_non_adopter_rate", "senders who\ndid not"),
	]
	fig, ax = plt.subplots(figsize=(max(9.5, 1.3 * len(colours) + 7.0), 5.0), facecolor=SURFACE)
	half = _grouped_bars(ax, rates, groups, colours)

	# qP and qN are claims about two of these five groups and only those two.
	# Solid over the group the paper names, dashed across the rest: the dashes
	# are there so the other three bars can be read against the same levels
	# without the figure asserting that BCDJ predicted them.
	for level, name, group in ((qP, "qP", 3), (qN, "qN", 4)):
		ax.axhline(level, color=INK_2, linewidth=1.0, linestyle=(0, (5, 4)), zorder=3)
		ax.plot([group - half, group + half], [level, level], color=INK, linewidth=1.6, zorder=6)
		ax.annotate(f"{name} {level:.0%}", (len(groups) - 0.42, level), textcoords="offset points",
					xytext=(0, 4), ha="right", fontsize=8, color=INK, zorder=6)

	# A rate axis, so the frame stays 0-100% rather than zooming to whatever
	# these models happened to do: this figure is meant to be read next to the
	# per-agent ones, and a rescaled axis would make a quiet model look loud.
	_style_axis(ax, "transmission rate (of eligible edges)")
	ax.set_title(_title(village, pair, "who tells whom, by model, against the paper's qN and qP"),
				 fontsize=13, color=INK, loc="left", pad=38)
	extra = [Line2D([], [], color=INK, lw=1.6, label="BCDJ estimate (solid where it applies)")]
	_pair_legend(ax, colours, rates, extra)

	logged = rates.loc[rates["tx_edges"] > 0]  # a replicate with no edge log is not a zero-edge replicate
	edges = logged.groupby("pair")["tx_edges"].mean()
	landed = logged.groupby("pair")["tx_landed_rate"].mean()
	volume = "; ".join(
		f"{model}: {edges[model]:.0f} edges/replicate, {landed[model]:.0%} of them landed"
		for model in models if model in edges.index
	)
	unusable = int(rates["tx_unusable"].sum())
	caption = (
		f"every model on this axis was run under {pair} and nothing else. eligible edge: informed sender, "
		"never-informed target -- the only edge the model is ever asked about. errored or unparseable calls "
		f"stay in the denominator as non-transmissions ({unusable} of {int(rates['tx_edges'].sum())} calls). "
		f"{volume}. qN={qN:.2f} and qP={qP:.2f} are BCDJ's published point estimates (Main_models_1_3.m, pooled "
		"over 43 villages) for a non-adopter's and an adopter's per-round chance of speaking over an edge; they "
		"say nothing about leader against non-leader, and the all-edges bar is a mixture of the two weighted by "
		"the model's own adoption level. no ground truth exists for this step: the bundle records no "
		"who-told-whom, so this is a comparison against the paper's estimate, not a score.")
	bottom = _footnote(fig, caption) if footnote else 0.0
	fig.tight_layout(rect=(0.0, bottom, 1.0, 1.0))
	_save(fig, outfile, dpi)
	return fig


# --------------------------------------------------------------------------
# The entry point
# --------------------------------------------------------------------------


def write_model_comparison(
	pair: str = DEFAULT_PAIR,
	village: int = VILLAGE,
	output_dir: Path | str = OUTPUT_DIR,
	root: Path | str | None = None,
	figure_dir: Path | str = FIGURE_DIR,
	views: tuple[str, ...] = ("adoption", "transmission"),
	models: list[LLMs] | None = None,
	dpi: int = 200,
	footnote: bool = True,
) -> list[Path]:
	"""Write the two cross-model figures into ``figures/full_llm/`` directly.

	The pair is in the filename because it is the thing held fixed: two pairs
	drawn into the same folder are two different comparisons, and neither
	should overwrite the other.
	"""
	rates = load_models(models=models, pair=pair, village=village, output_dir=output_dir, root=root)
	present = _model_order(rates)
	missing = [m.value for m in LLMs if m.value not in set(present)]
	print(f"{pair}: {len(present)} model(s) -- {', '.join(present)}")
	if missing:
		print(f"  not run under this pair: {', '.join(missing)}")

	out_dir = Path(figure_dir)
	written: list[Path] = []
	if "adoption" in views:
		path = out_dir / f"v{village}_{pair}_models_adoption_rates.png"
		plt.close(plot_model_adoption_rates(rates, pair=pair, village=village, root=root, outfile=path,
											dpi=dpi, footnote=footnote))
		written.append(path)
	if "transmission" in views:
		path = out_dir / f"v{village}_{pair}_models_transmission_rates.png"
		plt.close(plot_model_transmission_rates(rates, pair=pair, village=village, outfile=path,
												dpi=dpi, footnote=footnote))
		written.append(path)
	return written


def main(argv: list[str] | None = None) -> int:
	p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	p.add_argument("--village", type=int, default=VILLAGE)
	p.add_argument("--pair", default=DEFAULT_PAIR,
				   help=f"design-pair folder held fixed across models (default {DEFAULT_PAIR})")
	p.add_argument("--view", default="all", choices=("all", "adoption", "transmission"))
	p.add_argument("--model", action="append", default=None, choices=[m.value for m in LLMs],
				   help="restrict to these models; repeatable, default every model run under the pair")
	p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
	p.add_argument("--root", type=Path, default=None)
	p.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
	p.add_argument("--dpi", type=int, default=200)
	p.add_argument("--no-footnote", dest="footnote", action="store_false",
				   help="draw the axes without the caption block underneath")
	a = p.parse_args(argv)

	views = ("adoption", "transmission") if a.view == "all" else (a.view,)
	write_model_comparison(
		pair=a.pair.upper(), village=a.village, output_dir=a.output_dir, root=a.root,
		figure_dir=a.figure_dir, views=views,
		models=[LLMs(m) for m in a.model] if a.model else None, dpi=a.dpi, footnote=a.footnote,
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
