"""What each pilot design cost, in tokens, per prompt design.

    python -m src.pilot.token_costs                       # both pilots, to stdout
    python -m src.pilot.token_costs --pilot transmission  # just one
    python -m src.pilot.token_costs --price-in 0.05 --price-out 0.40   # USD per 1M

The two pilots' run logs already carry the Responses API's own `usage` on
every row, so this is a read of what was actually billed rather than a
re-count of the prompts: it includes whatever the API charged for reasoning,
and it is per *call*, which is the unit both pilots' arms are stated in.

Read it per call, not per design. A design's totals here are an artefact of
how many repetitions it happened to be run at (20 in the adoption pilot, 25 x
2 informer arms in the transmission pilot); the mean input and output per call
are the numbers that carry over to a full model run, where the same design
will be called tens of thousands of times.

The D axis is what moves output tokens: D0 answers with a token, D1 (MOA)
writes three short answers first, D2 (DT) fills a decision matrix under a
JSON schema. The A/B/C/L axes move input tokens, since they are what gets
pasted into the prompt.

Prices are not hardcoded -- pass `--price-in` and `--price-out` in USD per
million tokens and the cost columns appear; without them the table stays in
tokens, which is the thing the logs actually recorded.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

PILOTS = {
    "adoption": Path("output/pilot/adoption"),
    "transmission": Path("output/pilot/transmission"),
}
USAGE_COLUMNS = ["repetition", "input_tokens", "output_tokens", "total_tokens"]
_LOG = re.compile(r"^(?P<model>.+?)_(?P<design>[A-Z]\d(?:[A-Z]\d)*)\.csv$")


def load_usage(pilot: str, output_dir: Path | None = None, model: str | None = None) -> pd.DataFrame:
    """One row per (model, design): calls, per-call means, and totals.

    Only the per-design run logs are read. `design_tests.csv`, `dt_matrices.csv`
    and the module-effect tables live in the same folder and carry no usage,
    so the design-shaped filename is the filter.
    """
    directory = output_dir if output_dir is not None else PILOTS[pilot]
    rows = []
    for path in sorted(directory.glob("*.csv")):
        match = _LOG.match(path.name)
        if not match or (model is not None and match["model"] != model):
            continue
        df = pd.read_csv(path, usecols=USAGE_COLUMNS)
        if not len(df):
            continue
        rows.append({
            "pilot": pilot,
            "model": match["model"],
            "design": match["design"],
            "reps": int(df["repetition"].nunique()),
            "calls": int(len(df)),
            "in_per_call": float(df["input_tokens"].mean()),
            "out_per_call": float(df["output_tokens"].mean()),
            "input_tokens": int(df["input_tokens"].sum()),
            "output_tokens": int(df["output_tokens"].sum()),
            "total_tokens": int(df["total_tokens"].sum()),
        })
    if not rows:
        raise FileNotFoundError(f"no per-design run logs under {directory}")
    return pd.DataFrame(rows).sort_values(["model", "design"], ignore_index=True)


def priced(table: pd.DataFrame, price_in: float | None, price_out: float | None) -> pd.DataFrame:
    """Add USD columns for the given per-million prices, if both were given."""
    if price_in is None or price_out is None:
        return table
    table = table.copy()
    table["usd"] = table["input_tokens"] / 1e6 * price_in + table["output_tokens"] / 1e6 * price_out
    table["usd_per_1k_calls"] = (
        table["in_per_call"] / 1e6 * price_in + table["out_per_call"] / 1e6 * price_out
    ) * 1000
    return table


def to_markdown(table: pd.DataFrame) -> str:
    """The per-design table, plus a totals row, as a markdown table."""
    money = "usd" in table.columns
    header = ["design", "calls", "in/call", "out/call", "input", "output", "total"]
    if money:
        header += ["USD", "USD/1k calls"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]

    def row(cells: list[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    for _, r in table.iterrows():
        cells = [
            r["design"], f"{r['calls']:,}", f"{r['in_per_call']:,.0f}", f"{r['out_per_call']:,.0f}",
            f"{r['input_tokens']:,}", f"{r['output_tokens']:,}", f"{r['total_tokens']:,}",
        ]
        if money:
            cells += [f"{r['usd']:.2f}", f"{r['usd_per_1k_calls']:.2f}"]
        lines.append(row(cells))

    calls = int(table["calls"].sum())
    totals = [
        f"**all {len(table)} designs**", f"**{calls:,}**",
        f"**{table['input_tokens'].sum() / calls:,.0f}**", f"**{table['output_tokens'].sum() / calls:,.0f}**",
        f"**{int(table['input_tokens'].sum()):,}**", f"**{int(table['output_tokens'].sum()):,}**",
        f"**{int(table['total_tokens'].sum()):,}**",
    ]
    if money:
        totals += [f"**{table['usd'].sum():.2f}**", "--"]
    lines.append(row(totals))
    return "\n".join(lines)


AXES = {"adoption": "ABCD", "transmission": "ABLD"}
AXIS_NAMES = {
    "A": "A -- own profile",
    "B": "B -- other party",
    "C": "C -- endorsement",
    "L": "L -- leader framing",
    "D": "D -- instruction",
}


def axis_means(table: pd.DataFrame, pilot: str) -> pd.DataFrame:
    """Mean tokens per call at each level of each axis, for one pilot.

    Unweighted across designs, which is what the axis question asks: what does
    turning this one knob up cost, averaged over everything else the design
    could be doing. The calls-weighted total in `to_markdown` answers a
    different question -- what the pilot as run actually spent.
    """
    table = table.copy()
    rows = []
    for i, axis in enumerate(AXES[pilot]):
        level = table["design"].str[2 * i + 1].astype(int)
        for value, group in table.groupby(level):
            rows.append({
                "pilot": pilot,
                "axis": axis,
                "level": int(value),
                "in_per_call": group["in_per_call"].mean(),
                "out_per_call": group["out_per_call"].mean(),
            })
    return pd.DataFrame(rows)


def _tex_row(cells: list[str]) -> str:
    return "    " + " & ".join(cells) + r" \\"


def to_latex_designs(tables: dict[str, pd.DataFrame]) -> str:
    """The two pilots' per-design means, side by side in one `table` float.

    Side by side rather than stacked because both pilots have 54 designs and
    the interesting read is across them: the same A/B levels cost about the
    same in either instrument, and D is what separates them.
    """
    columns = list(tables)
    frames = [tables[name].reset_index(drop=True) for name in columns]
    height = max(len(frame) for frame in frames)

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering\footnotesize",
        r"  \setlength{\tabcolsep}{5pt}",
        r"  \renewcommand{\arraystretch}{0.92}",
        r"  \caption{Mean tokens per call by prompt design, from the two pilots' logged API usage.",
        r"  \emph{In} is prompt tokens, \emph{Out} completion tokens including reasoning.}",
        r"  \label{tab:pilot-token-cost}",
        r"  \begin{tabular}{lrr@{\hskip 2em}lrr}",
        r"    \toprule",
        _tex_row([r"\multicolumn{3}{c}{Adoption pilot}", "", "", r"\multicolumn{3}{c}{Transmission pilot}", "", ""])
        .replace(" &  &  & ", " & ", 1).replace(" &  & ", "", 1),
        r"    \cmidrule(r){1-3}\cmidrule(l){4-6}",
        _tex_row(["Design", "In", "Out", "Design", "In", "Out"]),
        r"    \midrule",
    ]
    for i in range(height):
        cells = []
        for frame in frames:
            if i < len(frame):
                row = frame.iloc[i]
                cells += [row["design"], f"{row['in_per_call']:,.0f}", f"{row['out_per_call']:,.0f}"]
            else:
                cells += ["", "", ""]
        lines.append(_tex_row(cells))
    lines.append(r"    \midrule")
    overall = []
    for frame in frames:
        calls = frame["calls"].sum()
        overall += [
            f"All {len(frame)}",
            f"{frame['input_tokens'].sum() / calls:,.0f}",
            f"{frame['output_tokens'].sum() / calls:,.0f}",
        ]
    lines.append(_tex_row(overall))
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def to_latex_axes(tables: dict[str, pd.DataFrame]) -> str:
    """One row per axis: what each level costs per call, averaged over the rest."""
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering\small",
        r"  \caption{Mean tokens per call at each level of each axis, averaged over the other axes.",
        r"  The instruction axis is the only one that moves output tokens; the profile axes move input.}",
        r"  \label{tab:pilot-token-axes}",
        r"  \begin{tabular}{lrrrrrr}",
        r"    \toprule",
        _tex_row(["", r"\multicolumn{2}{c}{Level 0}", "", r"\multicolumn{2}{c}{Level 1}", "",
                  r"\multicolumn{2}{c}{Level 2}"]).replace(" &  & ", " & "),
        r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
        _tex_row(["Axis", "In", "Out", "In", "Out", "In", "Out"]),
        r"    \midrule",
    ]
    for pilot, table in tables.items():
        lines.append(_tex_row([rf"\multicolumn{{7}}{{l}}{{\emph{{{pilot.capitalize()} pilot}}}}", "", "", "", "", "", ""])
                     .replace(" & ", "", 6))
        means = axis_means(table, pilot)
        for axis in AXES[pilot]:
            cells = [AXIS_NAMES[axis]]
            for level in (0, 1, 2):
                hit = means[(means["axis"] == axis) & (means["level"] == level)]
                if len(hit):
                    cells += [f"{hit['in_per_call'].iloc[0]:,.0f}", f"{hit['out_per_call'].iloc[0]:,.0f}"]
                else:
                    cells += ["--", "--"]
            lines.append(_tex_row(cells))
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pilot", default="all", choices=("all", *PILOTS))
    p.add_argument("--model", default=None, help="filename prefix, e.g. gpt_5_4_nano (default: every model present)")
    p.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens")
    p.add_argument("--price-out", type=float, default=None, help="USD per 1M output tokens")
    p.add_argument("--csv", type=Path, default=None, help="also write the joined table here")
    p.add_argument("--latex", type=Path, nargs="?", const=Path("report/token_costs.tex"), default=None,
                   help="emit LaTeX (booktabs) instead of markdown; optionally to a file")
    a = p.parse_args(argv)

    names = tuple(PILOTS) if a.pilot == "all" else (a.pilot,)
    tables = {name: priced(load_usage(name, model=a.model), a.price_in, a.price_out) for name in names}

    if a.latex is not None:
        tex = to_latex_designs(tables) + "\n\n" + to_latex_axes(tables)
        a.latex.parent.mkdir(parents=True, exist_ok=True)
        a.latex.write_text(tex + "\n")
        print(tex)
        print(f"\n% wrote {a.latex}  -- needs \\usepackage{{booktabs}}")
    else:
        for name, table in tables.items():
            print(f"\n## {name} pilot\n")
            print(to_markdown(table.drop(columns=["pilot", "model"])))
    joined = pd.concat(tables.values(), ignore_index=True)
    if a.csv is not None:
        a.csv.parent.mkdir(parents=True, exist_ok=True)
        joined.to_csv(a.csv, index=False)
        print(f"\nwrote {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
