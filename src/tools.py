"""Small data-wrangling utilities for the diffusion project.

Three jobs. The first two have a CLI; the third is library-only.

1. Turn Stata `.dta` files from the replication bundle into plain CSV, so they
   can be read without pandas/Stata.

       python src/tools.py dta2csv "diffusion-science-data/datav4.0/Stata Replication/data/panel.dta"
       -> .../Stata Replication/data/panel.csv

2. Build the household feature table described in `docs/household_design.md`
   -- one row per household, columns per that design's §5 field layout -- and
   write one CSV per village.

       python src/tools.py hh-features --villages 73 67
       -> output/hh_features_73.csv, output/hh_features_67.csv

3. Wire that table and the real adjacency into `agent.HH_Agent` objects the
   simulation can step -- `build_agents()`, plus the `seeds()` and
   `missing_contexts()` pre-flight helpers. The agents themselves know nothing
   about this module; construction lives here so `agent.py` stays free of
   pandas and of any route to the feature table.

       agents = tools.build_agents(73)
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import agent as ag
    from . import data_loader as dl
    from . import state as stt
except ImportError:  # running as a script, not a package
    import agent as ag
    import data_loader as dl
    import state as stt


def dta_to_csv(
    src: Path | str,
    dest: Path | str | None = None,
    *,
    convert_categoricals: bool = True,
    overwrite: bool = True,
) -> Path:
    """Convert one Stata .dta file to CSV.

    Parameters
    ----------
    src
        Path to the .dta file.
    dest
        Output path. Defaults to `src` with a .csv suffix, i.e. the same
        filename in the same directory. If `dest` is an existing directory,
        the file is written inside it under the source's name.
    convert_categoricals
        Stata stores labelled variables as integer codes plus a value label
        map. True (the default) writes the labels, e.g. "Hindu" rather than 1,
        which is what you want for a human-readable CSV. Set False to keep the
        raw codes.
    overwrite
        If False, refuse to clobber an existing output file.

    Returns
    -------
    Path to the file written.
    """
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"no such file: {src}")
    if src.suffix.lower() != ".dta":
        raise ValueError(f"expected a .dta file, got {src.name}")

    dest = src.with_suffix(".csv") if dest is None else Path(dest)
    if dest.is_dir():
        dest = dest / (src.stem + ".csv")
    if dest.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {dest} (overwrite=False)")

    df = pd.read_stata(src, convert_categoricals=convert_categoricals)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest


# --------------------------------------------------------------------------
# Household feature table (docs/household_design.md §5)
# --------------------------------------------------------------------------
#
# One row per household, the reduced feature set. Columns follow the design
# doc's §5 field layout, minus the `neighbours` list and the mutable `state`
# block -- both belong to the simulation, not a static table -- plus
# `_adopted`, kept here (leading underscore) purely for later evaluation.
# This is raw data, not a persona: agent.py's to_persona() must render from
# its own explicit allowlist, never straight off this CSV.
#
# Everything unknown is empty in the CSV (pd.NA), never 0 and never "no" --
# see the design's R1. That is why every integer column here is a nullable
# Int64 and the two survey flags are a nullable boolean: 0/False has to keep
# meaning "measured, and the answer was none/no".

# Ordinal. Verified monotone bundle-wide in mean rooms (1.62 / 1.95 / 2.66),
# mean beds and share owning a latrine, so the integers are a real gradient
# and not a convenience coding -- docs/household_design.md §4.1. Note it is a
# connection *type*: government schemes target poorer households.
_ELECTRICITY_CODE = {"No": 0, "Yes, Government": 1, "Yes, Private": 2}

# Binary, deliberately. The three latrine labels do NOT form a quality ladder:
# "Common" households are marginally *poorer* than "None" on both wealth
# proxies (2.02 vs 2.09 rooms, 0.57 vs 0.61 beds) and there are 84 of them in
# 14,904 -- docs/household_design.md §4.2. So the only contrast the data
# supports is owning a latrine or not.
_OWN_LATRINE_CODE = {"Owned": 1, "Common": 0, "None": 0}

# Whose answer counts as the household's, for shgparticipate / savings. The
# individual file covers ~2.5 adults per household (recovered size ~5), so
# there is no "all members" option to be had -- only a choice of which
# interviewed adults to read. See docs/household_design.md §4.3 for why
# "couple" is the default and what the alternatives cost.
#
# The `has_shg` / `has_savings` column names stay scope-neutral so the header
# does not change under the flag -- which means the scope is NOT recoverable
# from the CSV and has to be recorded alongside it. The three scopes differ by
# a factor of ten on SHG membership, so this is not a detail to leave implicit.
_SHG_SCOPES: dict[str, tuple[str, ...] | None] = {
    "couple": ("Head of Household", "Spouse of Head of Household"),
    "head": ("Head of Household",),
    "any": None,  # every interviewed member, whatever their relation to the head
}


def _reported_yes(answers: pd.Series) -> object:
    """Tri-state OR over one household's answers to a yes/no survey item.

    True if anyone said Yes, False if nobody did but someone said No, and
    pd.NA when the only answers are "Do not know" / "Refuse to say" (50 such
    records bundle-wide). Collapsing those to False would invent a measurement
    the survey did not make.
    """
    vals = set(answers.astype(str))
    if "Yes" in vals:
        return True
    if "No" in vals:
        return False
    return pd.NA


def _survey_flags(individuals: pd.DataFrame, scope: str) -> tuple[pd.Series, pd.Series]:
    """Per-hhid SHG membership and savings, tri-state, over `scope`'s members."""
    if scope not in _SHG_SCOPES:
        raise ValueError(f"unknown shg scope {scope!r}; expected one of {sorted(_SHG_SCOPES)}")
    statuses = _SHG_SCOPES[scope]
    rows = individuals if statuses is None else individuals[individuals.resp_status.isin(statuses)]
    grp = rows.groupby("hhid", observed=True)
    return grp["shgparticipate"].apply(_reported_yes), grp["savings"].apply(_reported_yes)


def _subcaste(individuals: pd.DataFrame) -> pd.Series:
    """Per-hhid subcaste: the head's answer, else the first member interviewed.

    An identity attribute rather than a capability, so it is read off one
    person rather than OR'd -- and it is near-constant within a household
    anyway (865 of 6,901 have more than one distinct string, and some of that
    is spelling, not marriage). The head alone would leave 9-10% of surveyed
    households empty for no reason; head-then-anyone covers 100% of them.

    Strings are returned exactly as stored: uppercase, 0% blank, but *not*
    normalised. See docs/household_design.md §4.3b -- the same group is often
    spelled several ways within one village, which inflates the level count.
    """
    heads = individuals[individuals.resp_status == "Head of Household"]
    by_head = heads.drop_duplicates("hhid", keep="first").set_index("hhid").subcaste
    by_any = individuals.drop_duplicates("hhid", keep="first").set_index("hhid").subcaste
    return by_head.astype("string").reindex(by_any.index).fillna(by_any.astype("string"))


def _occupation(individuals: pd.DataFrame) -> pd.Series:
    """Per-hhid occupation: the head's answer, else the first member interviewed.

    Same head-then-anyone rule as `_subcaste`, for the same reason -- 90-91% of
    surveyed households have a head row, and reading the first interviewed
    member for the rest closes the gap. Unlike `subcaste`, high cardinality is
    real here rather than a spelling artefact (971 distinct free-text answers
    among heads alone, bundle-wide) -- see docs/household_design.md §4.3c. Kept
    raw and unbucketed anyway: any taxonomy would need keyword heuristics the
    design doc already rejected once (§4.5), and the raw string is the richer
    input for narrative persona generation.

    A blank answer usually means "does not work", not a missing measurement:
    it agrees with `workflag == No` for 1,021 of 1,028 blank heads bundle-wide.
    So it is written out as the literal string "no work" rather than left
    blank -- a blank cell is ambiguous between "does not work" and "this
    field was never answered", and only one of those is actually the case.
    The remaining handful (blank occupation but `workflag == Yes`, or
    `workflag` itself unrecorded) is a real data gap, not a work status, and
    stays pd.NA rather than being mislabelled "no work".
    """
    cols = ["occupation", "workflag"]
    heads = individuals[individuals.resp_status == "Head of Household"]
    by_head = heads.drop_duplicates("hhid", keep="first").set_index("hhid")[cols]
    by_any = individuals.drop_duplicates("hhid", keep="first").set_index("hhid")[cols]
    chosen = by_head.reindex(by_any.index).fillna(by_any)

    occ = chosen.occupation.astype("string").str.strip().replace("", pd.NA)
    workflag = chosen.workflag.astype("string").str.strip()
    return occ.mask(occ.isna() & workflag.eq("No"), "no work")


def _covariates(root: Path, village: int, room_no: pd.Series) -> tuple[pd.DataFrame, bool]:
    """Per-capita columns and household size from hhcovariates<V>.csv.

    That file is BCDJ's own six-column covariate matrix, tab separated, Stata
    "." for missing: rooms, beds, electricity code, latrine code, rooms per
    capita, beds per capita. Columns 1-4 reproduce the household file exactly
    in the candidate villages, which is what licenses using columns 5-6 --
    they are the only route to household size, and the only one that works for
    the non-surveyed half (docs/household_design.md §4.4).

    size = rooms / (rooms per capita) = beds / (beds per capita), the two
    routes agreeing in 100% of households where both are defined. The beds
    route alone covers only 25-54% because `beds == 0` for half the bundle,
    which makes beds per capita a true 0 rather than a divisor.

    Returns (frame[rooms_per_capita, beds_per_capita, capita], ok). ok is
    False -- every column NA -- when the file is missing, misaligned in
    length, or its rooms column doesn't reproduce room_no for at least half
    the households (this is how village 48's known misalignment is caught
    generically, rather than hard-coded as a one-off exclusion).
    """
    blank = pd.DataFrame(
        {"rooms_per_capita": np.nan, "beds_per_capita": np.nan, "capita": np.nan},
        index=room_no.index,
    )
    path = Path(root) / "Matlab Replication" / "India Networks" / f"hhcovariates{village}.csv"
    if not path.exists() or len(room_no) == 0:
        return blank, False

    cov = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["rooms", "beds", "elec", "latr", "rooms_pc", "beds_pc"],
        na_values=".",
    )
    if len(cov) != len(room_no):
        return blank, False
    cov = cov.replace([np.inf, -np.inf], np.nan)
    cov.index = room_no.index

    both = cov.rooms.notna().to_numpy() & room_no.notna().to_numpy()
    if not both.any():
        return blank, False
    match_rate = float(np.isclose(cov.rooms.to_numpy(dtype=float)[both], room_no.to_numpy(dtype=float)[both]).mean())
    if match_rate < 0.5:
        return blank, False

    from_rooms = (cov.rooms / cov.rooms_pc).replace([np.inf, -np.inf], np.nan)
    from_beds = (cov.beds / cov.beds_pc).replace([np.inf, -np.inf], np.nan)
    capita = from_rooms.where(from_rooms.notna(), from_beds).round()

    return pd.DataFrame(
        {"rooms_per_capita": cov.rooms_pc, "beds_per_capita": cov.beds_pc, "capita": capita},
        index=room_no.index,
    ), True


def _code(labels: pd.Series, mapping: dict[str, int], what: str, village: int) -> pd.Series:
    """Map a labelled Stata column to integer codes, loudly.

    Anything non-null that the mapping doesn't know is a DataError rather than
    a silent NA: a new spelling would otherwise flood the column with
    "unknown" and look like missing data.
    """
    text = labels.astype(str)
    known = text.isin(mapping) | labels.isna()
    if not known.all():
        bad = sorted(set(text[~known]))
        raise dl.DataError(f"v{village}: unmapped {what} label(s) {bad}")
    return text.map(mapping).astype("Int64").where(labels.notna())


def build_household_features(
    village: int,
    root: Path | str | None = None,
    shg_scope: str = "couple",
) -> pd.DataFrame:
    """One row per household in `village`, columns per docs/household_design.md §5.

    Uses data_loader.load_village() for alignment (adjacency <-> household
    file <-> outcome), so every guarantee load_village checks -- shape,
    ordering, the two independent seed-set copies agreeing -- holds here too.
    Row i is adjacency row i, i.e. adjmatrix_key i + 1.
    """
    root = Path(root) if root is not None else dl.DEFAULT_ROOT
    v = dl.load_village(village, root=root)  # check=True (default): raises DataError on real misalignment
    hh = v.households.reset_index(drop=True)

    cov, _ = _covariates(root, village, hh.room_no)
    shg, savings = _survey_flags(v.individuals, shg_scope)
    subcaste = _subcaste(v.individuals)
    occupation = _occupation(v.individuals)
    surveyed = hh.hhSurveyed.astype(bool)

    out = pd.DataFrame(
        {
            # -- identity: joins and row alignment, never a persona ---------
            "village": village,
            "row": hh.adjmatrix_key.to_numpy(dtype=int),
            "hhid": hh.hhid.to_numpy(),
            "hh_num": hh.HHnum_in_village.to_numpy(dtype=int),
            # -- mechanics and evaluation -----------------------------------
            # degree is kept for slicing results, NOT for the prompt: telling
            # an agent it has 17 friends manufactures the network effect we
            # are trying to measure (design §5).
            "degree": pd.array(v.degree.astype(int), dtype="Int64"),
            "has_leader": pd.array(v.leader.astype(int), dtype="Int64"),
            "in_giant": pd.array(v.in_giant.astype(int), dtype="Int64")
            if v.in_giant is not None
            else pd.array([pd.NA] * v.n, dtype="Int64"),
            # -- base block: present for every household ---------------------
            "religion": hh.hohreligion.astype(str).str.lower().to_numpy(),
            "rooms": hh.room_no.astype("Int64").to_numpy(),
            "beds": hh.bed_no.astype("Int64").to_numpy(),
            "capita": cov.capita.astype("Int64").to_numpy(),
            "rooms_per_capita": cov.rooms_per_capita.to_numpy(dtype=float),
            "beds_per_capita": cov.beds_per_capita.to_numpy(dtype=float),
            "electricity": _code(hh.electricity, _ELECTRICITY_CODE, "electricity", village).to_numpy(),
            "own_latrine": _code(hh.latrine, _OWN_LATRINE_CODE, "latrine", village).to_numpy(),
            # -- survey block: the surveyed ~46-54% only ---------------------
            # A research artefact the household cannot know about itself: on
            # the object for the design's R2 ablation, never in the prompt.
            "surveyed": surveyed.to_numpy(),
            "subcaste": pd.array(hh.hhid.map(subcaste).where(surveyed), dtype="string"),
            "occupation_head": pd.array(hh.hhid.map(occupation).where(surveyed), dtype="string"),
            "has_shg": pd.array(hh.hhid.map(shg).where(surveyed), dtype="boolean"),
            "has_savings": pd.array(hh.hhid.map(savings).where(surveyed), dtype="boolean"),
            # -- privileged: evaluation only ---------------------------------
            # `_adopted` is the whole of the household-level ground truth.
            # There is no per-household adoption *time* anywhere in the public
            # bundle (design §2), so no column pretends there is: timing is
            # validated at village level against panel.dta.
            "_adopted": pd.array(v.mf.astype(int), dtype="Int64"),
        }
    )
    # Pinned explicitly rather than inferred: a village where every household
    # happens to have an electricity reading would otherwise get int64 here
    # and object/float in the three villages that don't, and the CSVs would
    # stop being comparable.
    return out.astype(
        {
            "village": "int64",
            "row": "int64",
            "hh_num": "int64",
            "degree": "Int64",
            "has_leader": "Int64",
            "in_giant": "Int64",
            "religion": "string",
            "rooms": "Int64",
            "beds": "Int64",
            "capita": "Int64",
            "rooms_per_capita": "float64",
            "beds_per_capita": "float64",
            "electricity": "Int64",
            "own_latrine": "Int64",
            "surveyed": "bool",
            "subcaste": "string",
            "occupation_head": "string",
            "has_shg": "boolean",
            "has_savings": "boolean",
            "_adopted": "Int64",
        }
    )


# --------------------------------------------------------------------------
# Agent construction (docs/household_design.md §5, network block)
# --------------------------------------------------------------------------


def build_agents(
    village: int,
    features_dir: Path | str = Path("output"),
    context_dir: Path | str = ag.DEFAULT_CONTEXT_DIR,
    root: Path | str | None = None,
    state_version: str = "default",
) -> dict[int, ag.HH_Agent]:
    """Every household in `village` as an `HH_Agent`, keyed by hhid, wired to the real network.

    Reads exactly two things and nothing else: `row` and `hhid` from
    `output/hh_features_<village>.csv`, and the adjacency from
    `data_loader.load_village()` -- which is also what guarantees row *i* of the
    CSV is row *i* of the matrix. Every other column, the persona features and
    `_adopted` alike, is left in the CSV on purpose: an agent that holds no
    reference to the feature table cannot leak the target through one
    (docs/household_design.md §1). This function is the one place that property
    could quietly be lost, which is why the column selection is an explicit
    `usecols` allowlist rather than a drop-list.

    Isolates are included. They are genuine non-adopters and a free correctness
    check (§7.2): a sane model must never have one adopt, since nobody can ever
    speak to it.
    """
    path = Path(features_dir) / f"hh_features_{village}.csv"
    ids = pd.read_csv(path, usecols=["row", "hhid"]).sort_values("row")
    hh_ids = ids.hhid.to_numpy(dtype=int)

    v = dl.load_village(village, root=root if root is not None else dl.DEFAULT_ROOT)
    if len(hh_ids) != v.n:
        raise stt.AgentError(f"v{village}: {len(hh_ids)} rows in {path.name} but {v.n} in the adjacency")

    return {
        int(hh): ag.HH_Agent(
            hh_id=int(hh),
            village=village,
            row=i + 1,  # adjmatrix_key is 1-based; adjacency row i is key i + 1
            neighbours=tuple(int(hh_ids[j]) for j in v.neighbours(i)),
            context_dir=Path(context_dir),
            state=stt.make_state(state_version),
        )
        for i, hh in enumerate(hh_ids)
    }


def missing_contexts(agents: Iterable[ag.HH_Agent]) -> list[int]:
    """hh_ids with no context file. Call before spending money on a run.

    Discovering a missing persona at t=4, several hundred API calls in, is an
    expensive way to learn it.
    """
    return [a.hh_id for a in agents if not a.has_context]


def seeds(village: int, features_dir: Path | str = Path("output")) -> list[int]:
    """hhids of the injection points -- `has_leader == 1` in the feature table.

    Not privileged (§4.7): the seed set is known ex ante and a household
    legitimately knows the MFI spoke to it. BCDJ's caveat travels with it --
    `leader` marks who the MFI *could* have informed, not who verifiably was.
    """
    path = Path(features_dir) / f"hh_features_{village}.csv"
    df = pd.read_csv(path, usecols=["hhid", "has_leader"])
    return df.loc[df.has_leader == 1, "hhid"].astype(int).tolist()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("dta2csv", help="convert Stata .dta file(s) to CSV")
    c.add_argument("input", type=Path, nargs="+", help="one or more .dta files")
    c.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output file, or a directory when converting several inputs "
        "(default: same name and directory as the input, with .csv)",
    )
    c.add_argument(
        "--raw-codes",
        action="store_true",
        help="keep Stata's integer codes instead of expanding value labels",
    )
    c.add_argument("--no-clobber", action="store_true", help="fail instead of overwriting an existing CSV")

    ca = sub.add_parser(
        "hh-features",
        help="build the household feature table (docs/household_design.md), one CSV per village",
    )
    ca.add_argument(
        "--villages",
        type=int,
        nargs="+",
        default=None,
        help="village numbers to build (default: every village with a household-level MF outcome)",
    )
    ca.add_argument("--root", type=Path, default=None, help="path to datav4.0 (default: data_loader.DEFAULT_ROOT)")
    ca.add_argument(
        "--shg-scope",
        choices=sorted(_SHG_SCOPES),
        default="couple",
        help="whose survey answer becomes has_shg / has_savings: the head and their "
        "spouse (default), the head alone, or every interviewed member",
    )
    ca.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="directory for hh_features_<village>.csv (default: output)",
    )

    a = p.parse_args(argv)

    if a.command == "dta2csv":
        if a.output is not None and len(a.input) > 1 and not a.output.is_dir():
            p.error("--output must be an existing directory when converting more than one file")

        for src in a.input:
            try:
                dest = dta_to_csv(
                    src,
                    a.output,
                    convert_categoricals=not a.raw_codes,
                    overwrite=not a.no_clobber,
                )
            except (FileNotFoundError, FileExistsError, ValueError) as e:
                print(f"error: {e}", file=sys.stderr)
                return 1
            print(f"{src} -> {dest}")
        return 0

    if a.command == "hh-features":
        root = a.root if a.root is not None else dl.DEFAULT_ROOT
        villages = a.villages if a.villages else dl.available_villages(root)

        written = 0
        for vil in villages:
            try:
                out = build_household_features(vil, root=root, shg_scope=a.shg_scope)
            except dl.DataError as e:
                print(f"warning: skipping village {vil}: {e}", file=sys.stderr)
                continue

            if out.capita.isna().all():
                print(
                    f"warning: village {vil}: capita and the per-capita columns are unrecoverable "
                    "(hhcovariates missing or misaligned)",
                    file=sys.stderr,
                )

            a.output_dir.mkdir(parents=True, exist_ok=True)
            dest = a.output_dir / f"hh_features_{vil}.csv"
            out.to_csv(dest, index=False)
            written += 1
            print(f"village {vil}: {len(out)} households -> {dest}")

        if not written:
            print("error: no villages produced any data", file=sys.stderr)
            return 1
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
