"""The persona layer: turn a household's feature row into the profile its agent carries.

Two modes, one grounding, one artefact. Both are functions of `render_traits()`
and nothing else, and both write **one JSON per village** --
`output/profiles/profiles_<village>.json`, keyed by hhid.

    python -m src.profiler --mode facts --villages 73 67      # free, instant, deterministic
    python -m src.profiler --mode story --villages 73         # dry-run preview, no API calls
    python -m src.profiler --mode story --villages 73 --live  # real run, costs money
    python -m src.profiler --mode story --villages 6 --live --leader-mode explicit
                                                              # -> profiles_6_explicit.json

That JSON is the single source of everything profile-related. One record per
household, and in story mode it carries all five halves of the story at once:

    "73001": {
      "traits":            {...},   # the twelve disclosed fields, labelled
      "static_profile":    "...",   # mode 1's profile, rule-based, no LLM
      "prompt":            {"model", "instructions", "input"},
      "narrative_profile": "...",   # mode 2's profile, what the LLM returned
      "usage":             {...}    # what this household cost, in tokens
    }

`facts` mode writes the first two keys and stops; `story` mode writes all five,
so a story-mode file is a strict superset and both arms can be read out of one
place. Keeping `traits` and `static_profile` beside the narrative is the point:
the two arms of the experiment are then provably built from the same twelve
facts, rather than from two files that have to be trusted to agree.

`prompt` is the *whole* request as the model receives it -- system instructions
as well as the fact listing -- so what the model was told is recoverable from the
artefact rather than from whichever version of this source produced it. `usage`
is per household rather than per village because a profile is the thing we pay
for once per agent, so its token size is what scales the simulation;
`total_usage()` aggregates when a total is wanted.

**No context `.txt` files.** An earlier version of this module also wrote
`output/context/<mode>/context_<hhid>.txt`, one per household, because that is
what `agent.HH_Agent.context` reads. That is withdrawn: the JSON is the artefact
going forward, and a second on-disk copy of the same text is one that can drift
from it. `game_master.from_village()` now reads this JSON through
`elicit.load_personas()` and pushes the chosen arm into each agent with
`set_context()`, so no context file is involved anywhere in a run; `agent.py`
keeps its file-reading fallback for tests and dry runs.

Two decisions this layer makes
------------------------------

**1. The neighbour view is rule-based in both modes.** The mode switch applies
to the agent's *own* profile and nothing else. `render_neighbour_profile()`
(moved here from `game_master._render_profile`, which called itself a
placeholder for this module) is the only thing one household is ever shown about
another. Three reasons, in order of weight: cost -- village 24 is ~8,400
`inform` calls, each carrying the recipient's profile, and the profile is the
*non*-cacheable tail of that prompt, so a 110-word narrative there is paid for
8,400 times while the self-profile is paid for once; epistemics -- you know your
neighbour's house, work and community, not their interior life; and experimental
hygiene -- exactly one thing should differ between the two arms.

**2. The surveyed/non-surveyed length gap is measured, not constrained.**
`facts` mode is fixed-length by construction -- twelve lines, always, `not known`
rather than dropped. `story` mode is not: a non-surveyed household is short four
facts, and a narrative written from fewer facts comes out shorter. That is a
real property of having less to say about a household, and it is left alone.
`balance_report()` reports the gap with a Welch *t*-test on every live run so the
number is on the record and can be quoted with the results -- it renders no
verdict and gates nothing. Measured on 40 households of village 73 under an
earlier prompt, the gap ran about 6 words on a ~110-word budget.

The non-negotiable, carried over from `docs/household_design.md` §1/§4.7
-------------------------------------------------------------------------
The privileged outcome `_adopted` must be *structurally* unable to reach a
prompt. `render_traits()` builds its output from an explicit allowlist of column
names read off the row -- the twelve of §5, plus `surveyed` as the gate on the
four survey-derived ones and `has_leader` in the one narrow case below. It never
touches `_adopted`, `degree` (never stated numerically -- §5), `in_giant`, or any
raw `hhid`/`row`/`hh_num`/`village` identifier (§4.6). Because both modes are
built from `render_traits()` output and the LLM is handed a `dict[str, str]`
rather than a row, that one control covers the story arm too: there is no route
by which the model could see a column the listing does not contain.

`has_leader` is not privileged (§4.7 -- it is the seed assignment, known ex
ante), and how much of it reaches the listing is now a parameter,
`--leader-mode`, because how loudly an agent is told it is a leader is itself
a thing to vary rather than a thing to settle in the source:

  `implicit` (default, the behaviour every existing artefact was written under)
      `has_leader` is read *only* as a fallback for the occupation line: a
      leader household whose `occupation_head` is unknown is described as
      working in a role that entails leadership, which is what BSS's leader
      designation actually means (teachers, shopkeepers, self-help-group
      leaders -- CODEBOOK.md §3.1). A leader household that already has an
      occupation keeps it, and nothing else is said, so leader status is never
      a standalone flag in a prompt.

  `explicit`
      Leadership is stated outright, as a thirteenth trait, and only for the
      households that have it -- a non-leader household is told nothing, not
      even "not known", so the absence of the line is the absence of the
      status. The occupation fallback is dropped here: the status is on the
      record in its own right, so inferring it into the occupation line as
      well would state the same fact twice and blur which line carried it.

The two modes write to *different files* -- `profiles_<village>.json` and
`profiles_<village>_explicit.json` -- because they are two different treatments
of the same village and one file holding both is one file that cannot say which
of them an agent was run under.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import openai
import pandas as pd

try:
    from . import llm
except ImportError:  # running as a script, not a package
    import llm  # type: ignore[no-redef]

KEYS_PATH = llm.KEYS_PATH
DEFAULT_FEATURES_DIR = Path("output/features")
DEFAULT_OUTPUT_DIR = Path("output/profiles")
DEFAULT_MODEL = llm.DEFAULT_MODEL

MODES = ("facts", "story")

# How `has_leader` reaches the fact listing -- the module docstring's two
# treatments. `implicit` first, and the default everywhere, so that every
# existing profiles_<village>.json is reproducible from the current code.
LEADER_MODES = ("implicit", "explicit")
DEFAULT_LEADER_MODE = "implicit"

# One file per treatment, keyed off the mode so a caller cannot pick a name that
# disagrees with what is inside it.
LEADER_SUFFIX = {"implicit": "", "explicit": "_explicit"}

UNKNOWN = "not known"

_RELIGION_LABELS = {"hinduism": "Hindu", "islam": "Muslim", "christianity": "Christian"}

# CODEBOOK.md §3.1: `electricity` is a connection *type*, and tools.py has
# already folded the survey's 1/2/3 into the ordinal 0/1/2 written to the CSV.
_ELECTRICITY_LABELS = {
    0: "none",
    1: "yes, a government connection",
    2: "yes, a private connection",
}

# CODEBOOK.md §3.1 / design §4.2: the survey's "Common" (84 households in
# 14,904) is folded in with "None", so 0 covers both.
_LATRINE_LABELS = {
    1: "yes, the household owns one",
    0: "no -- it uses a shared latrine or none at all",
}


# --------------------------------------------------------------------------
# The self view -- what a household knows about itself
# --------------------------------------------------------------------------


def _number(value: object, *, decimals: int = 0) -> str:
    """A bare figure, or "not known". Everything unknown is pd.NA in the CSV, never 0."""
    if pd.isna(value):
        return UNKNOWN
    return f"{float(value):.{decimals}f}"


def _yes_no(surveyed: bool, value: object) -> str:
    """R1: survey-derived facts are always rendered, "not known" rather than dropped."""
    if not surveyed or pd.isna(value):
        return UNKNOWN
    return "yes" if bool(value) else "no"


def _text(surveyed: bool, value: object) -> str:
    if not surveyed or pd.isna(value):
        return UNKNOWN
    return str(value).strip().lower() or UNKNOWN


def _insert_after(traits: dict[str, str], after: str, key: str, value: str) -> dict[str, str]:
    """`traits` with one entry spliced in behind `after`. Order is the listing order."""
    out: dict[str, str] = {}
    for k, v in traits.items():
        out[k] = v
        if k == after:
            out[key] = value
    return out


def _is_leader(row: pd.Series) -> bool:
    return bool(pd.notna(row.get("has_leader")) and int(row["has_leader"]) == 1)


def _occupation_line(row: pd.Series, surveyed: bool, leader_mode: str) -> str:
    """The head's occupation, with the leader-derived fallback in `implicit` mode only.

    Under `implicit`, `has_leader` is consulted when the occupation itself is
    unknown: BSS's leaders are teachers, shopkeepers and self-help-group leaders
    (CODEBOOK.md §3.1), so "a role that entails leadership" is the honest floor
    on what the seed assignment tells us about the head's work. A leader
    household with a recorded occupation keeps it and nothing further is said.

    Under `explicit` there is no fallback: `LEADER_TRAIT` states the status
    outright, so an unknown occupation stays unknown and the two facts do not
    overlap.
    """
    occupation = _text(surveyed, row.get("occupation_head"))
    if occupation != UNKNOWN:
        return occupation
    if leader_mode == "implicit" and _is_leader(row):
        return "works in a role that entails leadership in the village"
    return UNKNOWN


# `explicit` mode's thirteenth line, verbatim, and only for leader households.
LEADER_TRAIT = 'This household holds the status of a "leader" in their network originating from the occupation.'


def render_traits(row: pd.Series, *, leader_mode: str = DEFAULT_LEADER_MODE) -> dict[str, str]:
    """The twelve disclosed fields of `docs/household_design.md` §5, as a plain fact listing.

    One `field: value` line per field, always all twelve, `not known` where the
    value is missing -- no interpretation, no ranking, no derived features. The
    labels spell out what the codebook means by each column so a reader does not
    have to guess at `has_shg` or `own_latrine`.

    **This is the single grounding for both modes.** `facts` renders it directly;
    `story` hands it to a model as the entire content of the user turn. Nothing
    downstream of here ever sees the row, which is what makes the leakage control
    below structural rather than conventional.

    Reads only: religion, rooms, beds, capita, rooms_per_capita,
    beds_per_capita, electricity, own_latrine, subcaste, occupation_head,
    has_shg, has_savings -- plus `surveyed` (gates the last four) and
    `has_leader`. `_adopted`, `degree`, `in_giant` and every identifier column
    are simply not in this function's vocabulary.

    `leader_mode` is the only thing that varies the schema. Under `implicit`
    the twelve are the whole listing and `has_leader` only ever colours the
    occupation line. Under `explicit` a leader household gets a thirteenth
    entry, `has_leader`, immediately after the occupation it derives from, and
    a non-leader household gets nothing at all -- the one field in this
    function that is dropped rather than rendered `not known`, because "not
    known" would assert that leadership is a thing the household might have and
    the record merely fails to say.
    """
    if leader_mode not in LEADER_MODES:
        raise ValueError(f"unknown leader_mode {leader_mode!r}; expected one of {LEADER_MODES}")

    surveyed = bool(row.get("surveyed", False))

    religion = row.get("religion")
    religion_label = (
        _RELIGION_LABELS.get(str(religion).lower(), str(religion).title()) if pd.notna(religion) else UNKNOWN
    )

    electricity = row.get("electricity")
    latrine = row.get("own_latrine")

    # Free text, 434 distinct spellings (CODEBOOK.md §3.2) -- kept verbatim
    # apart from case, because normalising it would be a judgement call.
    subcaste = _text(surveyed, row.get("subcaste"))
    if subcaste != UNKNOWN:
        subcaste = subcaste.title()

    traits = {
        "religion": f"Religion: {religion_label}",
        "rooms": f"Rooms in the house: {_number(row.get('rooms'))}",
        "beds": f"Beds or cots in the house: {_number(row.get('beds'))}",
        "capita": f"People living in the household: {_number(row.get('capita'))}",
        "rooms_per_capita": f"Rooms per person: {_number(row.get('rooms_per_capita'), decimals=2)}",
        "beds_per_capita": f"Beds or cots per person: {_number(row.get('beds_per_capita'), decimals=2)}",
        "electricity": (
            "Electricity connection at home: "
            f"{_ELECTRICITY_LABELS[int(electricity)] if pd.notna(electricity) else UNKNOWN}"
        ),
        "own_latrine": (
            f"Latrine of its own: {_LATRINE_LABELS[int(latrine)] if pd.notna(latrine) else UNKNOWN}"
        ),
        "subcaste": f"Subcaste: {subcaste}",
        "occupation_head": (
            f"Occupation of the head of the household: {_occupation_line(row, surveyed, leader_mode)}"
        ),
        "has_shg": f"Member of a savings self-help group (SHG): {_yes_no(surveyed, row.get('has_shg'))}",
        "has_savings": f"Has a bank or savings account: {_yes_no(surveyed, row.get('has_savings'))}",
    }

    if leader_mode == "explicit" and _is_leader(row):
        # After the occupation, which is where the status comes from, and before
        # the two survey flags -- insertion order is listing order.
        traits = _insert_after(traits, "occupation_head", "has_leader", LEADER_TRAIT)

    return traits


FACTS_PREAMBLE = "This is a household in a village in rural Karnataka, India. This is what is true of it:"


def persona_facts(traits: dict[str, str]) -> str:
    """The `facts` arm's profile: the fact listing, third person frame, nothing else.

    Third person to match the narrative arm, which `SYSTEM_PROMPT` asks for in
    the third person ("describing it to someone else"). The two arms are the same
    experiment run two ways, so a difference in grammatical person between them
    would be an uncontrolled second variable sitting on top of the one being
    tested. The twelve trait labels were already person-neutral -- `Latrine of
    its own`, `Occupation of the head of the household` -- so this frame is the
    only line that carried a person at all.

    Deterministic, free, and fixed-length by construction -- twelve lines for
    every household in every village, which is R1 satisfied structurally rather
    than by measurement. That is the property the story arm is measured for and
    this one cannot lose.
    """
    lines = "\n".join(f"- {t}" for t in traits.values())
    return f"{FACTS_PREAMBLE}\n{lines}"


# --------------------------------------------------------------------------
# The neighbour view -- what one household may be told about another
# --------------------------------------------------------------------------
#
# Moved here from `game_master._render_profile`, which described itself as "a
# placeholder for the real persona renderer, which arrives with the LLM layer".
# This is that module, so the two renderings now share one vocabulary and one
# set of labels instead of drifting apart in separate files. `game_master`
# re-exports both names.
#
# The allowlist is narrower than the twelve of the self view, and legitimately
# so: it is what a villager would know about a neighbour, not what a household
# knows about itself. Excluded, and why each one:
#   _adopted            the target. The whole game.
#   surveyed, in_giant  research artefacts. No villager knows whether the
#                       enumerators reached their neighbour.
#   degree              §5, explicitly: stating it manufactures the network
#                       effect being measured. It is not observable anyway --
#                       nobody knows their neighbour's friend count.
#   has_leader          marks who the MFI approached. Leadership itself is
#                       visible in a village, but this column is the seed
#                       indicator, and handing it over tells an agent which
#                       neighbours were briefed. Kept out.
#   village, row,
#     hh_num, hhid      identity and ordering. §4.6: a numeric id invites the
#                       model to invent orderings that do not exist. `hhid` is
#                       read as the join key and never rendered.
#   *_per_capita        §5, "three columns, one fact": rooms + capita already
#                       say it, and rendering it twice pays for it twice.
NEIGHBOUR_FIELDS = (
    "hhid",
    "religion",
    "rooms",
    "beds",
    "capita",
    "electricity",
    "own_latrine",
    "subcaste",
    "occupation_head",
    "has_shg",
    "has_savings",
)

# Never render these, whatever a caller passes. Belt and braces behind `usecols`,
# and the thing `test_game_master.py` asserts against.
FORBIDDEN_FIELDS = ("_adopted", "surveyed", "degree", "has_leader", "in_giant")

_ELECTRICITY_WORDS = {
    0: "no electricity connection",
    1: "a government electricity connection",
    2: "a private electricity connection",
}


def _has(value: Any) -> bool:
    return value is not None and not pd.isna(value)


def render_neighbour_profile(row: pd.Series) -> str:
    """One household as a neighbour would describe it: third person, no numbers that rank.

    Third person because this text is only ever *about* someone else -- the
    module docstring's decision 1. It stays rule-based in both modes: it is
    attached to every `inform` call, which is per edge per round and is the
    non-cacheable tail of that prompt, so its cost scales with edges rather than
    with households. Renders from `NEIGHBOUR_FIELDS` and nothing else.
    """
    out: list[str] = []

    who = f"A {row.religion} household" if _has(row.religion) else "A household"
    if _has(row.capita):
        who += f" of {int(row.capita)}"
    out.append(who + ".")

    home: list[str] = []
    if _has(row.rooms):
        home.append(f"{int(row.rooms)} room{'s' if int(row.rooms) != 1 else ''}")
    if _has(row.electricity):
        home.append(_ELECTRICITY_WORDS[int(row.electricity)])
    if _has(row.own_latrine):
        home.append("their own latrine" if int(row.own_latrine) else "no latrine of their own")
    if home:
        out.append("They live in a house with " + ", ".join(home) + ".")
    # §5: `beds == 0` is half the bundle and reads as a broken record if left as
    # a bare number, so it gets words. It is a poverty marker, not a typo.
    if _has(row.beds):
        out.append("They have no cot or bedstead." if int(row.beds) == 0 else f"They own {int(row.beds)} bed(s).")

    if _has(row.subcaste):
        out.append(f"They are {str(row.subcaste).strip().lower()}.")
    if _has(row.occupation_head):
        occ = str(row.occupation_head).strip().lower()
        out.append("The head of the household does not work." if occ == "no work" else f"The head works as a {occ}.")

    if _has(row.has_shg):
        out.append(
            "Someone in the household takes part in a self-help group."
            if bool(row.has_shg)
            else "Nobody in the household takes part in a self-help group."
        )
    if _has(row.has_savings):
        out.append("The household has savings." if bool(row.has_savings) else "The household has no savings.")

    return " ".join(out)


def neighbour_profiles(village: int, features_dir: Path | str = DEFAULT_FEATURES_DIR) -> dict[int, str]:
    """hhid -> the text a neighbour may be shown, for every household in `village`.

    `usecols=NEIGHBOUR_FIELDS` is the control: the outcome column is not
    filtered out downstream, it is never read into memory.
    """
    path = Path(features_dir) / f"hh_features_{village}.csv"
    df = pd.read_csv(path, usecols=list(NEIGHBOUR_FIELDS))
    return {int(r.hhid): render_neighbour_profile(r) for r in df.itertuples(index=False)}


# --------------------------------------------------------------------------
# Mode 2: the story
# --------------------------------------------------------------------------

# Word budget. Stated as a narrow range rather than a loose one because of the
# module docstring's decision 4: a non-surveyed household has four fewer facts
# to work with, and left to its own devices a model writes proportionally less.
# Persona length would then track `surveyed`, which tracks degree (3.41x) and
# take-up (+2.4pp). `balance_report()` checks whether the instruction held.
STORY_WORDS = (125, 150)

# Write {STORY_WORDS[0]}-{STORY_WORDS[1]} words.

SYSTEM_PROMPT = f"""\

You will be given a listing of a household's facts.
Your task is to write an engaging, colourful story by incorporating those facts, turning them into interesting descriptive phrases.
Your story should talk about the household in third person, imagine you are talking about them to your friends or someone who is interested in knowing them.
You can add interesting details that the facts could imply.

"""


def build_prompt(traits: dict[str, str]) -> str:
    """The user turn: the fact listing and nothing else."""
    lines = "\n".join(f"- {t}" for t in traits.values())
    return f"Recorded facts about this household:\n{lines}"


def build_request(traits: dict[str, str], model: str) -> dict[str, str]:
    """Everything that goes to the model, in the Responses API's own field names.

    Written into the JSON verbatim so a reader of the output never has to
    reconstruct what the model was actually told.
    """
    return {"model": model, "instructions": SYSTEM_PROMPT, "input": build_prompt(traits)}


def prompt_fingerprint(request: dict[str, str]) -> str:
    """A short hash of the whole request -- model, instructions and facts.

    The guard against a silently mixed corpus. Generation is resumable, so
    editing `SYSTEM_PROMPT` and re-running without `--overwrite` would otherwise
    leave the already-written households on the old instructions and the rest on
    the new ones, in one file, with nothing in the artefact saying so. A
    household whose stored fingerprint no longer matches is regenerated and
    counted in the run's summary line.
    """
    blob = json.dumps(request, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# --------------------------------------------------------------------------
# OpenAI Responses API
# --------------------------------------------------------------------------

# The client, the retry policy and the token accounting now live in `llm.py`,
# because `elicit.py` needs the same four functions and a second copy of a retry
# loop is a second thing to keep in step. Re-exported under their old names so
# this module's public surface is unchanged.
_load_openai_client = llm.load_client
usage_dict = llm.usage_dict
total_usage = llm.total_usage
format_usage = llm.format_usage
_one_call = llm.one_call


def generate_narrative(
    client: openai.OpenAI,
    request: dict[str, str],
    *,
    temperature: float = 0.9,
    max_output_tokens: int = 1000,
    max_attempts: int = 3,
) -> tuple[str, dict[str, int]]:
    """The narrative profile for one household, and what it cost in tokens.

    `_one_call` does the work; this is the name the orchestration uses, kept
    separate so a caller reads "generate a narrative" rather than "make an API
    call". Retries transient errors only -- what the model chooses to write is
    not this layer's business to second-guess.
    """
    return _one_call(
        client,
        request,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        max_attempts=max_attempts,
    )


# --------------------------------------------------------------------------
# How narrative length varies across the surveyed split
# --------------------------------------------------------------------------


def _welch(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch's t and its degrees of freedom. Hand-rolled to avoid a scipy dependency
    for one number; the p-value is left to the reader, since t and df are what a
    write-up quotes anyway. NaN when either side is too small to have a variance."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan"), float("nan")
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    se2 = va / na + vb / nb
    if se2 == 0:
        return float("nan"), float("nan")
    t = (ma - mb) / math.sqrt(se2)
    df = se2**2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return t, df


def balance_report(profiles: dict[str, dict], features: pd.DataFrame) -> dict[str, Any]:
    """Narrative length by `surveyed`: the two means, the gap, and Welch's *t*.

    A descriptive measurement, not a gate. A surveyed household has four more
    facts to write from (subcaste, occupation, SHG, savings), so its narrative
    runs longer -- that is a real consequence of knowing more about it, and it is
    left in place rather than engineered away. What matters is that the size of
    the effect is on the record: `surveyed` tracks degree (3.41x) and take-up
    (+2.4pp) per `docs/household_design.md` §3, so anything that co-varies with
    it has to be quoted alongside the results rather than found afterwards.

    The `facts` arm is fixed-length by construction, so this only ever describes
    the story arm. Households with no narrative yet simply do not contribute.
    """
    surveyed_by_hhid = {int(r.hhid): bool(r.surveyed) for r in features.itertuples(index=False)}
    groups: dict[bool, list[float]] = {True: [], False: []}
    for hhid, rec in profiles.items():
        text = rec.get("narrative_profile")
        flag = surveyed_by_hhid.get(int(hhid))
        if text and flag is not None:
            groups[flag].append(float(len(text.split())))

    srv, non = groups[True], groups[False]
    mean_srv = sum(srv) / len(srv) if srv else float("nan")
    mean_non = sum(non) / len(non) if non else float("nan")
    t, df = _welch(srv, non)
    return {
        "n_surveyed": len(srv),
        "n_not_surveyed": len(non),
        "mean_words_surveyed": mean_srv,
        "mean_words_not_surveyed": mean_non,
        "gap_words": mean_srv - mean_non,
        "welch_t": t,
        "welch_df": df,
    }


def format_balance(report: dict[str, Any]) -> str:
    """One human line. Descriptive throughout -- it states the gap, never judges it."""
    if not report["n_surveyed"] or not report["n_not_surveyed"]:
        return (
            f"narrative length: {report['n_surveyed']} surveyed / {report['n_not_surveyed']} not surveyed "
            "-- one side empty, no comparison to make"
        )
    direction = "longer" if report["gap_words"] >= 0 else "shorter"
    line = (
        f"narrative length: surveyed {report['mean_words_surveyed']:.1f} words (n={report['n_surveyed']}) vs "
        f"not surveyed {report['mean_words_not_surveyed']:.1f} (n={report['n_not_surveyed']}) -- "
        f"surveyed {abs(report['gap_words']):.1f} words {direction}"
    )
    if not math.isnan(report["welch_t"]):
        line += f", t={report['welch_t']:.2f}, df={report['welch_df']:.0f}"
    return line


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def discover_villages(features_dir: Path | str) -> list[int]:
    """Village numbers with an existing hh_features_<village>.csv, sorted."""
    out = []
    for p in Path(features_dir).glob("hh_features_*.csv"):
        try:
            out.append(int(p.stem.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    return sorted(out)


def _load_features(village: int, features_dir: Path | str) -> pd.DataFrame:
    path = Path(features_dir) / f"hh_features_{village}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"no {path} -- build it first with `python -m src.tools hh-features --villages {village}`"
        )
    return pd.read_csv(path)


def _rows(village: int, features_dir: Path | str, limit: int | None) -> tuple[pd.DataFrame, list]:
    df = _load_features(village, features_dir)
    rows = list(df.itertuples(index=False))
    return df, rows[:limit] if limit is not None else rows


def _carry_forward(dest: Path, limit: int | None) -> dict[str, dict]:
    """Records already in the file that this run will not touch, so `--limit` never deletes.

    Both modes write the same `profiles_<village>.json`, so without this a
    `--mode story --limit 1` smoke run would replace a full 174-household village
    with a one-household file. `--limit` is a smoke-test flag and must not be
    able to destroy an artefact that cost money to build. A full run (no `limit`)
    rewrites wholesale, which is what keeps a stale record from surviving forever
    after the rendering code changes.
    """
    if limit is None or not dest.is_file():
        return {}
    return json.loads(dest.read_text(encoding="utf-8"))


def profiles_path(village: int, output_dir: Path | str, leader_mode: str = DEFAULT_LEADER_MODE) -> Path:
    """Where a village's profiles go, given the `has_leader` treatment they were built under.

    `implicit` keeps the historical `profiles_<village>.json`; `explicit` gets
    its own `profiles_<village>_explicit.json`. The name is derived rather than
    passed so a file cannot end up labelled as a treatment it does not contain.
    """
    if leader_mode not in LEADER_MODES:
        raise ValueError(f"unknown leader_mode {leader_mode!r}; expected one of {LEADER_MODES}")
    return Path(output_dir) / f"profiles_{village}{LEADER_SUFFIX[leader_mode]}.json"


def _static_record(row: pd.Series, leader_mode: str = DEFAULT_LEADER_MODE) -> dict[str, Any]:
    """The half of a profile record that costs nothing: the disclosed facts, and the
    rule-based profile built from them. Present in both modes, so a story-mode
    file is a strict superset of a facts-mode one and the two arms are visibly
    grounded in the same facts rather than in two files taken on trust.
    """
    traits = render_traits(row, leader_mode=leader_mode)
    return {"traits": traits, "static_profile": persona_facts(traits)}


def profile_village_facts(
    village: int,
    *,
    features_dir: Path | str = DEFAULT_FEATURES_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    limit: int | None = None,
    leader_mode: str = DEFAULT_LEADER_MODE,
) -> Path:
    """Mode 1 end to end. No API, no key, no cost, and no `--live` to think about.

    Deliberately has no resume path and no overwrite flag: regenerating is free
    and instantaneous, so the only sane behaviour is to rewrite the file from the
    current code. Anything else invites a stale artefact for no saving.
    """
    df, rows = _rows(village, features_dir, limit)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = profiles_path(village, output_dir, leader_mode)

    profiles = _carry_forward(dest, limit)
    profiles |= {str(int(r.hhid)): _static_record(pd.Series(r._asdict()), leader_mode) for r in rows}
    dest.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    print(f"village {village}: {len(profiles)} static profile(s), no API calls -> {dest}")
    return dest


def profile_village_story(
    village: int,
    *,
    features_dir: Path | str = DEFAULT_FEATURES_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    model: str = DEFAULT_MODEL,
    workers: int = 8,
    limit: int | None = None,
    overwrite: bool = False,
    live: bool = False,
    temperature: float = 0.9,
    leader_mode: str = DEFAULT_LEADER_MODE,
) -> Path:
    """Mode 2 end to end: render the traits, build the static profile, call the
    model for the narrative, and write all five keys per household to one JSON.

    Resumable, with the fingerprint guard of `prompt_fingerprint()`: a live run
    skips an hhid only when it already has a non-empty narrative **and** that
    narrative was produced by a byte-identical request. Change `SYSTEM_PROMPT` or
    the model and every household is regenerated, because the alternative is one
    file holding two different experiments. A household whose call fails after
    retries keeps its static half and gets no narrative, so a re-run without
    `--overwrite` retries exactly the gaps and the stale.
    """
    df, rows = _rows(village, features_dir, limit)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dest = profiles_path(village, output_dir, leader_mode)
    preview_dest = dest.with_suffix(".preview.json")

    existing: dict[str, dict] = {}
    if live and not overwrite and dest.is_file():
        existing = json.loads(dest.read_text(encoding="utf-8"))

    # The static half is rebuilt from the current code every run, for every
    # household, in both modes -- it is free, so there is no reason to carry a
    # stale copy forward alongside a fresh narrative.
    profiles: dict[str, dict[str, Any]] = _carry_forward(dest, limit)
    jobs: list[tuple[str, dict[str, str]]] = []
    stale = 0
    for r in rows:
        hhid = str(int(r.hhid))
        record = _static_record(pd.Series(r._asdict()), leader_mode)
        request = build_request(record["traits"], model)
        sha = prompt_fingerprint(request)

        prior = existing.get(hhid)
        if prior and prior.get("narrative_profile"):
            if prior.get("prompt_sha") == sha:
                record |= {k: prior[k] for k in ("prompt", "prompt_sha", "narrative_profile", "usage") if k in prior}
                profiles[hhid] = record
                continue
            stale += 1  # same household, different prompt -- regenerate rather than mix
        profiles[hhid] = record
        jobs.append((hhid, request))

    note = f", {stale} stale (prompt changed since they were written)" if stale else ""
    print(
        f"village {village}: {len(jobs)} narrative(s) to generate "
        f"(of {len(rows)} considered, {len(rows) - len(jobs)} already done{note})"
    )

    if not live:
        preview = {
            hhid: profiles[hhid] | {"prompt": request, "prompt_sha": prompt_fingerprint(request)}
            for hhid, request in jobs
        }
        preview_dest.write_text(json.dumps(preview, indent=2), encoding="utf-8")
        print(f"village {village}: dry-run -- {len(jobs)} request(s) would be made -> {preview_dest}")
        return preview_dest

    if jobs:
        client = _load_openai_client()
        errors: list[str] = []
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(generate_narrative, client, request, temperature=temperature): (hhid, request)
                for hhid, request in jobs
            }
            for fut in as_completed(futures):
                hhid, request = futures[fut]
                try:
                    text, usage = fut.result()
                except Exception as exc:  # noqa: BLE001 -- one household's failure must not sink the village
                    errors.append(hhid)
                    print(f"village {village}: hhid {hhid} failed: {exc}", file=sys.stderr)
                    continue
                profiles[hhid] |= {
                    "prompt": request,
                    "prompt_sha": prompt_fingerprint(request),
                    "narrative_profile": text,
                    "usage": usage,
                }
                done += 1
                if done % 10 == 0 or done == len(jobs):
                    print(f"village {village}: {done}/{len(jobs)} narratives generated")

        tail = f" ({len(errors)} failed, re-run to retry)" if errors else ""
        print(f"village {village}: {len(profiles)} households{tail}")
        fresh = (profiles[h] for h, _ in jobs if profiles[h].get("usage"))
        print(f"village {village}: this run  {format_usage(total_usage(fresh))}")
        if existing:
            print(f"village {village}: file     {format_usage(total_usage(profiles.values()))}")
    else:
        print(f"village {village}: nothing to generate")

    dest.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    print(f"village {village}: {len(profiles)} profile(s) -> {dest}")
    print(f"village {village}: {format_balance(balance_report(profiles, df))}")
    return dest


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--mode",
        choices=MODES,
        default="facts",
        help="facts: static profile only, no API, no cost (default). story: + the LLM narrative.",
    )
    p.add_argument(
        "--villages",
        type=int,
        nargs="+",
        default=None,
        help="village numbers (default: every village with an output/features/hh_features_<village>.csv)",
    )
    p.add_argument(
        "--leader-mode",
        choices=LEADER_MODES,
        default=DEFAULT_LEADER_MODE,
        help=(
            "how has_leader reaches the fact listing. implicit: occupation fallback only, "
            "-> profiles_<village>.json (default). explicit: a stated trait on leader households only, "
            "omitted entirely on the rest, -> profiles_<village>_explicit.json"
        ),
    )
    p.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"where profiles_<village>.json goes (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"story mode, default: {DEFAULT_MODEL}")
    p.add_argument("--workers", type=int, default=8, help="story mode: concurrent API calls (default: 8)")
    p.add_argument("--limit", type=int, default=None, help="only profile the first N households per village")
    p.add_argument("--overwrite", action="store_true", help="story mode: regenerate narratives that already exist")
    p.add_argument(
        "--live",
        action="store_true",
        help="story mode: actually call the OpenAI API (default: dry-run preview, no cost, no calls made)",
    )
    p.add_argument("--temperature", type=float, default=0.9)
    a = p.parse_args(argv)

    if a.mode == "facts" and a.live:
        print("note: --live has no meaning in facts mode; it makes no API calls by construction", file=sys.stderr)

    villages = a.villages if a.villages else discover_villages(a.features_dir)
    if not villages:
        print(f"error: no hh_features_<village>.csv found in {a.features_dir}", file=sys.stderr)
        return 1

    ok = True
    for v in villages:
        try:
            if a.mode == "facts":
                profile_village_facts(
                    v,
                    features_dir=a.features_dir,
                    output_dir=a.output_dir,
                    limit=a.limit,
                    leader_mode=a.leader_mode,
                )
            else:
                profile_village_story(
                    v,
                    features_dir=a.features_dir,
                    output_dir=a.output_dir,
                    model=a.model,
                    workers=a.workers,
                    limit=a.limit,
                    overwrite=a.overwrite,
                    live=a.live,
                    temperature=a.temperature,
                    leader_mode=a.leader_mode,
                )
        except FileNotFoundError as e:
            print(f"warning: skipping village {v}: {e}", file=sys.stderr)
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
