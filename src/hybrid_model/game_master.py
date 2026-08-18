"""The hybrid model: BCDJ's information model, an LLM's adoption decision.

    python -m src.hybrid_model.game_master --design A1B0C1D1            # dry run, no calls
    python -m src.hybrid_model.game_master --design A1B0C1D1 --live     # one replicate, for real

One run is one village. Transmission is `diffusion_model.m` step 2, transliterated
and unchanged -- a node-level Bernoulli draw per ordered edge per round, at qP if
the sender has joined and qN if it has not. Take-up is an LLM call. Nothing else
about the loop moves, which is what licenses attributing any difference between
this and the BCDJ baseline to the substitution alone
(`docs/experiment_design.md` §1.3, §5.1).

The loop
--------

    t = 0   the leaders are approached by the MFI directly
    t = 1   every household informed before this round and not yet asked is
            prompted once -> (Y)/(N)
            then every informed household draws once per ordered edge at
            q_i = qP if it joined, qN if it did not
    ...
    t = T   stop. T is the village's last trimester, `panel.t.max()`, verified
            equal to BCDJ's own ceil(months/4)+1 for all 43 analysis villages.

Two treatments, and only the first is built
-------------------------------------------

**I -- one prompt per household per run.** A household reached by several
neighbours in the same round is prompted about one of them, drawn uniformly; the
whole sender set is logged but only one informer is shown. Whatever it answers it
is then `informed`, and it is never prompted again however many times transmission
reaches it later. This is `contagiousbefore` in the Matlab, exactly, and it keeps
the prompt byte-identical to the adoption-rate pilot's in every case.

**II -- one prompt per received transmission, until adoption.** Not implemented.
It needs a memory model, a belief state and an update rule before it can run, and
those are not decided; see `_decide_per_transmission`. It is
`docs/experiment_design.md` §9.1 and it changes the estimand, so it gets its own
re-baselined BCDJ comparison when it lands.

Why the wording here is a copy and not an import
------------------------------------------------

The label maps and, later, the prompt templates are the adoption-rate pilot's,
reproduced rather than imported from `src/pilot/adoption_rate_pilot.py`. That
pilot is a completed experiment whose logs are only readable against the wording
it ran under, and §10.7 fixes an instrument per experiment: the hybrid's prompt
has to be editable without touching a finished run's instrument. The cost is two
copies of the same words, and the check on them is a diff.

What an agent may hold
----------------------

`build_village` reads the feature table through an explicit `usecols` allowlist,
so `_adopted` is never in memory on the construction path and cannot reach a
prompt through an agent that has no route to it (`docs/household_design.md` §1).
The ground truth is read separately, by the scorer, after a run is over.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

try:
    from .. import data_loader as dl
    from ..llm import load_client, one_call
except ImportError:  # running as a script, not a package
    # `python src/hybrid_model/game_master.py` puts src/hybrid_model on sys.path,
    # not src, so the sibling modules are only importable once src itself is on it.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import data_loader as dl  # type: ignore[no-redef]
    from llm import load_client, one_call  # type: ignore[no-redef]


VILLAGE = 6

# BCDJ's published point estimates for the information model (`Main_models_1_3.m`,
# the two-step-optimal block), pooled over 43 villages. They are the transmission
# half of the hybrid and are never fitted here.
QN = 0.09
QP = 0.45

# The CLEANED table and the profiles built from it, as the pilot ran: both carry
# the merged subcaste spelling, so one subcaste is named one way in the fields and
# in the narrative. `profiles_6.json` is post-merge -- `profiles_6.pre_clean.json`
# is what it replaced.
DEFAULT_FEATURES = Path("output/features/CLEANED_hh_features_{village}.csv")
DEFAULT_PROFILES = Path("output/profiles/profiles_{village}.json")

# The MFI's own approach to a leader household. Leaders were told directly, so
# their prompt carries this line instead of a neighbour, and axes B and C have
# nothing to bind to. Placeholder wording, to be revised.
MFI_MESSAGE = """
The organisation's staff identified your household to have a 'leader' role in the village through occupation.
They invited your household to a meeting where they explained the programme.
"""

# `profiler.UNKNOWN`'s wording. R1 of the household doc: a survey-derived fact is
# always rendered, "not known" rather than dropped, so a non-surveyed household is
# short four facts rather than short four lines.
UNKNOWN = "not known"

# The pilot's vocabulary -- see the module docstring on why these are a copy.
RELIGION_LABELS = {"hinduism": "Hindu", "islam": "Muslim", "christianity": "Christian"}
ELECTRICITY_LABELS = {0: "none", 1: "yes, a government connection", 2: "yes, a private connection"}

# --------------------------------------------------------------------------
# The instrument. Every word an agent reads is in this block and `MFI_MESSAGE`
# above it, reproduced from the adoption-rate pilot -- see the module docstring
# on why these are a copy and not an import. Revised by hand, fixed before the
# first live call, and any later change recorded (§10.7).
# --------------------------------------------------------------------------

BASE_CONTEXT = """
An institution providing microfinance services has started a new programme in villages across Karnataka, India.
You have been informed that their services are now available in your village too.
You are the head of a household in this village, and as the head you represent your household and its interests.
"""

DEMOGRAPHIC_ENHANCEMENT = """
Your household has the following characteristics:
- Religion: {religion}
- Caste: {caste}
- Household size: {hh_size}
- Number of rooms: {num_rooms}
- Number of beds: {num_beds}
- Access to electricity: {electricity}
- Access to a private latrine: {latrine}
- Member of a savings self-help group: {savings_group}
- Access to a bank account: {bank_account}
"""

NARRATIVE_ENHANCEMENT = """
Your household has been described as follows:
{narrative}
"""

INFORMER = "You were told about the programme by a neighbour."
INFORMER_PROFILE = """
Your neighbour has the following characteristics:
- Religion: {religion}
- Caste: {caste}
- Household size: {hh_size}
- Number of rooms: {num_rooms}
- Number of beds: {num_beds}
- Access to electricity: {electricity}
- Access to a private latrine: {latrine}
- Member of a savings self-help group: {savings_group}
- Access to a bank account: {bank_account}
- Occupation: {occupation}
"""
INFORMER_NARRATIVE = """
Your neighbour has been described as follows:
{narrative}
"""
JOINER = "They joined the programme."
NON_JOINER = "They have not joined the programme."

FORMAT_INSTRUCTION = "Does your household join the programme? You must highlight your decision with the token {Y} for yes or {N} for no at the end of your response, on a new line, and nothing else."
MOA_INSTRUCTION = """
You should decide whether your household joins by answering the following three questions:
1. What kind of situations is this?
2. What kind of person am I?
3. What would a person like me do in a situation like this?

You must highlight your decision with the token {Y} for yes or {N} for no at the end of your response, on a new line, and nothing else.
"""

DT_INSTRUCTION = """
You should decide whether your household joins by conducting a decision-theoretic analysis.

Use everything you have been told and your own subjective judgement to fill out a decision matrix over two actions -- joining the programme and not joining it -- and three states of nature describing what taking a loan would do for your household: it turns out beneficial, it turns out to have limited effect, or it turns out harmful.

For each state, estimate the probability that it is the state you are in, give the utility your household would receive under that state from each of the two actions, and state the evidence that justifies those numbers. The three probabilities must sum to 1.

Then give your decision: Y if your household joins the programme, N if it does not.
"""

# The levels of each axis, in the order `design_label` numbers them. DT is a
# level of the instruction axis rather than an axis of its own, because it and
# MOA cannot both be in effect: one instruction ends the prompt, so the
# combination cannot be expressed in the first place.
PROFILE_LEVELS = ("", "DEMOGRAPHIC", "NARRATIVE")
INFORMER_LEVELS = ("", "DEMOGRAPHIC", "NARRATIVE")
ENDORSEMENT_LEVELS = ("", "ENDORSEMENT")
INSTRUCTION_LEVELS = ("", "MOA", "DT")

# What `build_village` is allowed to read. An allowlist rather than a drop-list
# because this is the one place the `_adopted` firewall could quietly be lost:
# the column is not excluded here, it is never loaded.
#
# `row`, `hh_num`, `has_leader` and `in_giant` are mechanics -- they place a
# household in the adjacency, seed it, and prune it. They are not rendered, and
# `profiler.FORBIDDEN_FIELDS` bars the last two from ever being.
FEATURE_COLUMNS = (
    "row",
    "hh_num",
    "hhid",
    "has_leader",
    "in_giant",
    "religion",
    "subcaste",
    "capita",
    "rooms",
    "beds",
    "electricity",
    "own_latrine",
    "has_shg",
    "has_savings",
    "occupation_head",
)


@dataclass
class HH_Agent:
    """One household: an identity, its place in the network, and what may be said about it.

    Deliberately thinner than `src/agent.py`'s agent of the same name. That one
    carries a message ledger, a round clock and a mutable `AgentState`, because
    its loop advanced every agent in lockstep. This one carries no run state at
    all: `informed`, `asked` and `adopted` live in arrays indexed by `idx`,
    owned by the run. Two reasons, and the second is the one that matters.

    Transmission is matrix algebra -- `(A * q[:, None]) > rand(n, n)` -- so the
    mechanics need a row-ordered array view regardless, and keeping the state
    there rather than on the objects means there is one copy of it rather than
    two that can disagree. And a replicate is a fresh set of arrays over the
    same agents, so S runs need no `reset()` and cannot inherit each other's
    state, which is the whole point of running S of them.

    Treatment II will need per-agent memory, and that is the one thing this
    layout does not already have a home for. It gets a per-run companion
    structure when the treatment is specified, not a mutable field here.

    Attributes
    ----------
    hh_id
        The bundle's `hhid`, village-prefixed (6001..6114). Identity and join
        key. Never rendered into a prompt -- household doc §4.6: a numeric id
        invites the model to invent orderings that do not exist.
    idx
        Position in the row-ordered population this agent was built with, and so
        the index into the adjacency submatrix and into every state array. When
        `build_village` prunes to the giant component this is 0..106, *not* the
        adjacency row -- `row` is that.
    row
        1-based `adjmatrix_key` over the whole village, pruned or not. Provenance
        for the join back to the feature table.
    is_leader
        `has_leader`. The seed set, and the one thing that changes which prompt a
        household is asked. Not privileged (§4.7): a household legitimately knows
        the organisation spoke to it.
    neighbours
        hhids this household has an edge to, from the real network. Mechanics
        only. §5 is explicit that degree must not be stated numerically to the
        model -- "you have 17 friends" reads as an instruction to be influential
        and manufactures the network effect being measured. Network position
        reaches an agent only through who actually speaks to it.
    fields
        Every `{placeholder}` the prompt templates read, already rendered to
        strings. One dict serves the ego templates and the informer ones alike:
        `str.format` ignores the keys a template does not use, so the demographic
        block simply never asks for the narrative. Holding the rendered strings
        rather than the row is what makes the allowlist structural -- there is no
        table behind this object to read `_adopted` out of.
    """

    hh_id: int
    idx: int
    row: int
    is_leader: bool
    neighbours: tuple[int, ...] = ()
    fields: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def degree(self) -> int:
        return len(self.neighbours)

    @property
    def narrative(self) -> str:
        return self.fields.get("narrative", "")

    def __repr__(self) -> str:
        kind = "leader" if self.is_leader else "household"
        return f"HH_Agent({self.hh_id}, {kind}, idx={self.idx}, deg={self.degree})"


def _render_fields(row: pd.Series, narrative: str) -> dict[str, str]:
    """One household's values for every placeholder the templates use.

    Missing values render as "not known" rather than raising. The pilot could
    refuse them -- it ran on four hand-picked households chosen to have every
    field -- but 70 of village 6's 114 households have no individual survey block
    at all, so subcaste, occupation and the two financial facts are absent for
    the majority and "not known" is the only honest rendering.

    `occupation_head` gets no leader-derived fallback, unlike
    `profiler.render_traits` under its `implicit` mode. The fallback exists there
    to colour a persona; here the only template with an occupation line is the
    *informer* one, and `profiler.render_neighbour_profile` refuses to leak
    leader status into a neighbour's description. A fallback with no legitimate
    destination is just a leak, so there is none.
    """

    def yes_no(value: object) -> str:
        return UNKNOWN if pd.isna(value) else ("yes" if bool(value) else "no")

    def whole(value: object) -> str:
        return UNKNOWN if pd.isna(value) else f"{int(value)}"

    def text(value: object) -> str:
        if pd.isna(value):
            return UNKNOWN
        return str(value).strip().lower() or UNKNOWN

    religion = str(row["religion"]).strip()
    return {
        "religion": RELIGION_LABELS.get(religion.lower(), religion.title()),
        "caste": UNKNOWN if pd.isna(row["subcaste"]) else str(row["subcaste"]).strip().title(),
        "hh_size": whole(row["capita"]),
        "num_rooms": whole(row["rooms"]),
        "num_beds": whole(row["beds"]),
        "electricity": ELECTRICITY_LABELS[int(row["electricity"])],
        "latrine": yes_no(row["own_latrine"]),
        "savings_group": yes_no(row["has_shg"]),
        "bank_account": yes_no(row["has_savings"]),
        "occupation": text(row["occupation_head"]),
        "narrative": narrative.strip(),
    }


def build_village(
    village: int = VILLAGE,
    features_path: Path | str | None = None,
    profiles_path: Path | str | None = None,
    root: Path | str | None = None,
    giant_only: bool = True,
) -> tuple[list[HH_Agent], list[HH_Agent]]:
    """The agent population of one village, split into leaders and everyone else.

    Returns `(leaders, households)`, both sorted by adjacency row. Together they
    are the whole population in row order, which is the index convention every
    array in a run shares: `idx` is the position in that combined ordering, so
    `leaders[0].idx` is wherever the first leader falls among all of them, not 0.
    Splitting is presentation; `idx` is the truth.

    Three empirical objects enter, and they are the only ones a run is allowed:
    the real households, the real network, and the real leader seeds
    (`docs/experiment_design.md` §3). The horizon comes from `panel.dta` and the
    narratives from the profiles file; neither is an outcome. `_adopted` is not
    read here at all -- see `FEATURE_COLUMNS`.

    Parameters
    ----------
    giant_only
        Prune to the giant component, as BCDJ do for everything they score
        (village 6: 107 of 114). Off keeps the isolates, which are a free
        correctness check -- a model that has one adopt is a delivery bug, since
        nobody can ever speak to it -- but puts the run on a different
        denominator from the baseline.

    Raises
    ------
    RuntimeError
        If the feature table and the adjacency disagree about how many households
        there are, which household sits in which row, or which are in the giant
        component. Those three are the joins every later step assumes, and a
        silent misalignment would be a run that scores the wrong households
        against each other.
    """
    features_path = Path(str(features_path) if features_path else str(DEFAULT_FEATURES).format(village=village))
    profiles_path = Path(str(profiles_path) if profiles_path else str(DEFAULT_PROFILES).format(village=village))

    frame = pd.read_csv(features_path, usecols=list(FEATURE_COLUMNS)).sort_values("row").reset_index(drop=True)
    v = dl.load_village(village, root=root if root is not None else dl.DEFAULT_ROOT)

    # The three joins, checked rather than assumed. `hh_num` against `hh_key` is
    # the strong one: it says row i of the table is row i of the matrix for the
    # same household, not merely that the two have the same length.
    if len(frame) != v.n:
        raise RuntimeError(f"v{village}: {len(frame)} rows in {features_path.name} but {v.n} in the adjacency")
    if not (frame.hh_num.to_numpy() == v.hh_key).all():
        raise RuntimeError(f"v{village}: {features_path.name} and the adjacency disagree on row order")
    if v.in_giant is not None and not (frame.in_giant.to_numpy() == v.in_giant).all():
        raise RuntimeError(f"v{village}: {features_path.name} and the bundle disagree on the giant component")

    narratives = _load_narratives(profiles_path)
    hh_ids = frame.hhid.to_numpy(dtype=int)

    keep = frame.in_giant.to_numpy().astype(bool) if giant_only else None
    if keep is not None and not keep.any():
        raise RuntimeError(f"v{village}: no households in the giant component")

    leaders: list[HH_Agent] = []
    households: list[HH_Agent] = []
    idx = 0
    for i, hh in enumerate(hh_ids):
        if keep is not None and not keep[i]:
            continue
        row = frame.iloc[i]
        # An edge out of the giant component cannot exist -- if i is in it, so is
        # every neighbour of i -- so this filter is a no-op under `giant_only`
        # and is here to make that a fact rather than a belief.
        neighbours = tuple(int(hh_ids[j]) for j in v.neighbours(i) if keep is None or keep[j])
        agent = HH_Agent(
            hh_id=int(hh),
            idx=idx,
            row=int(row["row"]),
            is_leader=bool(int(row["has_leader"]) == 1),
            neighbours=neighbours,
            fields=_render_fields(row, narratives.get(int(hh), "")),
        )
        (leaders if agent.is_leader else households).append(agent)
        idx += 1

    if not leaders:
        raise RuntimeError(f"v{village}: no leader households, so there is nothing to seed a run with")
    return leaders, households


def _load_narratives(path: Path) -> dict[int, str]:
    """hhid -> the narrative profile, or an empty mapping if none were built.

    Absent profiles are not an error at construction: the demographic arm never
    reads them, and a mechanics-only run reads nothing. `missing_narratives`
    is the pre-flight check for the arm that does need them.
    """
    if not path.is_file():
        return {}
    records = json.loads(path.read_text(encoding="utf-8"))
    return {int(hh): (rec.get("narrative_profile") or "").strip() for hh, rec in records.items()}


def missing_narratives(agents: list[HH_Agent]) -> list[int]:
    """hhids with no narrative profile. Call before paying for a narrative-arm run.

    Discovering an empty persona at t=4, several hundred calls in, is an
    expensive way to learn it.
    """
    return [a.hh_id for a in agents if not a.narrative]


# --------------------------------------------------------------------------
# The information model: `diffusion_model.m` step 2, unchanged
# --------------------------------------------------------------------------


def population(leaders: list[HH_Agent], households: list[HH_Agent]) -> list[HH_Agent]:
    """`build_village`'s two lists put back in `idx` order.

    The split is presentation and this undoes it. Everything below is indexed by
    `idx`, so this ordering -- not the order the two lists happen to be in -- is
    what an array position means.
    """
    pop = sorted([*leaders, *households], key=lambda a: a.idx)
    if [a.idx for a in pop] != list(range(len(pop))):
        raise RuntimeError("agents do not form a contiguous 0..n-1 index; were two villages mixed?")
    return pop


def adjacency_matrix(pop: list[HH_Agent]) -> np.ndarray:
    """The real network as an (n, n) boolean matrix in `idx` order.

    Rebuilt from the agents' own neighbour lists rather than re-read from the
    bundle, so the matrix the mechanics run on is the same object the agents
    were wired with. Under `giant_only` the bundle's matrix would have to be
    re-pruned to match, and a second pruning is a second chance to get it wrong.
    """
    n = len(pop)
    by_id = {a.hh_id: a.idx for a in pop}
    A = np.zeros((n, n), dtype=bool)
    for a in pop:
        for j in a.neighbours:
            A[a.idx, by_id[j]] = True
    if A.diagonal().any():
        raise RuntimeError("the adjacency has a self-loop")
    if not (A == A.T).all():
        raise RuntimeError("the adjacency is not symmetric")
    return A


def transmit(
    A: np.ndarray,
    informed: np.ndarray,
    adopted: np.ndarray,
    rng: np.random.Generator,
    qN: float = QN,
    qP: float = QP,
) -> np.ndarray:
    """One period of BCDJ's information flow. Returns `hit[i, j]` -- i told j.

    `diffusion_model.m`'s step 2, transliterated:

        transmitPROB = (contagious & infected)*qP + (contagious & ~infected)*qN
        contagionlikelihood = X(contagious,:).*(transmitPROB(contagious)*ones(1,N))
        contagious = ((contagionlikelihood > rand(C,N))'*ones(C,1) > 0) | contagiousbefore

    Matlab's `contagious` is `informed` here and its `infected` is `adopted`; the
    names are the only thing that changed. Four properties of that line are the
    model rather than incidental, and all four are reproduced:

    **The rate is node-level.** `q_i` depends on whether *i* has taken up and on
    nothing else -- not on who *j* is, not on how often *i* has spoken to them,
    not on the round. This is exactly what `docs/experiment_design.md` §5.3 says
    a Stage II transmission policy can express and this cannot, so it must not
    be quietly improved. It is the half of the hybrid that stays BCDJ's.

    **Everyone ever informed transmits, every period.** `contagious` accumulates
    -- it is OR'd with `contagiousbefore` at the end of every period -- so it is
    the whole informed set and not the newly informed. A household that heard in
    round 1 and refused still speaks in round 5, at qN.

    **The draw is per ordered edge, per period.** *i* telling *j* at t and again
    at t+1 are independent events, and a neighbour who has already heard is drawn
    for anyway. Both are kept: the second is harmless under treatment I -- an
    already-informed household is never re-asked -- but removing it would change
    which draws land where and so the informed-set trajectory.

    **An adopter transmits at qP in the period it adopts.** Step 1 runs before
    step 2 within a period and `adopted` is updated in between, so a household
    that joins at t is already a taker when it speaks at t. That is the caller's
    ordering to honour, not this function's.

    One deliberate departure, and it is about the random stream rather than the
    model. Matlab draws `rand(C, N)` where C is the number contagious, so the
    number of uniforms consumed in a period depends on the state. This draws the
    full `(n, n)` every period regardless. The distribution is identical -- i.i.d.
    uniforms per ordered pair either way -- but the consumption is fixed, which
    is what lets two arms seeded alike stay on the same stream while their
    adoption histories diverge. Common random numbers are the main variance
    reduction available at S = 20, so the property is worth the wasted draws
    (11,449 doubles a period on village 6).

    Returns the full sender x receiver matrix rather than the union, because
    treatment I needs to know *who* reached a household in order to pick the one
    informer its prompt names. `hit.any(axis=0)` is the union BCDJ uses.

    Parameters
    ----------
    A
        (n, n) boolean adjacency, symmetric, zero diagonal.
    informed
        (n,) boolean. Everyone who has ever heard, Matlab's `contagious`.
    adopted
        (n,) boolean. Everyone who has joined, Matlab's `infected`. Must be a
        subset of `informed` -- adopting without hearing is a delivery bug, and
        it is checked rather than assumed.
    rng
        Consumed for exactly n*n uniforms. Keep it separate from the generator
        that samples informers, or the two couple and common random numbers stop
        working the moment a household hears from two neighbours instead of one.
    """
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError(f"adjacency must be square, got {A.shape}")
    if informed.shape != (n,) or adopted.shape != (n,):
        raise ValueError(f"state arrays must be ({n},), got {informed.shape} and {adopted.shape}")
    if np.any(adopted & ~informed):
        raise RuntimeError("a household has adopted without ever being informed")

    # Zero for anyone who has not heard, so an uninformed household cannot
    # transmit: rng.random() is in [0, 1) and the comparison is strict.
    q = np.where(adopted, qP, qN) * informed
    return A & (q[:, None] > rng.random((n, n)))


# --------------------------------------------------------------------------
# Designs, and the prompt one assembles
# --------------------------------------------------------------------------


def design_label(
    profile_enhancement: str = "",
    informer_enhancement: str = "",
    endorsement_enhancement: str = "",
    instruction: str = "",
) -> str:
    """`A1B0C1D0` -- one digit per axis: ego profile, informer profile, endorsement, instruction.

    The pilot's numbering, unchanged, so a design named there is the same design
    named here and the two experiments' logs can be read side by side. The digit
    is the level's index in that axis's tuple.
    """
    levels = {"": 0, "DEMOGRAPHIC": 1, "NARRATIVE": 2}
    instructions = {"": 0, "MOA": 1, "DT": 2}
    return (
        f"A{levels[profile_enhancement]}"
        f"B{levels[informer_enhancement]}"
        f"C{int(bool(endorsement_enhancement))}"
        f"D{instructions[instruction]}"
    )


def effective_design(design: tuple[str, str, str, str], is_leader: bool) -> tuple[str, str, str, str]:
    """The design as it is actually rendered to one household.

    For a non-leader this is the design. For a leader the B and C axes are
    dropped, because a leader was approached by the MFI and there is no
    neighbour informer for them to describe or endorse. So a run of `A1B2C1D0`
    asks its 22 leaders `A1B0C0D0` and everyone else `A1B2C1D0`.

    Projection rather than refusal: a leader with no informer is the model
    working, not a caller error, and a run cannot stop at t=1 over it. It is
    made a public function -- and logged per row beside the nominal design --
    so that the collapse is a recorded fact rather than something a reader has
    to reconstruct from `is_leader` afterwards. Every leader row in an A0 design
    is also the same prompt whatever B and C were, which is worth seeing before
    those rows are pooled with anything.
    """
    profile, informer, endorsement, instruction = design
    if is_leader:
        return (profile, "", "", instruction)
    return (profile, informer, endorsement, instruction)


def get_prompt(
    agent: HH_Agent,
    profile_enhancement: str = "",
    informer_enhancement: str = "",
    endorsement_enhancement: str = "",
    instruction: str = "",
    informer: HH_Agent | None = None,
    informer_adopted: bool | None = None,
) -> str:
    """What one household is asked, in one round of one run.

    The pilot's assembly, with the axis names and the part order unchanged, and
    three differences that are all consequences of running it inside a loop.

    **The endorsement is a simulated state, never the ground truth.** The pilot
    read `_adopted` off the feature table, because its informer was a fixed
    household standing for "an adopter" or "a non-adopter". Here the informer is
    whoever actually reached this household in this run, and what it says about
    itself is what it decided *in this run* -- so `informer_adopted` is passed
    in from the run's own state array and there is no route from here to the
    outcome column. This is the firewall of household doc §4.7 and it is the
    single most important line in this function.

    **A leader is asked a different question.** `MFI_MESSAGE` replaces the
    neighbour block: the organisation approached them directly, so there is no
    informer to describe and B and C are dropped by `effective_design`. The
    projection happens here rather than at the call site, so no caller can
    forget it, and the same function names what was rendered for the log.

    **The informer may be drawn but not shown.** Under B0C0 a non-leader's
    prompt carries no informer block at all -- that is what the pilot's base
    level means and it keeps the axis comparable across the two experiments. The
    loop still draws an informer for that household and still logs it, so a B0
    run and a B1 run seeded alike see the same histories and differ only in what
    the prompt said about them.

    Raises
    ------
    ValueError
        An axis is set to a level that does not exist, or a non-leader is asked
        a B or C design without the informer that axis describes. `_adopted`
        being unknown is an error rather than a default: silently endorsing as
        "has not joined" would put a fact in the prompt that nothing in the run
        supports.
    """
    if profile_enhancement not in PROFILE_LEVELS:
        raise ValueError(f"profile_enhancement must be one of {PROFILE_LEVELS}, got {profile_enhancement!r}")
    if informer_enhancement not in INFORMER_LEVELS:
        raise ValueError(f"informer_enhancement must be one of {INFORMER_LEVELS}, got {informer_enhancement!r}")
    if endorsement_enhancement not in ENDORSEMENT_LEVELS:
        raise ValueError(f"endorsement_enhancement must be one of {ENDORSEMENT_LEVELS}, got {endorsement_enhancement!r}")
    if instruction not in INSTRUCTION_LEVELS:
        raise ValueError(f"instruction must be one of {INSTRUCTION_LEVELS}, got {instruction!r}")

    profile_enhancement, informer_enhancement, endorsement_enhancement, instruction = effective_design(
        (profile_enhancement, informer_enhancement, endorsement_enhancement, instruction), agent.is_leader
    )

    parts = [BASE_CONTEXT]

    # Axis A: the deciding household describes itself. Identical for a leader
    # and a non-leader -- the seed assignment changes who told them, not who
    # they are, and `has_leader` is not a persona fact (profiler.FORBIDDEN_FIELDS).
    if profile_enhancement:
        template = DEMOGRAPHIC_ENHANCEMENT if profile_enhancement == "DEMOGRAPHIC" else NARRATIVE_ENHANCEMENT
        parts.append(template.format(**agent.fields))

    if agent.is_leader:
        # Not privileged information (household doc §4.7): a household plainly
        # knows the organisation's staff came to see it.
        parts.append(MFI_MESSAGE)
    elif informer_enhancement or endorsement_enhancement:
        # Axes B and C both speak about the same neighbour, so the informer line
        # is added once for either.
        if informer is None:
            raise ValueError(
                f"household {agent.hh_id} is not a leader and design "
                f"{design_label(profile_enhancement, informer_enhancement, endorsement_enhancement, instruction)} "
                "describes an informer, but none was given"
            )
        parts.append(INFORMER)
        if informer_enhancement:
            template = INFORMER_PROFILE if informer_enhancement == "DEMOGRAPHIC" else INFORMER_NARRATIVE
            parts.append(template.format(**informer.fields))
        if endorsement_enhancement:
            if informer_adopted is None:
                raise ValueError(
                    f"household {agent.hh_id} is asked an endorsement design but the informer's "
                    f"simulated decision was not passed; it is never read from the feature table"
                )
            parts.append(JOINER if informer_adopted else NON_JOINER)

    # Axis D: MOA and DT replace the plain instruction rather than adding to it,
    # so exactly one of the three ends every prompt. That replacement is also
    # what keeps MOA and DT from ever running together. DT asks for no format,
    # because the shape of its answer is enforced by the schema on the request.
    template = {"": FORMAT_INSTRUCTION, "MOA": MOA_INSTRUCTION, "DT": DT_INSTRUCTION}[instruction]
    parts.append(template.format(Y=YES_TOKEN, N=NO_TOKEN))

    return "\n\n".join(part.strip() for part in parts if part.strip())


# --------------------------------------------------------------------------
# The elicitation: one call, one decision
# --------------------------------------------------------------------------

# The pilot's models and the keys.json block each is reached through. All three
# speak the OpenAI Responses API; claude and grok differ only by base_url.
class LLMs(Enum):
    GPT_5_4_NANO = "gpt-5.4-nano"
    GPT_5_6_LUNA = "gpt-5.6-luna"
    HAIKU_4_5 = "claude-haiku-4-5-20251001"
    GROK_4_2 = "grok-4.20-0309-non-reasoning"


PROVIDERS = {
    LLMs.GPT_5_4_NANO: "openai",
    LLMs.GPT_5_6_LUNA: "openai",
    LLMs.HAIKU_4_5: "claude",
    LLMs.GROK_4_2: "grok",
}

# Which models this path has actually been run against, DT's strict schema
# included. The gate in `get_response` reads this rather than naming one model,
# so wiring the next one up is an entry here and a run, not an edit to the
# elicitation. `full_llm_model.game_master` gates its transmission call on the
# same set.
WIRED_UP = frozenset({LLMs.GPT_5_4_NANO, LLMs.GPT_5_6_LUNA})

# The reasoning budget, for the models that take one. A reasoning model left at
# its provider default spends most of a response thinking, and this study asks a
# household for one token of decision -- `"none"` buys the answer without the
# thinking, and keeps the response short enough that MAX_OUTPUT_TOKENS stays as
# far from binding as it was in the pilot. Absent from this map means the
# parameter is not sent at all, which is what every model here did before.
REASONING_EFFORT = {
    LLMs.GPT_5_6_LUNA: "none",
}

# What the model must answer with -- the tokens the pilot's instruction asks for
# and the pilot's parser reads. The prompt and the parser are two halves of one
# contract, which is why they are copied into this file together rather than one
# being imported and the other rewritten.
YES_TOKEN = "(Y)"
NO_TOKEN = "(N)"

# The third value a decision can take. A response with no decision in it is still
# a response worth keeping: the rate of these is a fact about the prompt design,
# and the text is the only record of what the model said instead.
PARSING_ERROR = "PARSING ERROR"

# The one call parameter set; everything else, temperature included, stays at
# whatever each provider ships, as the pilot ran. 1024 is the pilot's cap and it
# is not close to binding: over its 2,100 calls the longest response was 643
# tokens and none was truncated.
MAX_OUTPUT_TOKENS = 1024

# The three states of nature, in the order they are asked for and reported.
DT_STATES = ("beneficial", "limited_effect", "harmful")

_DT_STATE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["probability", "Y_utility", "N_utility", "evidence"],
    "properties": {
        "probability": {"type": "number", "description": "The probability that this is the state of nature."},
        "Y_utility": {"type": "number", "description": "The utility of joining the programme under this state."},
        "N_utility": {"type": "number", "description": "The utility of not joining the programme under this state."},
        "evidence": {
            "type": "array",
            "items": {"type": "string"},
            "description": "What justifies the probability and the utilities of this state.",
        },
    },
}

# Keyed by state name rather than a list of state objects: strict mode rejects
# `minItems`/`maxItems`, so an array could come back with two states or with
# `harmful` twice, whereas an object with three required properties and
# `additionalProperties: false` can only come back as all three, once each.
DT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["states", "decision"],
    "properties": {
        "states": {
            "type": "object",
            "additionalProperties": False,
            "required": list(DT_STATES),
            "properties": {state: _DT_STATE_SCHEMA for state in DT_STATES},
        },
        "decision": {"type": "string", "enum": ["Y", "N"]},
    },
}

DT_FORMAT = {"format": {"type": "json_schema", "name": "dt_analysis", "strict": True, "schema": DT_SCHEMA}}

# One equivalence class per decision. Models answer the same instruction in a
# dozen dialects -- bare or bracketed, the letter or the word, wrapped in the
# markdown they were reasoning in.
_YES = {"y", "yes"}
_NO = {"n", "no"}
_DECORATION = "()[]{}<>*_`~\"'“”‘’.,!?:;- \t"
_BRACKETED = re.compile(r"[(\[{<]\s*(y|n|yes|no)\s*[)\]}>]", re.IGNORECASE)


@dataclass
class Response:
    """One call's result: what the model said, what it decided, and what it cost.

    A record rather than the pilot's three-tuple, because the hybrid needs a
    fourth field. `attempts` is how many calls it took to get a readable
    decision, and it has to reach the log: a run's parse-failure rate is a
    property of the prompt design, and a silent retry would hide both the rate
    and the extra tokens it spent.
    """

    text: str
    decision: str
    usage: dict[str, int]
    attempts: int = 1

    @property
    def joined(self) -> bool:
        """The decision as the loop needs it -- one bit, no third value.

        **A response that could not be parsed counts as not joining.** The loop
        cannot carry a third state: a household either adopts this period or it
        does not, and it is never asked again. Refusing to resolve it would stall
        the run, and dropping the household would silently change the
        denominator every rate in the scoring is computed on.

        The bias this introduces is bounded by the parse-failure rate, which is
        measured rather than hoped for: 1 call in 2,100 over the whole pilot
        (0.05%), and `get_response` retries once before giving up, so the rate
        here should be lower again. It is logged per row and must be reported
        with any result -- if a design ever pushes it above a fraction of a
        percent, this rule stops being negligible and the design is the problem.
        """
        return self.decision == YES_TOKEN


@lru_cache(maxsize=None)
def _client(provider: str):
    """One client per keys.json block, reused across calls.

    Cached because a round's decisions are made concurrently and building a
    client per call would be the expensive part of a cheap request. Both this
    and `llm.one_call` are stateless from the caller's side, so a round's
    threads share the one client safely.
    """
    return load_client(provider)


def get_llm(label: str) -> LLMs:
    try:
        return LLMs(label)
    except ValueError:
        raise ValueError(f"Invalid LLM label: {label}. Valid labels are: {[llm.value for llm in LLMs]}")


def _as_decision(word: str) -> str | None:
    """`(Y)`, `**y**`, `{Y}`, `"Yes."` -> YES_TOKEN. Anything else -> None."""
    word = word.strip(_DECORATION).lower()
    if word in _YES:
        return YES_TOKEN
    if word in _NO:
        return NO_TOKEN
    return None


def extract_decision(response: str) -> str:
    """The decision token in a response, or ValueError if there is none.

    The instruction asks for the token last, on a line of its own, so that is
    where a loose form is trusted: the final line, whole or its last word. A
    bracketed token is distinct enough to be read from anywhere, and the last one
    wins, because a model that rehearses "(Y) if ... (N) if ..." commits at the end.
    """
    text = (response or "").strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        last = lines[-1]
        decision = _as_decision(last) or _as_decision(last.split()[-1])
        if decision is not None:
            return decision

    found = _BRACKETED.findall(text)
    if found:
        return YES_TOKEN if found[-1].lower() in _YES else NO_TOKEN

    raise ValueError(f"Response does not contain a valid decision token: {response}")


def parse_dt(response: str) -> dict | None:
    """The decision matrix in a DT response, or None if there is no usable one.

    Strict structured output guarantees the shape -- the three states, their four
    fields each, the types, and a decision that is `Y` or `N` -- so this is not a
    lenient parse of a model's idea of JSON. What it guards against is the two
    ways a well-formed request still comes back unusable: a response truncated by
    the token limit, which arrives as a prefix of the object, and anything the
    schema was not applied to at all.

    The arithmetic the schema cannot express is *not* grounds for rejection. A
    set of three probabilities that does not sum to 1 makes the distribution
    invalid, but the decision is still the model's answer to the same question
    every other design asks, and dropping it would bias the adoption rate by a
    property of the analysis rather than of the answer.
    """
    text = (response or "").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    if not isinstance(payload, dict) or payload.get("decision") not in ("Y", "N"):
        return None
    states = payload.get("states")
    if not isinstance(states, dict) or set(states) != set(DT_STATES):
        return None
    for state in DT_STATES:
        block = states[state]
        if not isinstance(block, dict):
            return None
        try:
            values = [float(block[key]) for key in ("probability", "Y_utility", "N_utility")]
        except (KeyError, TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in values):
            return None
    return payload


def get_response(
    llm: LLMs,
    prompt: str,
    instruction: str = "",
    max_parse_attempts: int = 2,
) -> Response:
    """One household's decision: call the model, read the answer out of it.

    Mirrors the pilot's function of the same name, and differs from it in four
    places -- each because the hybrid asks inside a loop rather than at a fixed
    set of four households.

    **A leader's call is not a special case here.** The MFI variant changes what
    the prompt says, not how it is asked or how the answer is read, and it is
    built upstream by `get_prompt`. What the leader path does constrain is the
    *design*: with no neighbour informer there is nothing for axes B and C to
    bind to, so a leader is only ever asked an A-and-D design, and `instruction`
    is the D half of that. The projection belongs to the design layer, and by the
    time a prompt reaches this function it is just a prompt.

    **A response with no decision in it is retried.** The pilot logged the
    failure and moved on, because a lost row there was one cell of a rate. Here
    the household still has to be resolved: the loop asks once and never again,
    and there is no third state to park it in. So the call is repeated -- a fresh
    sample, since temperature is left at the provider default -- and only a
    second failure is recorded as `PARSING_ERROR`. `Response.joined` documents
    what that then counts as, and `attempts` carries the cost of the retry into
    the log.

    **Only OpenAI is wired up**, as in the pilot -- and the pilot is the whole
    evidence base for that: all 2,100 of its calls went to `gpt-5.4-nano`, so
    what claude and grok do through this path has never been observed. They are
    reached through the same client with `base_url` pointed elsewhere and are
    not guaranteed to accept a DT design's strict schema, which is the narrow
    reason; the broad one is that an unverified provider inside a loop fails
    somewhere in the middle of a paid run. The gate lifts per model once it has
    been tested, not before.

    **It is called concurrently.** A round's decisions are independent by
    construction -- every one of them reads the state as it stood at the start of
    the round -- so `decide_round` runs them in a thread pool. Nothing here holds
    state between calls; the only shared object is the cached client.

    Parameters
    ----------
    instruction
        The design's D axis, and the only one of the four that changes the
        request rather than the prompt: a DT design asks for its decision matrix
        under a schema and reads the decision out of the matrix rather than out
        of the prose.
    max_parse_attempts
        How many times to ask before giving up and recording `PARSING_ERROR`.
        1 restores the pilot's behaviour exactly.

    Raises
    ------
    NotImplementedError
        A DT design on a model whose provider has not been shown to accept the
        strict schema.
    RuntimeError
        The call itself failed -- `llm.one_call` has already retried the
        transient cases.
    """
    if llm not in WIRED_UP:
        raise NotImplementedError(
            f"{llm.value} is not wired up yet -- only {', '.join(sorted(m.value for m in WIRED_UP))} "
            "answer so far. A model has to be tested against this path, DT's strict schema included, "
            "before a paid run is allowed to depend on it."
        )
    if max_parse_attempts < 1:
        raise ValueError(f"max_parse_attempts must be >= 1, got {max_parse_attempts}")

    request: dict[str, object] = {"model": llm.value, "input": prompt}
    if llm in REASONING_EFFORT:
        request["reasoning"] = {"effort": REASONING_EFFORT[llm]}
    if instruction == "DT":
        request["text"] = DT_FORMAT

    text, decision, usage = "", PARSING_ERROR, {}
    for attempt in range(1, max_parse_attempts + 1):
        text, usage = one_call(_client(PROVIDERS[llm]), request, max_output_tokens=MAX_OUTPUT_TOKENS)
        if instruction == "DT":
            payload = parse_dt(text)
            decision = PARSING_ERROR if payload is None else (YES_TOKEN if payload["decision"] == "Y" else NO_TOKEN)
        else:
            try:
                decision = extract_decision(text)
            except ValueError:
                decision = PARSING_ERROR
        if decision != PARSING_ERROR:
            return Response(text=text, decision=decision, usage=usage, attempts=attempt)

    # Only the last attempt's text and usage survive. The earlier ones cost
    # tokens that `attempts` accounts for but does not itemise; if that ever
    # matters, it is because the failure rate has stopped being negligible.
    return Response(text=text, decision=PARSING_ERROR, usage=usage, attempts=max_parse_attempts)


# --------------------------------------------------------------------------
# One round of decisions
# --------------------------------------------------------------------------


@dataclass
class Decision:
    """One household's take-up decision, and everything needed to audit it.

    This is the log row. It carries the prompt and the response in full because
    the run is the elicitation -- there is no cached propensity to go back to,
    so a decision that is not written down when it happens is gone. It also
    carries the things a reader would otherwise have to reconstruct: which
    design was *rendered* as against nominated, how many neighbours reached this
    household when only one was named, and whether the answer took two calls.
    """

    round: int
    idx: int
    hh_id: int
    is_leader: bool
    design: str
    effective_design: str
    n_senders: int
    sender_hh_ids: tuple[int, ...]
    informer_hh_id: int | None
    informer_adopted: bool | None
    prompt: str
    response: str
    decision: str
    joined: bool
    attempts: int
    usage: dict[str, int] = field(default_factory=dict, repr=False)
    error: str = ""

    def to_row(self) -> dict[str, object]:
        """The CSV row. `sender_hh_ids` is space-joined so the column stays one field."""
        return {
            "round": self.round,
            "hh_id": self.hh_id,
            "is_leader": int(self.is_leader),
            "design": self.design,
            "effective_design": self.effective_design,
            "n_senders": self.n_senders,
            "sender_hh_ids": " ".join(str(h) for h in self.sender_hh_ids),
            "informer_hh_id": "" if self.informer_hh_id is None else self.informer_hh_id,
            "informer_adopted": "" if self.informer_adopted is None else int(self.informer_adopted),
            "prompt": self.prompt,
            "response": self.response,
            "decision": self.decision,
            "joined": int(self.joined),
            "attempts": self.attempts,
            "input_tokens": self.usage.get("input_tokens", ""),
            "output_tokens": self.usage.get("output_tokens", ""),
            "total_tokens": self.usage.get("total_tokens", ""),
            "error": self.error,
        }


def draw_informers(
    pop: list[HH_Agent],
    deciding: np.ndarray,
    hit: np.ndarray | None,
    rng: np.random.Generator,
) -> dict[int, tuple[np.ndarray, int | None]]:
    """Who each deciding household hears from, and which one of them its prompt names.

    Returns `idx -> (all senders, the one drawn)`. Both halves are kept: the
    prompt names one neighbour, the log keeps the set. That is the difference
    between treatment I and the render-all-senders variant, and logging the set
    means a later switch to that variant has something to compare against
    without re-running anything. It also means BCDJ's endorsement regressor --
    the share of a household's informers who are takers -- is recoverable from
    the log afterwards, though no prompt was ever shown it.

    Uniform among the senders, and the uniform is the whole of it: 22% of
    newly-informed households on village 6 hear from more than one neighbour in
    the same round (up to seven), so this is a frequent case rather than a
    corner one, and which neighbour gets named is a real draw rather than a
    tie-break.

    **Consumes exactly n uniforms, whoever is deciding.** One draw per household
    in the population, indexed by `idx`, rather than one per decider. The number
    of deciders depends on the run's adoption history, so drawing per decider
    would desynchronise two arms the moment their histories differed, and common
    random numbers are the main variance reduction available at S = 20. Keep
    this generator separate from the transmission one for the same reason.

    `hit is None` is the seed round: nobody has spoken yet, everyone deciding is
    a leader, and their informer is the MFI rather than a household.
    """
    n = len(pop)
    u = rng.random(n)
    out: dict[int, tuple[np.ndarray, int | None]] = {}
    for idx in np.flatnonzero(deciding):
        idx = int(idx)
        if hit is None:
            out[idx] = (np.empty(0, dtype=int), None)
            continue
        senders = np.flatnonzero(hit[:, idx])
        if not len(senders):
            if not pop[idx].is_leader:
                raise RuntimeError(
                    f"household {pop[idx].hh_id} is deciding but nobody reached it; "
                    "a non-leader can only be asked because it was told"
                )
            out[idx] = (senders, None)
            continue
        # int(u * k) is uniform over 0..k-1 for u in [0, 1); the min() guards the
        # float edge where u rounds up to exactly k.
        out[idx] = (senders, int(senders[min(int(u[idx] * len(senders)), len(senders) - 1)]))
    return out


def decide_round(
    pop: list[HH_Agent],
    deciding: np.ndarray,
    adopted: np.ndarray,
    hit: np.ndarray | None,
    design: tuple[str, str, str, str],
    llm: LLMs,
    round_r: int,
    informer_rng: np.random.Generator,
    max_workers: int = 8,
    responder: Callable[..., Response] | None = None,
    progress: tqdm | None = None,
) -> list[Decision]:
    """Every household that has just heard, asked once, concurrently.

    Treatment I's step 1. Returns one `Decision` per household in `deciding`,
    sorted by `idx`, whatever happened to it -- including a call that failed, so
    that a round always comes back as a complete slate and the caller can write
    the whole log before deciding whether to go on.

    **The calls are independent by construction, so they are threaded.** Every
    prompt is built from the state as it stood at the start of the round, and
    all of them are built *before* any call is dispatched. That is what makes
    the round order-invariant: no decision can be conditioned on another
    decision taken in the same round, in the model or in the code. It is also
    what makes a dry run exact -- the prompts a dry run prints are the prompts a
    live run would send, not a reconstruction of them.

    **A sender's endorsement is its state as of when it spoke, and that is the
    state passed in.** Under treatment I a household decides once and is never
    asked again, and every household that transmits has already decided by the
    time it transmits -- so an informer's adoption is frozen before it speaks
    and `adopted` at the start of this round is the same value it held then.
    The invariant is asserted rather than assumed: a household deciding now
    cannot have been a sender last round.

    Parameters
    ----------
    deciding
        (n,) boolean: informed before this round and not yet asked. At the seed
        round that is the leaders; afterwards it is exactly whoever `hit`
        reached last round and had not heard before.
    adopted
        (n,) boolean, the simulated state at the start of this round. The only
        source of an informer's endorsement -- `_adopted` is not readable from
        here.
    hit
        Last round's sender x receiver matrix, or None at the seed round.
    responder
        Injected for tests and dry runs; defaults to `get_response`. A dry run
        passes something that returns a canned `Response` and spends nothing.
    progress
        `main`'s bar, advanced one step per call as it lands. Optional, because a
        round run from a notebook or a test has nothing to advance.

    Raises
    ------
    RuntimeError
        The seed round is asked to decide a non-leader; a decider was also a
        sender last round; or every call in the round failed, which is a
        provider that is down rather than a result and is not worth paying to
        continue past. Individual failures are returned on the `Decision`
        instead, in `error`.
    """
    responder = responder if responder is not None else get_response
    label = design_label(*design)

    if hit is None and not all(pop[i].is_leader for i in np.flatnonzero(deciding)):
        raise RuntimeError("the seed round can only ask leaders; nobody else has been told anything yet")
    if hit is not None:
        overlap = np.flatnonzero(deciding & hit.any(axis=1))
        if len(overlap):
            raise RuntimeError(
                f"households {[pop[int(i)].hh_id for i in overlap]} are deciding now but transmitted "
                "last round; a household that speaks has already been asked"
            )
    if np.any(adopted & deciding):
        raise RuntimeError("a household is deciding now but has already adopted")

    informers = draw_informers(pop, deciding, hit, informer_rng)

    # Build every prompt first, then dispatch. Nothing below reads run state.
    work: list[tuple[HH_Agent, Decision]] = []
    for idx in sorted(informers):
        agent = pop[idx]
        senders, chosen = informers[idx]
        informer = pop[chosen] if chosen is not None else None
        informer_adopted = bool(adopted[chosen]) if chosen is not None else None
        effective = effective_design(design, agent.is_leader)
        prompt = get_prompt(
            agent,
            *design,
            informer=informer,
            # Only passed when the design will read it; a B-only design gets a
            # neighbour's profile without a claim about what they did.
            informer_adopted=informer_adopted if effective[2] else None,
        )
        work.append(
            (
                agent,
                Decision(
                    round=round_r,
                    idx=idx,
                    hh_id=agent.hh_id,
                    is_leader=agent.is_leader,
                    design=label,
                    effective_design=design_label(*effective),
                    n_senders=len(senders),
                    sender_hh_ids=tuple(pop[int(s)].hh_id for s in senders),
                    informer_hh_id=informer.hh_id if informer is not None else None,
                    informer_adopted=informer_adopted,
                    prompt=prompt,
                    response="",
                    decision=PARSING_ERROR,
                    joined=False,
                    attempts=0,
                ),
            )
        )

    if not work:
        return []

    def ask(item: tuple[HH_Agent, Decision]) -> Decision:
        _, record = item
        try:
            reply = responder(llm, record.prompt, design[3])
        except Exception as exc:  # noqa: BLE001 -- one bad call must not lose the round's log
            record.error = f"{type(exc).__name__}: {exc}"
            return record
        record.response = reply.text
        record.decision = reply.decision
        record.joined = reply.joined
        record.attempts = reply.attempts
        record.usage = reply.usage
        return record

    # Submitted rather than mapped, so the bar advances as each call lands
    # instead of in submission order. The round is order-invariant by
    # construction -- every prompt above was built before any of them was sent --
    # so completion order is free to be whatever the provider makes it, and the
    # slate is sorted back into `idx` order below.
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        decisions = []
        for future in as_completed([pool.submit(ask, item) for item in work]):
            decisions.append(future.result())  # `ask` swallows its own exceptions
            if progress is not None:
                progress.update(1)

    if all(d.error for d in decisions):
        raise RuntimeError(
            f"every one of the {len(decisions)} calls in round {round_r} failed; "
            f"first error: {decisions[0].error}"
        )
    return sorted(decisions, key=lambda d: d.idx)


# --------------------------------------------------------------------------
# Treatment II: one prompt per received transmission. Not implemented.
# --------------------------------------------------------------------------

TREATMENT_ONCE = 1
TREATMENT_PER_TRANSMISSION = 2
TREATMENTS = (TREATMENT_ONCE, TREATMENT_PER_TRANSMISSION)


def _decide_per_transmission(
    pop: list[HH_Agent],
    informed: np.ndarray,
    adopted: np.ndarray,
    hit: np.ndarray | None,
    design: tuple[str, str, str, str],
    llm: LLMs,
    round_r: int,
    informer_rng: np.random.Generator,
    **kwargs: object,
) -> list[Decision]:
    """Treatment II's step 1, which does not exist yet.

    The rule is easy to state and that is exactly why it should not be written
    before it is designed: **there is no `informed` flag, and every household
    that has not yet adopted is prompted once for each transmission it
    receives.** Adoption is absorbing; refusal is not. A household reached by
    four neighbours in one round is asked four times; a household that refuses
    in round 2 is asked again in round 3 if anyone speaks to it.

    Five things have to be decided first, and none of them is a detail:

    1. **What the household remembers.** A second pitch is only a different
       question from the first if the agent knows there was a first. Whether
       that memory is a transcript in the prompt, a summary written by the
       agent, or a belief held in code is the substance of the treatment.
    2. **What updates, and where.** If the update is in the prompt the model
       does it and we can read it; if it is in code we have imposed a
       likelihood and should say so. The A2 decision-matrix arm of
       `docs/experiment_design.md` §2 is the obvious home for an explicit one.
    3. **Whether a repeat from the same neighbour differs from a first from a
       new one.** BCDJ's model draws both identically, but they are plainly not
       the same event to a household, and treatment II is where that difference
       could be expressed.
    4. **Whether the round index is visible.** It is not information a household
       has, but "the third time this month" and "the third time this year" are
       different pitches.
    5. **Order within a round.** Treatment I's decisions are simultaneous and
       therefore commute, which is what lets the round be threaded and asserted
       order-invariant. Four pitches to one household in one round do not
       obviously commute, and if they do not, the round has an ordering that has
       to be justified rather than inherited from `idx`.

    Two measured facts that bound the answer, both from village 6 at BCDJ's
    published rates:

    - **It costs 3.8x treatment I.** ~312 calls a run against ~81, because the
      unit is a transmission rather than a household and there are 730 ordered
      edges over 5 rounds.
    - **The naive rule is wrong, and visibly so.** Treating each pitch as an
      independent draw at the same take-up probability puts ~60 of 107
      households into adoption against a real 25. Repetition has to *do*
      something other than roll the dice again, which is the whole reason
      points 1 and 2 above cannot be skipped.

    This is `docs/experiment_design.md` §9.1. It changes the estimand -- the
    one-shot rule is BCDJ's and treatment I inherits it -- so when it lands it
    is scored against its own BCDJ baseline re-run under the same rule, never
    pooled with treatment I's.

    The signature above is the one it will need; the arguments are accepted and
    discarded so that the call site in `hybrid_run` is real rather than a
    placeholder that will have to be rewritten.
    """
    raise NotImplementedError(
        "treatment II (one prompt per received transmission) is not implemented: it needs a memory "
        "model, a belief update and a within-round ordering first. See _decide_per_transmission's "
        "docstring for the five open decisions and the two measurements that bound them."
    )


# --------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------

# Where a round index means "before the loop began" (a leader, informed by the
# MFI at seeding) and "never" (never informed, or never adopted).
SEEDED = 0
NEVER = -1


def default_rounds(village: int, root: Path | str | None = None) -> int:
    """`T` for a village: the last trimester in `panel.dta`.

    Verified equal to BCDJ's own `ceil(TMonths/4) + 1` for all 43 analysis
    villages (`docs/experiment_design.md` §7), so the two definitions agree and
    nothing has to be chosen. Village 6 is 5.
    """
    v = dl.load_village(village, root=root if root is not None else dl.DEFAULT_ROOT)
    if v.panel is None or not len(v.panel):
        raise dl.DataError(f"v{village}: no panel, so no horizon; pass rounds= explicitly")
    return int(v.panel.t.max())


@dataclass
class RunResult:
    """One replicate, as everything downstream reads it.

    The same record for a hybrid run and for the BCDJ baseline, so the scorer
    takes one type and the ladder of `docs/experiment_design.md` §1.2 reads down
    a single column. The only field a baseline run leaves empty is `decisions`:
    it has no prompts, because its take-up came from a logit rather than a model.

    Everything is in `idx` order and `hh_ids` is the join key back out.
    """

    village: int
    arm: str
    design: str
    llm: str
    treatment: int
    rounds: int
    seed: int
    replicate: int

    hh_ids: np.ndarray
    is_leader: np.ndarray
    adopted: np.ndarray
    informed: np.ndarray
    asked: np.ndarray
    adopted_round: np.ndarray
    informed_round: np.ndarray
    curve: np.ndarray
    info_curve: np.ndarray
    decisions: list[Decision] = field(default_factory=list, repr=False)
    swept: int = 0

    @property
    def n(self) -> int:
        return len(self.hh_ids)

    @property
    def n_calls(self) -> int:
        """API calls this run actually made, retries included."""
        return sum(d.attempts for d in self.decisions)

    @property
    def errors(self) -> list[Decision]:
        """Decisions whose call failed. **Check this before scoring a run.**

        A failed call is recorded as "did not join" so that the round comes back
        a complete slate, which means a provider quietly failing produces a run
        of plausible-looking refusals. Nothing downstream can tell that apart
        from a model that said no.
        """
        return [d for d in self.decisions if d.error]

    @property
    def parse_failures(self) -> list[Decision]:
        return [d for d in self.decisions if d.decision == PARSING_ERROR and not d.error]

    def usage(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for d in self.decisions:
            for key, value in d.usage.items():
                totals[key] = totals.get(key, 0) + int(value)
        return totals

    def summary(self) -> str:
        rate = self.adopted.sum() / self.n
        leader_rate = self.adopted[self.is_leader].mean() if self.is_leader.any() else float("nan")
        non_leader = ~self.is_leader
        non_leader_rate = self.adopted[non_leader].mean() if non_leader.any() else float("nan")
        return (
            f"v{self.village} {self.arm} {self.design} rep {self.replicate}: "
            f"asked {int(self.asked.sum())}, joined {int(self.adopted.sum())} ({rate:.1%}), "
            f"leaders {leader_rate:.1%}, non-leaders {non_leader_rate:.1%}, "
            f"informed {int(self.informed.sum())}/{self.n}, calls {self.n_calls}"
            + (f", ERRORS {len(self.errors)}" if self.errors else "")
        )


def hybrid_run(
    pop: list[HH_Agent],
    A: np.ndarray,
    design: tuple[str, str, str, str],
    llm: LLMs,
    village: int = VILLAGE,
    rounds: int | None = None,
    seed: int = 0,
    replicate: int = 0,
    qN: float = QN,
    qP: float = QP,
    treatment: int = TREATMENT_ONCE,
    final_sweep: bool = False,
    max_workers: int = 8,
    responder: Callable[..., Response] | None = None,
    progress: tqdm | None = None,
) -> RunResult:
    """One replicate: BCDJ's information model, an LLM's take-up decision.

    The loop, which is `diffusion_model.m`'s with step 1 replaced::

        seed    the leaders are informed by the MFI
        r = 1   step 1  everyone informed before this round and not yet asked
                        is prompted once, concurrently -> (Y)/(N)
                step 2  everyone ever informed draws once per ordered edge at
                        qP if they joined and qN if they did not
        ...
        r = T   the same, and then stop

    Three orderings inside that are the model rather than the code, and are held
    to deliberately:

    **Take-up is one-shot, at the moment of first hearing.** A household that
    declines is never asked again, however many neighbours pitch it afterwards.
    That is `(~contagiousbefore & contagious)` in the Matlab and it is what
    treatment II exists to relax.

    **Hearing and deciding are a period apart.** A household reached during
    round *r* decides at *r+1*. So information moves one hop per round and
    decisions trail it by one, and the households reached in round `T` are never
    asked -- BCDJ's own behaviour, which `final_sweep` optionally departs from.

    **An adopter transmits at qP in the round it adopts**, because step 1
    updates `adopted` before step 2 reads it.

    Two generators, spawned from one seed and never crossed. Transmission and
    the informer draw each consume a fixed number of uniforms per round
    regardless of the run's state, so two arms given the same `seed` walk the
    same random stream even as their adoption histories diverge. Common random
    numbers are the main variance reduction available at S = 20, and they are
    the reason `seed` is a parameter rather than something derived from the
    design or the model.

    Parameters
    ----------
    pop, A
        `population(*build_village(...))` and `adjacency_matrix(pop)`, built once
        and reused across replicates. Nothing in a run mutates either.
    seed
        The replicate's random stream. **Pass the same value to every arm of a
        replicate** -- that is what makes them paired.
    final_sweep
        After round `T`, ask every household that heard something but never got
        its turn. Off by default, because the default should match the Matlab
        and because every swept household is a paid call for a decision BCDJ
        never make -- about 5 a run on village 6. `RunResult.swept` counts them
        and their `Decision.round` is `T + 1`.
    responder
        Injected for tests and dry runs; defaults to `get_response`.
    progress
        `main`'s bar, advanced one step per call and re-labelled each round with
        how far through the village the information has got. A run cannot say up
        front how many calls it will make -- that is the diffusion -- so the bar
        is owned by the caller, which knows the upper bound of one call per
        household per replicate.

    Raises
    ------
    NotImplementedError
        `treatment=2`. See `_decide_per_transmission`.
    RuntimeError
        A round in which every call failed, or a broken state invariant.
    """
    if treatment not in TREATMENTS:
        raise ValueError(f"treatment must be one of {TREATMENTS}, got {treatment}")
    n = len(pop)
    if A.shape != (n, n):
        raise ValueError(f"adjacency is {A.shape} but there are {n} agents")
    rounds = rounds if rounds is not None else default_rounds(village)
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")

    is_leader = np.array([a.is_leader for a in pop], dtype=bool)
    informed = is_leader.copy()
    asked = np.zeros(n, dtype=bool)
    adopted = np.zeros(n, dtype=bool)
    informed_round = np.where(is_leader, SEEDED, NEVER)
    adopted_round = np.full(n, NEVER, dtype=int)
    curve = np.zeros(rounds, dtype=float)
    info_curve = np.zeros(rounds, dtype=float)
    decisions: list[Decision] = []
    hit: np.ndarray | None = None

    # Spawned rather than offset, so the two streams are independent by
    # construction and neither depends on how far the other has run.
    transmit_rng, informer_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))

    if treatment == TREATMENT_PER_TRANSMISSION:
        # Treatment II's loop is not this loop -- decisions there follow each
        # transmission rather than preceding it -- so the divergence starts here
        # rather than inside the round. Raises before anything is spent.
        _decide_per_transmission(pop, informed, adopted, hit, design, llm, 1, informer_rng)

    for r in range(1, rounds + 1):
        deciding = informed & ~asked
        if progress is not None:
            progress.set_postfix_str(f"round {r}/{rounds}, {int(deciding.sum())} asked", refresh=True)
        if deciding.any():
            batch = decide_round(
                pop, deciding, adopted, hit, design, llm, r, informer_rng,
                max_workers=max_workers, responder=responder, progress=progress,
            )
            for d in batch:
                if d.joined:
                    adopted[d.idx] = True
                    adopted_round[d.idx] = r
            decisions.extend(batch)
            asked |= deciding
        else:
            # Nobody new heard last round. The informer generator still advances,
            # so a quiet round costs the same number of draws as a busy one and
            # the streams stay aligned across arms.
            informer_rng.random(n)

        # Step 2 reads `adopted` as step 1 just left it: a household that joined
        # this round already speaks at qP.
        hit = transmit(A, informed, adopted, transmit_rng, qN=qN, qP=qP)
        newly = hit.any(axis=0) & ~informed
        informed_round[newly] = r
        informed |= newly

        curve[r - 1] = adopted.mean()
        info_curve[r - 1] = informed.mean()

    swept = 0
    if final_sweep:
        # Everyone who heard in the last round and so never got a turn. Adds no
        # transmission and no round of information flow, only the decisions the
        # loop already opened.
        deciding = informed & ~asked
        if progress is not None:
            progress.set_postfix_str(f"sweep, {int(deciding.sum())} asked", refresh=True)
        if deciding.any():
            batch = decide_round(
                pop, deciding, adopted, hit, design, llm, rounds + 1, informer_rng,
                max_workers=max_workers, responder=responder, progress=progress,
            )
            for d in batch:
                if d.joined:
                    adopted[d.idx] = True
                    adopted_round[d.idx] = rounds + 1
            decisions.extend(batch)
            asked |= deciding
            swept = int(deciding.sum())
            curve[-1] = adopted.mean()

    if np.any(adopted & ~informed):
        raise RuntimeError("a household adopted without ever being informed")
    if np.any(adopted & ~asked):
        raise RuntimeError("a household adopted without ever being asked")

    return RunResult(
        village=village,
        arm="hybrid",
        design=design_label(*design),
        llm=llm.value,
        treatment=treatment,
        rounds=rounds,
        seed=seed,
        replicate=replicate,
        hh_ids=np.array([a.hh_id for a in pop], dtype=int),
        is_leader=is_leader,
        adopted=adopted,
        informed=informed,
        asked=asked,
        adopted_round=adopted_round,
        informed_round=informed_round,
        curve=curve,
        info_curve=info_curve,
        decisions=decisions,
        swept=swept,
    )


# --------------------------------------------------------------------------
# The BCDJ baseline: `diffusion_model.m` and `moments.m`, in numpy
# --------------------------------------------------------------------------

# BCDJ's `Z = W(:,1:6)` -- the whole of `hhcovariates<V>.csv`, tab separated,
# Stata "." for missing. Column order is theirs and the fit depends on it.
COVARIATE_NAMES = ("rooms", "beds", "electricity", "latrine", "rooms_per_capita", "beds_per_capita")


def covariates(village: int, root: Path | str | None = None, giant_only: bool = True) -> np.ndarray:
    """BCDJ's six household covariates, in `idx` order. `Main_models_1_3.m` §5."""
    root = Path(root) if root is not None else Path(dl.DEFAULT_ROOT)
    path = root / "Matlab Replication" / "India Networks" / f"hhcovariates{village}.csv"
    frame = pd.read_csv(path, sep="\t", header=None, na_values=".")
    Z = frame.iloc[:, :6].to_numpy(dtype=float)
    v = dl.load_village(village, root=root)
    if len(Z) != v.n:
        raise RuntimeError(f"v{village}: {len(Z)} rows in {path.name} but {v.n} households")
    if giant_only and v.in_giant is not None:
        Z = Z[v.in_giant.astype(bool)]
    return np.where(np.isinf(Z), np.nan, Z)


def _irls(X: np.ndarray, y: np.ndarray, max_iter: int = 100, tol: float = 1e-11) -> tuple[np.ndarray, np.ndarray]:
    """Logistic regression by iteratively reweighted least squares -- Matlab's `glmfit`.

    Hand-rolled rather than pulled from statsmodels for one seven-parameter fit,
    the same call `profiler.py` made about Welch's t. IRLS *is* what `glmfit`
    runs, so this is a transliteration and not an alternative estimator.
    Returns the coefficients and their standard errors.
    """
    beta = np.zeros(X.shape[1])
    W = np.ones(len(y))
    for _ in range(max_iter):
        mu = 1.0 / (1.0 + np.exp(-(X @ beta)))
        W = np.clip(mu * (1.0 - mu), 1e-12, None)
        z = X @ beta + (y - mu) / W
        step = np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (W * z))
        if np.max(np.abs(step - beta)) < tol:
            beta = step
            break
        beta = step
    mu = 1.0 / (1.0 + np.exp(-(X @ beta)))
    W = np.clip(mu * (1.0 - mu), 1e-12, None)
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * W[:, None]))))
    return beta, se


def fit_betas(
    villages: list[int] | None = None,
    root: Path | str | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """BCDJ's take-up logit: leaders only, pooled over the 43 analysis villages.

    `Main_models_1_3.m` §6: `glmfit(Covars, Outcome, 'binomial', 'link', 'logit')`
    where `Outcome` is `TakingLeaders` and `Covars` is `ZLeaders`, both stacked
    across villages after pruning to the giant component. Leaders are the one
    population where information is not confounding -- they were told directly --
    which is why beta is identified there and nowhere else.

    Returns `(beta, report)`; `beta` is 7 long, intercept first, and goes
    straight into `logit_p`. Rows with a missing covariate are dropped, as
    `glmfit` drops them, and the report says how many.
    """
    villages = villages if villages is not None else dl.analysis_villages(root or dl.DEFAULT_ROOT)
    rows, outcome = [], []
    for vn in villages:
        v = dl.load_village(vn, root=root if root is not None else dl.DEFAULT_ROOT)
        giant = v.in_giant.astype(bool) if v.in_giant is not None else np.ones(v.n, bool)
        is_leader = v.leader[giant].astype(bool)
        rows.append(covariates(vn, root)[is_leader])
        outcome.append(v.mf[giant][is_leader])
    Z = np.vstack(rows)
    y = np.concatenate(outcome).astype(float)

    keep = ~np.isnan(Z).any(axis=1)
    X = np.column_stack([np.ones(keep.sum()), Z[keep]])
    beta, se = _irls(X, y[keep])
    return beta, {
        "villages": len(villages),
        "leaders": int(len(y)),
        "used": int(keep.sum()),
        "dropped_missing": int((~keep).sum()),
        "adopters": int(y[keep].sum()),
        "take_up": float(y[keep].mean()),
        "names": ("intercept", *COVARIATE_NAMES),
        "se": se,
    }


def logit_p(Z: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """`1./(1+exp(-([ones(N,1) Z]*Betas)))`.

    A household with a missing covariate comes back NaN, and that is BCDJ's
    behaviour rather than a gap: `x < NaN` is false in Matlab as it is in numpy,
    so such a household is asked and never joins. It is left NaN rather than
    imputed so that the count of them is visible.
    """
    return 1.0 / (1.0 + np.exp(-(np.column_stack([np.ones(len(Z)), Z]) @ beta)))


def second_neighbours(A: np.ndarray) -> np.ndarray:
    """`Sec = (X^2>0); Sec(i,i)=0; Sec = (Sec - X > 0)` -- second neighbours only."""
    reach2 = (A.astype(int) @ A.astype(int)) > 0
    np.fill_diagonal(reach2, False)
    return reach2 & ~A


def leader_neighbourhoods(A: np.ndarray, leaders: np.ndarray, adopted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The two exclusive leader neighbourhoods moments 2 and 3 are computed on.

    **Defined by the empirical take-up, not the simulated one, and that is the
    Matlab's behaviour rather than a choice made here.** `moments.m` caches these
    in a `persistent netstats` keyed by village, and `divergence_model.m`
    computes the *empirical* moments first -- so the partition is fixed by the
    data on that first call and every simulated moment afterwards reuses it.
    Only the numerator varies with the simulation. Recomputing it per run would
    be a different statistic, so `adopted` here must always be `v.mf`.

    `neighborOfInfected` is adjacent to an adopting leader and *not* adjacent to
    a non-adopting one; `neighborOfNonInfected` is the reverse. A node that
    neighbours both is in neither, which is what makes the contrast between the
    two a contrast. Distance is read off the adjacency rather than by breadth-
    first search because only "minimum distance == 1" is ever asked for, and an
    adopting leader is at distance 0 from itself so it is excluded from its own
    neighbourhood.
    """
    inf_leaders = leaders & adopted
    non_leaders = leaders & ~adopted
    near_inf = A[:, inf_leaders].any(axis=1) & ~inf_leaders if inf_leaders.any() else np.zeros(len(A), bool)
    near_non = A[:, non_leaders].any(axis=1) & ~non_leaders if non_leaders.any() else np.zeros(len(A), bool)
    return near_inf & ~near_non, near_non & ~near_inf


def moments_v1(
    A: np.ndarray,
    Sec: np.ndarray,
    adopted: np.ndarray,
    near_infected_leader: np.ndarray,
    near_noninfected_leader: np.ndarray,
) -> np.ndarray:
    """BCDJ's five SMM moments, `moments.m` version 1, transliterated.

    1. take-up among households with no adopting neighbour
    2. take-up in the neighbourhood of adopting leaders
    3. take-up in the neighbourhood of non-adopting leaders
    4. covariance of taking with the share of first neighbours taking
    5. covariance of taking with the share of second neighbours taking

    Scoring on the paper's own criterion is the only way to say "this fits
    better than theirs" without picking a metric that flatters us
    (`docs/experiment_design.md` §4.3), and all five are cross-sectional, so they
    survive village 6's flat curve where the timing criterion does not.

    Moment 5 divides the second-neighbour count by the *first*-degree, which is
    what `moments.m` does. It is odd and it is reproduced deliberately: a
    "corrected" denominator would be a different statistic and would not be
    comparable to the published numbers.
    """
    degree = A.sum(axis=1)
    non_hermit = degree > 0
    adopting_neighbours = A @ adopted
    stats = np.zeros(5)

    alone = (adopting_neighbours == 0) & non_hermit
    stats[0] = (alone & adopted).sum() / alone.sum() if alone.any() else 0.0
    stats[1] = (adopted & near_infected_leader).sum() / near_infected_leader.sum() if near_infected_leader.any() else 0.0
    stats[2] = (
        (adopted & near_noninfected_leader).sum() / near_noninfected_leader.sum()
        if near_noninfected_leader.any()
        else 0.0
    )

    takers = adopted[non_hermit]
    deg = degree[non_hermit]
    stats[3] = float((takers * (adopting_neighbours[non_hermit] / deg)).sum() / non_hermit.sum())
    stats[4] = float((takers * ((Sec @ adopted)[non_hermit] / deg)).sum() / non_hermit.sum())
    return stats


def divergence(simulated: np.ndarray, empirical: np.ndarray) -> np.ndarray:
    """`MeanSimulatedMoments - EmpiricalMoments`, per moment (`divergence_model.m`)."""
    return np.asarray(simulated, dtype=float) - np.asarray(empirical, dtype=float)


def criterion(d: np.ndarray) -> float:
    """The SMM objective at identity weights: `d' W d` with `W = eye(m)`.

    `Main_models_1_3.m` §8 defaults to `W = eye(m)` and only builds the two-step
    optimal weight matrix from the 43-village divergence matrix, which is
    baseline #2's job rather than this one's. On one village at identity weights
    the criterion is a plain sum of squares, and it is reported as such.
    """
    d = np.asarray(d, dtype=float)
    return float(d @ d)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank AUC, ties averaged. NaN if either class is empty."""
    scores, labels = np.asarray(scores, float), np.asarray(labels).astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average the ranks of tied scores, which is what makes a constant predictor 0.5
    for value in np.unique(scores):
        tie = scores == value
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bcdj_run(
    pop: list[HH_Agent],
    A: np.ndarray,
    p: np.ndarray,
    village: int = VILLAGE,
    rounds: int | None = None,
    seed: int = 0,
    replicate: int = 0,
    qN: float = QN,
    qP: float = QP,
) -> RunResult:
    """`diffusion_model.m`, unchanged. The thing the hybrid substitutes into.

    The same loop as `hybrid_run` with step 1 restored to BCDJ's: instead of an
    LLM call, a newly-informed household joins if `x[i, t] < p[i]`, where `x` is
    drawn once up front as an `(n, T)` matrix exactly as `rand(N,T)` is.

    Its transmission stream is spawned to the same position as the hybrid's, so
    `bcdj_run(seed=s)` and `hybrid_run(seed=s)` see identical edge draws given
    identical adoption histories -- the baseline is paired with the arms it is
    compared against, not merely run beside them.
    """
    n = len(pop)
    rounds = rounds if rounds is not None else default_rounds(village)
    if p.shape != (n,):
        raise ValueError(f"p must be ({n},), got {p.shape}")

    is_leader = np.array([a.is_leader for a in pop], dtype=bool)
    informed = is_leader.copy()
    asked = np.zeros(n, dtype=bool)
    adopted = np.zeros(n, dtype=bool)
    informed_round = np.where(is_leader, SEEDED, NEVER)
    adopted_round = np.full(n, NEVER, dtype=int)
    curve, info_curve = np.zeros(rounds), np.zeros(rounds)

    # Child 0 is transmission in both arms; child 1 is the hybrid's informer draw
    # and is skipped here so the streams stay aligned across the two.
    children = np.random.SeedSequence(seed).spawn(3)
    transmit_rng = np.random.default_rng(children[0])
    x = np.random.default_rng(children[2]).random((n, rounds))

    for r in range(1, rounds + 1):
        deciding = informed & ~asked
        joined = deciding & (x[:, r - 1] < p)  # NaN p never joins, as in Matlab
        adopted |= joined
        adopted_round[joined] = r
        asked |= deciding

        hit = transmit(A, informed, adopted, transmit_rng, qN=qN, qP=qP)
        newly = hit.any(axis=0) & ~informed
        informed_round[newly] = r
        informed |= newly
        curve[r - 1] = adopted.mean()
        info_curve[r - 1] = informed.mean()

    return RunResult(
        village=village,
        arm="bcdj",
        design="",
        llm="",
        treatment=TREATMENT_ONCE,
        rounds=rounds,
        seed=seed,
        replicate=replicate,
        hh_ids=np.array([a.hh_id for a in pop], dtype=int),
        is_leader=is_leader,
        adopted=adopted,
        informed=informed,
        asked=asked,
        adopted_round=adopted_round,
        informed_round=informed_round,
        curve=curve,
        info_curve=info_curve,
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

# One CSV per (village, design, model, replicate), which is the same key the
# pilot's logs are named by minus the replicate. A run is never appended to: a
# paid replicate is a fact about one seed and one design, and two of them in one
# file could not be told apart afterwards.
OUTPUT_DIR = Path("output/hybrid")

_LABEL = re.compile(r"^A(?P<a>\d)B(?P<b>\d)C(?P<c>\d)D(?P<d>\d)$", re.IGNORECASE)


def parse_design(label: str) -> tuple[str, str, str, str]:
    """`A1B0C1D1` -> the four levels `hybrid_run` takes.

    The inverse of `design_label`, so a design named in the pilot's tables can be
    handed to a run as the string it is named by rather than re-typed as four
    levels in the right order.
    """
    match = _LABEL.match(label.strip())
    if not match:
        raise ValueError(f"not a design label: {label!r}; expected the pilot's form, e.g. A1B0C1D1")
    axes = (PROFILE_LEVELS, INFORMER_LEVELS, ENDORSEMENT_LEVELS, INSTRUCTION_LEVELS)
    levels = []
    for key, axis in zip("abcd", axes):
        digit = int(match[key])
        if digit >= len(axis):
            raise ValueError(f"{label}: axis {key.upper()} has {len(axis)} levels, so {key.upper()}{digit} is not one")
        levels.append(axis[digit])
    return tuple(levels)  # type: ignore[return-value]


def run_path(
    village: int,
    design: str,
    llm: LLMs,
    replicate: int,
    output_dir: Path | str = OUTPUT_DIR,
) -> Path:
    """`output/hybrid/gpt_5_4_nano/A1B0C1D1/v6_rep0.csv`.

    One subfolder per model and, under it, one per design -- both are axes a
    reader sorts runs by before ever looking at a replicate, and a directory
    does that without parsing the filename.
    """
    return Path(output_dir) / llm.name.lower() / design / f"v{village}_rep{replicate}.csv"


def write_decisions(result: RunResult, path: Path) -> Path:
    """The run's log, one row per decision, in `Decision.to_row`'s columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([d.to_row() for d in result.decisions]).to_csv(path, index=False)
    return path


def _stub_responder(llm: LLMs, prompt: str, instruction: str = "") -> Response:
    """A dry run's answer: a coin keyed on the prompt, and no API call.

    Deterministic, so a dry run is repeatable and two dry runs of the same design
    give the same histories -- but the decisions are *not* the model's, and the
    adoption curve a dry run reports is therefore meaningless. What it is for is
    the mechanics: that the village builds, that every prompt renders, and how
    many calls the live run would pay for. A coin rather than all-yes because
    the call count depends on the qP/qN split and an all-adopting village would
    over-report it.
    """
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    decision = YES_TOKEN if digest[0] < 128 else NO_TOKEN
    return Response(text="DRY RUN -- no call was made.", decision=decision, usage={}, attempts=0)


def ground_truth_rates(
    village: int,
    giant_only: bool = True,
    root: Path | str | None = None,
) -> dict[str, float | int]:
    """Human take-up for a village, on the same denominator a run sees.

    Mirrors `build_village`'s `giant_only` pruning so the printed comparison and
    the simulated one share a population. Reads straight from `dl.load_village`
    rather than through `FEATURE_COLUMNS` -- this never touches an agent or a
    prompt, only the terminal summary.
    """
    v = dl.load_village(village, root=root if root is not None else dl.DEFAULT_ROOT)
    keep = v.in_giant.astype(bool) if (giant_only and v.in_giant is not None) else np.ones(v.n, dtype=bool)
    leader = v.leader.astype(bool) & keep
    non_leader = keep & ~v.leader.astype(bool)
    return {
        "n": int(keep.sum()),
        "n_leaders": int(leader.sum()),
        "n_non_leaders": int(non_leader.sum()),
        "all": float(v.mf[keep].mean()) if keep.any() else float("nan"),
        "leaders": float(v.mf[leader].mean()) if leader.any() else float("nan"),
        "non_leaders": float(v.mf[non_leader].mean()) if non_leader.any() else float("nan"),
    }


def main(argv: list[str] | None = None) -> int:
    import sys

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--design", nargs="+", required=True, help="one or more pilot labels, e.g. A1B0C1D1")
    p.add_argument("--village", type=int, default=VILLAGE)
    p.add_argument("--model", default=LLMs.GPT_5_4_NANO.value, choices=[m.value for m in LLMs])
    p.add_argument("--reps", type=int, default=1, help="replicates per design (default: 1)")
    p.add_argument("--first-rep", type=int, default=0, help="index of the first replicate (default: 0)")
    p.add_argument("--seed", type=int, default=0, help="base seed; replicate r runs at seed + r, in every design")
    p.add_argument("--rounds", type=int, default=None, help="default: the village's last trimester in panel.dta")
    p.add_argument("--qn", type=float, default=QN)
    p.add_argument("--qp", type=float, default=QP)
    p.add_argument("--sweep", action="store_true", help="after T, ask everyone informed but never asked")
    p.add_argument("--keep-isolates", action="store_true", help="do not prune to the giant component")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--features", type=Path, default=None)
    p.add_argument("--profiles", type=Path, default=None)
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--overwrite", action="store_true", help="replace a CSV that is already there")
    p.add_argument("--live", action="store_true", help="actually call the API (default: dry run, no calls, no cost)")
    a = p.parse_args(argv)

    llm = get_llm(a.model)
    try:
        designs = [(label.upper(), parse_design(label)) for label in a.design]
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    # Checked before anything is built, so a run that would refuse to write on
    # its last replicate refuses now instead of after paying for the first.
    if a.live and not a.overwrite:
        clashes = [
            run_path(a.village, label, llm, r, a.output_dir)
            for label, _ in designs
            for r in range(a.first_rep, a.first_rep + a.reps)
            if run_path(a.village, label, llm, r, a.output_dir).exists()
        ]
        if clashes:
            print(f"error: {len(clashes)} log(s) already exist, e.g. {clashes[0]}. "
                  "Use --first-rep to run further replicates, or --overwrite.", file=sys.stderr)
            return 1

    leaders, households = build_village(
        a.village, features_path=a.features, profiles_path=a.profiles,
        root=a.root, giant_only=not a.keep_isolates,
    )
    pop = population(leaders, households)
    A = adjacency_matrix(pop)

    missing = missing_narratives(pop)
    if missing and any("NARRATIVE" in design for _, design in designs):
        print(f"error: {len(missing)} household(s) have no narrative profile (first few: {missing[:5]}), "
              "and a NARRATIVE design needs one for every household it renders.", file=sys.stderr)
        return 1

    rounds = a.rounds if a.rounds is not None else default_rounds(a.village, root=a.root)
    if not a.live:
        print("DRY RUN -- no API calls made, nothing written. Add --live to run for real.\n")
    print(f"v{a.village}: {len(pop)} households ({len(leaders)} leaders), T = {rounds}, model {llm.value}")
    print(f"{len(designs)} design(s) x {a.reps} replicate(s) = at most {len(designs) * a.reps * len(pop):,} calls")
    gt = ground_truth_rates(a.village, giant_only=not a.keep_isolates, root=a.root)
    print(
        f"ground truth: joined {gt['all']:.1%}, leaders {gt['leaders']:.1%} ({gt['n_leaders']}), "
        f"non-leaders {gt['non_leaders']:.1%} ({gt['n_non_leaders']})\n"
    )

    # One call per household per replicate is the ceiling, not the count: a run
    # asks only the households the information reaches, and by round T it has
    # usually reached most but never all of them. So the bar is scaled to the
    # ceiling and re-synced when each replicate ends, which is the same bargain
    # the pilot's bar makes -- the fraction done understates, and the estimate of
    # what is left stays honest across the replicates still to come.
    runs = [(label, design, r) for label, design in designs for r in range(a.first_rep, a.first_rep + a.reps)]
    bar = tqdm(total=len(runs) * len(pop), unit="call", desc="hybrid", dynamic_ncols=True)
    done = 0  # calls the replicates so far were budgeted
    try:
        for index, (label, design, r) in enumerate(runs, start=1):
            bar.set_description(f"v{a.village} {label} rep {r} [{index}/{len(runs)}]")
            done += len(pop)
            try:
                result = hybrid_run(
                    pop, A, design, llm,
                    village=a.village, rounds=a.rounds, seed=a.seed + r, replicate=r,
                    qN=a.qn, qP=a.qp, final_sweep=a.sweep, max_workers=a.workers,
                    responder=None if a.live else _stub_responder,
                    progress=bar,
                )
            finally:
                # Re-sync rather than update: the replicate advanced the bar once
                # per call it actually made, which is fewer than it was budgeted
                # -- and fewer still if it raised part-way through.
                bar.set_postfix_str("")
                bar.update(max(0, done - bar.n))
            # tqdm.write rather than print, so a summary landing between rounds
            # does not cut through the bar.
            tqdm.write(result.summary())
            if result.parse_failures:
                tqdm.write(f"  {len(result.parse_failures)} unparseable answer(s)")
            if a.live:
                path = write_decisions(result, run_path(a.village, label, llm, r, a.output_dir))
                usage = result.usage()
                tqdm.write(f"  -> {path}" + (f"  ({usage.get('total_tokens', 0):,} tokens)" if usage else ""))
    finally:
        bar.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
