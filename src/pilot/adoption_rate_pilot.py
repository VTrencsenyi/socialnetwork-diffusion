from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import sys
from enum import Enum
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # same as plots.py: figures are written, not shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from tqdm.auto import tqdm

try:
    from ..llm import load_client, one_call
except ImportError:  # running as a script, not a package
    # `python src/pilot/adoption_rate_pilot.py` puts src/pilot on sys.path, not src,
    # so the sibling module is only importable once src itself is on it.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from llm import load_client, one_call  # type: ignore[no-redef]


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

# The three states of nature, in the order they are asked for and reported.
DT_STATES = ("beneficial", "limited_effect", "harmful")

# One state's row of the decision matrix.
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

# `states` is keyed by state name rather than a list of state objects, because
# that is the one shape strict mode can guarantee is complete: `minItems` and
# `maxItems` are rejected keywords, so an array could come back with two states
# or with `harmful` twice, whereas an object with three required properties and
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

# What the request carries under `text` for a DT design. Not logged per row: the
# design label is D2 or it is not, and the schema is the same for all of them.
DT_FORMAT = {"format": {"type": "json_schema", "name": "dt_analysis", "strict": True, "schema": DT_SCHEMA}}

# How far the three elicited probabilities may miss 1.0 before the distribution is
# reported as invalid. Strict mode cannot express the constraint -- `minimum`,
# `maximum` and arithmetic are all outside JSON Schema's strict subset -- so this
# is checked on the way back rather than enforced on the way out.
DT_PROBABILITY_TOLERANCE = 0.02


class LLMs(Enum):
    GPT_5_4_NANO = "gpt-5.4-nano"
    HAIKU_4_5 = "claude-haiku-4-5-20251001"
    GROK_4_2 = "grok-4.20-0309-non-reasoning"

VILLAGE = 6
# They are from the same subcaste and are non leaders, so in-group and influence factors are mitigated here.
SAMPLE_HH_ADOPT_SELF = 6026
SAMPLE_HH_ADOPT_OTHER = 6032 
SAMPLE_HH_NON_ADOPT_SELF = 6039
SAMPLE_HH_NON_ADOPT_OTHER = 6099

# The CLEANED table and the profiles built from it: both carry the merged subcaste
# spelling, so two households of one subcaste are named the same way in the fields
# and in the narrative. The raw table's NAYAKARU / NAYAKA / NAIKA would read as
# three different castes and undo the in-group control the samples were picked for.
FEATURES_PATH = Path(f"output/features/CLEANED_hh_features_{VILLAGE}.csv")
PROFILES_PATH = Path(f"output/profiles/profiles_{VILLAGE}.json")

# What the model must end its response with -- the same tokens elicit.parse_answer reads.
YES_TOKEN = "(Y)"
NO_TOKEN = "(N)"

# The third value the decision column can take. A response with no decision in it is
# still a response worth keeping: the rate of these is a fact about the prompt design,
# and the text is the only way to find out what the model said instead.
PARSING_ERROR = "PARSING ERROR"

# The one call parameter we set. Everything else -- temperature included -- stays at
# whatever each provider ships, so the three models are compared under their own
# defaults rather than under a number picked for one of them.
#
# One number for all 54 designs rather than one per instruction: a cap that moved
# with the design would be a second thing changing between arms. 1024 because DT
# is what sets the floor -- its first live call spent 477 tokens on the *base*
# case, the shortest prompt in the grid, and a reasoning model's thinking comes
# out of the same budget as its answer. The plain and MOA designs never came near
# 512 (their maxima over 700 calls each were 245 and 391), so raising it changes
# what they can do, not what they did.
MAX_OUTPUT_TOKENS = 1024

# Which block of keys.json each model is reached through. All three speak the OpenAI
# Responses API; claude and grok differ only by base_url.
PROVIDERS = {
    LLMs.GPT_5_4_NANO: "openai",
    LLMs.HAIKU_4_5: "claude",
    LLMs.GROK_4_2: "grok",
}

# profiler.py's vocabulary, so the pilot describes a household in the same words
# the main pipeline does.
RELIGION_LABELS = {"hinduism": "Hindu", "islam": "Muslim", "christianity": "Christian"}
ELECTRICITY_LABELS = {0: "none", 1: "yes, a government connection", 2: "yes, a private connection"}


def get_llm(label: str) -> LLMs:
    try:
        return LLMs(label)
    except ValueError:
        raise ValueError(f"Invalid LLM label: {label}. Valid labels are: {[llm.value for llm in LLMs]}")

@lru_cache(maxsize=None)
def _client(provider: str):
    """One client per keys.json block, reused across calls."""
    return load_client(provider)


def get_response(llm: LLMs, prompt: str, instruction: str = "") -> tuple[str, str, dict[str, int]]:
    """One call to one model: the text, the decision in it, and what it cost.

    Three of the CSV's four columns, so that a row is one call to this function.
    A response with no decision in it comes back as `PARSING_ERROR` rather than an
    exception, so the text still reaches the log -- it is the only record of what
    the model said instead, and losing it to a traceback would lose the evidence.
    Transient API errors are retried, and anything left raises.

    `instruction` is the design's D axis, and the only one of the four that changes
    the request rather than the prompt: a DT design asks for its decision matrix
    under a schema, and reads the decision out of that matrix rather than out of
    the prose.
    """
    if llm is not LLMs.GPT_5_4_NANO:
        raise NotImplementedError(
            f"{llm.value} is not wired up yet -- only {LLMs.GPT_5_4_NANO.value} answers so far."
        )
    request: dict[str, object] = {"model": llm.value, "input": prompt}
    if instruction == "DT":
        # The provider seam. `text.format` is the OpenAI Responses API's strict
        # structured output, enforced server-side -- which is what makes reading
        # the matrix back a `json.loads` rather than a lenient search through
        # prose. The other two models are reached through the same client with
        # `base_url` pointed elsewhere and neither is guaranteed to accept this
        # key; wiring them up means deciding then whether they get the schema or
        # a prompt-only JSON request read leniently, and that decision needs
        # their behaviour in front of it rather than a fallback written blind.
        request["text"] = DT_FORMAT
    text, usage = one_call(_client(PROVIDERS[llm]), request, max_output_tokens=MAX_OUTPUT_TOKENS)
    if instruction == "DT":
        payload = parse_dt(text)
        decision = PARSING_ERROR if payload is None else (YES_TOKEN if payload["decision"] == "Y" else NO_TOKEN)
    else:
        try:
            decision = extract_decision(text)
        except ValueError:
            decision = PARSING_ERROR
    return text, decision, usage

# One equivalence class per decision. Models answer the same instruction in a
# dozen dialects -- bare or bracketed, the letter or the word, wrapped in the
# markdown they were reasoning in, or the {Y} the template itself is written with.
_YES = {"y", "yes"}
_NO = {"n", "no"}

# Everything a token may arrive dressed in: the brackets of any kind, markdown
# emphasis, quotes, and whatever punctuation ends the sentence.
_DECORATION = "()[]{}<>*_`~\"'“”‘’.,!?:;- \t"

# A bracketed token anywhere in the response -- the one shape unambiguous enough
# to read out of the middle of a sentence.
_BRACKETED = re.compile(r"[(\[{<]\s*(y|n|yes|no)\s*[)\]}>]", re.IGNORECASE)


def _as_decision(word: str) -> str | None:
    """`(Y)`, `**y**`, `{Y}`, `"Yes."` -> YES_TOKEN. Anything else -> None."""
    word = word.strip(_DECORATION).lower()
    if word in _YES:
        return YES_TOKEN
    if word in _NO:
        return NO_TOKEN
    return None


def extract_decision(response: str) -> str:
    """Extract the decision token from the model's response.

    Returns the token itself, or raises a ValueError if neither is found.

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
    probability outside [0, 1] or a set of three that does not sum to 1 makes the
    distribution invalid, but the decision is still the model's answer to the same
    question every other design asks, and dropping it out of the adoption rate
    would bias that rate by a property of the analysis rather than of the answer.
    Those responses come back parsed, and `dt_frame` flags them.
    """
    text = (response or "").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # A truncated response is the expected way to land here, and a prefix of an
        # object is not recoverable. Anything else is prose around the object.
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
            values = [float(block[field]) for field in ("probability", "Y_utility", "N_utility")]
        except (KeyError, TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in values):
            return None
    return payload


@lru_cache(maxsize=1)
def _features() -> pd.DataFrame:
    return pd.read_csv(FEATURES_PATH).set_index("hhid")


@lru_cache(maxsize=1)
def _narratives() -> dict[int, str]:
    records = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
    return {int(hhid): (rec.get("narrative_profile") or "").strip() for hhid, rec in records.items()}


def get_household(hhid: int) -> dict[str, str]:
    """One household's values for every `{placeholder}` the templates use.

    One dict serves all four -- str.format ignores the keys a template does not
    read, so the demographic block simply never asks for the narrative.
    """
    row = _features().loc[hhid]
    fields = {
        "religion": RELIGION_LABELS.get(str(row["religion"]).lower(), str(row["religion"]).title()),
        "caste": str(row["subcaste"]).strip().title(),
        "hh_size": f"{int(row['capita'])}",
        "num_rooms": f"{int(row['rooms'])}",
        "num_beds": f"{int(row['beds'])}",
        "electricity": ELECTRICITY_LABELS[int(row["electricity"])],
        "latrine": "yes" if int(row["own_latrine"]) else "no",
        "savings_group": "yes" if bool(row["has_shg"]) else "no",
        "bank_account": "yes" if bool(row["has_savings"]) else "no",
        "occupation": str(row["occupation_head"]).strip().lower(),
        "narrative": _narratives().get(hhid, ""),
    }
    missing = [name for name, value in fields.items() if not value or value == "nan"]
    if missing:
        raise ValueError(f"household {hhid} has no value for {missing}; pick a sample without 'not known' fields")
    return fields


def has_adopted(hhid: int) -> bool:
    return bool(int(_features().loc[hhid, "_adopted"]))


def get_prompt(
    profile_enhancement: str = "",
    informer_enhancement: str = "",
    endorsement_enhancement: str = "",
    instruction: str = "",
    hhid: int | None = None,
    informer_hhid: int | None = None,
) -> str:
    base_context = BASE_CONTEXT

    assert profile_enhancement in ["", "DEMOGRAPHIC", "NARRATIVE"], "Invalid profile enhancement option"
    assert informer_enhancement in ["", "DEMOGRAPHIC", "NARRATIVE"], "Invalid informer enhancement option"
    assert endorsement_enhancement in ["", "ENDORSEMENT"], "Invalid endorsement enhancement option"
    assert instruction in ["", "MOA", "DT"], "Invalid instruction option"

    prompt_parts = [base_context]

    # Axis A: the deciding household describes itself.
    if profile_enhancement:
        if hhid is None:
            raise ValueError("profile_enhancement needs an hhid: there is no household to describe")
        template = DEMOGRAPHIC_ENHANCEMENT if profile_enhancement == "DEMOGRAPHIC" else NARRATIVE_ENHANCEMENT
        prompt_parts.append(template.format(**get_household(hhid)))

    # Axes B and C both speak about the neighbour, so the informer line is added
    # once for either, and the endorsement follows that household's own status:
    # the adopter neighbour joined, the non-adopter one did not.
    if informer_enhancement or endorsement_enhancement:
        if informer_hhid is None:
            raise ValueError("informer_enhancement/endorsement_enhancement needs an informer_hhid")
        prompt_parts.append(INFORMER)
        if informer_enhancement:
            template = INFORMER_PROFILE if informer_enhancement == "DEMOGRAPHIC" else INFORMER_NARRATIVE
            prompt_parts.append(template.format(**get_household(informer_hhid)))
        if endorsement_enhancement:
            prompt_parts.append(JOINER if has_adopted(informer_hhid) else NON_JOINER)

    # Axis D: the MOA and DT instructions replace the plain one rather than adding
    # to it, so exactly one of the three ends every prompt. MOA carries its own
    # copy of the format instruction; DT needs none, because the shape of its
    # answer is enforced by the schema on the request rather than asked for here.
    # That replacement is also what keeps MOA and DT from ever running together:
    # one instruction ends the prompt, so the two are alternatives by construction.
    template = {"": FORMAT_INSTRUCTION, "MOA": MOA_INSTRUCTION, "DT": DT_INSTRUCTION}[instruction]
    prompt_parts.append(template.format(Y=YES_TOKEN, N=NO_TOKEN))

    # Filter out empty strings and join the parts with two newlines
    return "\n\n".join(part.strip() for part in prompt_parts if part.strip())

OUTPUT_DIR = Path("output/pilot")

CSV_COLUMNS = (
    "repetition",
    "sample",
    "ego_hhid",
    "informer_hhid",
    "prompt",
    "response",
    "decision",
    "input_tokens",
    "output_tokens",
    "total_tokens",
)


def design_label(
    profile_enhancement: str = "",
    informer_enhancement: str = "",
    endorsement_enhancement: str = "",
    instruction: str = "",
) -> str:
    """`A1B0C1D0` -- one digit per axis: ego profile, informer profile, endorsement, instruction.

    The digit is the level's index in that axis's tuple, so `D2` was added to the
    end of the instruction axis rather than anywhere in it: every label written
    before DT existed still names the design it named then.
    """
    levels = {"": 0, "DEMOGRAPHIC": 1, "NARRATIVE": 2}
    instructions = {"": 0, "MOA": 1, "DT": 2}
    return (
        f"A{levels[profile_enhancement]}"
        f"B{levels[informer_enhancement]}"
        f"C{int(bool(endorsement_enhancement))}"
        f"D{instructions[instruction]}"
    )


def log_path(llm: LLMs, label: str) -> Path:
    return OUTPUT_DIR / f"{llm.name.lower()}_{label}.csv"


def design_samples(
    profile_enhancement: str = "",
    informer_enhancement: str = "",
    endorsement_enhancement: str = "",
) -> dict[str, tuple[int | None, int | None]]:
    """Which (ego, informer) households a design is run over, per sample arm.

    Every axis but the base case renders differently for an adopter household than
    for a non-adopter one, so those designs have two arms. The base case reads the
    same either way, so it has one, `none` -- which is what makes it the control.
    The instruction axis is not asked for: MOA and DT change how the question is
    put, not which household it is put to.
    """
    if not (profile_enhancement or informer_enhancement or endorsement_enhancement):
        return {"none": (None, None)}
    return {
        "adopter": (SAMPLE_HH_ADOPT_SELF, SAMPLE_HH_ADOPT_OTHER),
        "non_adopter": (SAMPLE_HH_NON_ADOPT_SELF, SAMPLE_HH_NON_ADOPT_OTHER),
    }


def _next_repetition(path: Path, sample: str) -> int:
    """Where this sample's numbering resumes, so a second pass stacks rather than restarts."""
    if not path.is_file():
        return 0
    done = pd.read_csv(path)
    done = done[done["sample"].astype(str) == sample]
    return 0 if done.empty else int(done["repetition"].max()) + 1


def _append_row(path: Path, row: dict[str, object]) -> None:
    """One row, written as it happens: an interrupted run keeps everything before it."""
    is_new = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def run_agent_with_config(
    llm: LLMs,
    profile_enhancement: str = "",
    informer_enhancement: str = "",
    endorsement_enhancement: str = "",
    instruction: str = "",
    reps: int = 1,
    progress: tqdm | None = None,
) -> Path:
    """One prompt design, `reps` times per sample, appended to its own CSV.

    Every axis but the base case renders differently for an adopter household than
    for a non-adopter one, so a design is run twice -- once per sample -- into one
    file, with the `sample` column telling them apart. That is the comparison the
    study is after, so the two arms belong in the same place. The base case reads
    the same either way and is run once, as `none`.

    Repetitions stack: an existing file is appended to and the numbering picks up
    where each sample left off, so calling this again asks for `reps` more.

    `progress` is `run_pilot`'s bar, advanced one step per API call. It is optional
    because a single design run from a notebook has nothing to advance.
    """
    samples = design_samples(profile_enhancement, informer_enhancement, endorsement_enhancement)

    path = log_path(llm, design_label(profile_enhancement, informer_enhancement, endorsement_enhancement, instruction))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for sample, (ego_hhid, informer_hhid) in samples.items():
        # Only the households the prompt actually names get logged, so the columns
        # say which record the model saw rather than which one the sample stands for.
        ego_hhid = ego_hhid if profile_enhancement else None
        informer_hhid = informer_hhid if (informer_enhancement or endorsement_enhancement) else None
        prompt = get_prompt(
            profile_enhancement,
            informer_enhancement,
            endorsement_enhancement,
            instruction,
            ego_hhid,
            informer_hhid,
        )
        first = _next_repetition(path, sample)
        for repetition in range(first, first + reps):
            response, decision, usage = get_response(llm, prompt, instruction)
            _append_row(
                path,
                {
                    "repetition": repetition,
                    "sample": sample,
                    "ego_hhid": ego_hhid if ego_hhid is not None else "",
                    "informer_hhid": informer_hhid if informer_hhid is not None else "",
                    "prompt": prompt,
                    "response": response,
                    "decision": decision,
                    "input_tokens": usage.get("input_tokens", ""),
                    "output_tokens": usage.get("output_tokens", ""),
                    "total_tokens": usage.get("total_tokens", ""),
                },
            )
            if progress is not None:
                progress.update(1)
        # tqdm.write rather than print, so a line landing mid-call does not cut
        # through the bar. With no bar open it is just a write to stdout.
        tqdm.write(f"{path.name}: {sample} repetitions {first}-{first + reps - 1}")

    return path


# The levels of each axis, in the order `design_label` numbers them.
PROFILE_LEVELS = ("", "DEMOGRAPHIC", "NARRATIVE")
INFORMER_LEVELS = ("", "DEMOGRAPHIC", "NARRATIVE")
ENDORSEMENT_LEVELS = ("", "ENDORSEMENT")
# DT is a level of the instruction axis rather than an axis of its own, because it
# and MOA cannot both be in effect: one instruction ends the prompt. As a fifth
# axis the two would form a cell that had to be excluded from the grid everywhere
# it is generated, counted, resolved or rendered; as one axis of three levels the
# combination cannot be expressed in the first place.
INSTRUCTION_LEVELS = ("", "MOA", "DT")


def all_designs() -> list[tuple[str, str, str, str]]:
    """The full factorial, 3 x 3 x 2 x 3 = 54 designs, base case first."""
    return list(itertools.product(PROFILE_LEVELS, INFORMER_LEVELS, ENDORSEMENT_LEVELS, INSTRUCTION_LEVELS))


def planned_calls(llms: list[LLMs], designs: list[tuple[str, str, str, str]], reps: int) -> int:
    """How many API calls a `run_pilot` with these arguments will make.

    Not simply designs x reps: every design but the base case is run once per
    sample, and a base case only once.
    """
    per_model = sum(1 if not (a or b or c) else 2 for a, b, c, _ in designs)
    return per_model * reps * len(llms)


def run_pilot(
    llms: list[LLMs] | None = None,
    designs: list[tuple[str, str, str, str]] | None = None,
    reps: int = 1,
) -> list[Path]:
    """Every design on every model, `reps` times each, one CSV per pair.

    Defaults to the whole study: all three models over the full 54-design grid.
    Repetitions stack the same way they do for a single config, so re-running this
    adds `reps` more of everything rather than starting over.

    A model that is not wired up yet is reported and skipped. A design that fails
    outright is reported, and the run carries on to the next one -- an afternoon of
    calls should not be lost to a single bad configuration. Both are listed again
    at the end, so nothing that went wrong is only visible in scrollback.

    The bar counts API calls rather than designs, because that is the unit the wait
    is made of: a design is one or two calls depending on whether it renders per
    sample, so 54 of them are not 54 equal steps. Each design's calls are counted
    whatever happens to it -- a failure or a skipped model advances the bar by what
    that design would have cost, so the remaining estimate stays honest.
    """
    llms = list(llms) if llms is not None else list(LLMs)
    designs = list(designs) if designs is not None else all_designs()

    total = planned_calls(llms, designs, reps)
    print(f"{len(llms)} model(s) x {len(designs)} design(s) x {reps} rep(s) = {total} calls")

    written: list[Path] = []
    skipped: list[str] = []
    failed: list[tuple[str, str, Exception]] = []

    bar = tqdm(total=total, unit="call", desc="pilot", dynamic_ncols=True)
    done = 0  # calls the designs so far should have accounted for
    try:
        for llm in llms:
            for index, design in enumerate(designs, start=1):
                label = design_label(*design)
                bar.set_description(f"{llm.name.lower()} {label} [{index}/{len(designs)}]")
                done += planned_calls([llm], [design], reps)
                try:
                    written.append(run_agent_with_config(llm, *design, reps=reps, progress=bar))
                except NotImplementedError as exc:
                    # Every remaining design would fail on the same missing provider.
                    tqdm.write(f"  skipping {llm.name.lower()}: {exc}")
                    skipped.append(llm.name.lower())
                    done += planned_calls([llm], designs[index:], reps)
                    break
                except Exception as exc:  # noqa: BLE001 -- one bad design must not end the run
                    tqdm.write(f"  failed: {type(exc).__name__}: {exc}")
                    failed.append((llm.name.lower(), label, exc))
                finally:
                    # Re-sync rather than update: a design that raised part-way
                    # through its calls has advanced the bar by less than it cost.
                    bar.update(max(0, done - bar.n))
    finally:
        bar.close()

    print(f"\n{len(written)} configuration file(s) under {OUTPUT_DIR}")
    for name in skipped:
        print(f"  skipped: {name} is not wired up yet")
    for name, label, exc in failed:
        print(f"  failed:  {name} {label}: {type(exc).__name__}: {exc}")

    return written


# --------------------------------------------------------------------------
# Reading the logs back, and the adoption-rate plot
# --------------------------------------------------------------------------

FIGURE_DIR = Path("figures/pilot")

# plots.py's light surface and categorical slots, so a pilot figure sits next to
# the network ones without a second palette.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
HAIRLINE = "#e1e0d9"

SAMPLE_COLOURS = {"adopter": "#eb6834", "non_adopter": "#2a78d6", "none": "#c3c2b7"}
# plots.py's categorical slot 8, the one red in the palette, kept for the one thing
# on these figures that is a warning rather than a category: an inverted separation.
WARNING = "#e34948"
SAMPLE_LABELS = {"adopter": "adopter sample", "non_adopter": "non-adopter sample", "none": "base case"}
SAMPLE_ORDER = ("adopter", "non_adopter", "none")


# What a log file is called: `<model>_<design label>.csv`. Matched rather than
# globbed for, because anything else written into the same directory -- a saved test
# table, a spreadsheet -- would otherwise be read back as a model and a design.
_LOG_STEM = re.compile(r"^(?P<model>.+)_(?P<design>A\dB\dC\dD\d)$")


def load_results(llm: LLMs | None = None, output_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    """Every logged row, with `llm` and `design` recovered from the filenames.

    The run writes one CSV per (model, design) and the filename is the only place
    those two live, so reading them back means putting them into columns.
    """
    pattern = f"{llm.name.lower()}_*.csv" if llm is not None else "*.csv"
    frames = []
    for path in sorted(output_dir.glob(pattern)):
        # The model name has its own underscores (gpt_5_4_nano); the design label
        # never does, which is what makes the split unambiguous.
        match = _LOG_STEM.match(path.stem)
        if match is None:
            continue
        frame = pd.read_csv(path)
        frame["llm"] = match["model"]
        frame["design"] = match["design"]
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No pilot logs matching {pattern} under {output_dir}")
    return pd.concat(frames, ignore_index=True)


def adoption_rates(results: pd.DataFrame) -> pd.DataFrame:
    """One row per (llm, design, sample): the share of (Y) answers and its SE.

    The rate is over *answered* repetitions -- a PARSING ERROR is not a refusal, so
    counting it as one would drag the rate down for whichever designs the model
    happens to answer untidily. They are counted separately instead, in `unparsed`,
    and the plot marks any design that has them.

    The error bar is the standard error of a proportion, sqrt(p(1-p)/n): with ten
    binary repetitions that is what "how firm is this rate" means. It is zero when
    every repetition agreed, which for a decisive model is a real result rather
    than a missing bar.
    """
    rows = []
    for (llm, design, sample), group in results.groupby(["llm", "design", "sample"], sort=False):
        answered = group[group["decision"] != PARSING_ERROR]
        n = len(answered)
        adopted = int((answered["decision"] == YES_TOKEN).sum())
        rate = adopted / n if n else float("nan")
        rows.append(
            {
                "llm": llm,
                "design": design,
                "sample": sample,
                "n": n,
                "adopted": adopted,
                "rate": rate,
                "se": math.sqrt(rate * (1.0 - rate) / n) if n else float("nan"),
                "unparsed": len(group) - n,
            }
        )
    table = pd.DataFrame(rows)
    return table.sort_values(["llm", "design", "sample"], ignore_index=True)


def _draw_rates(
    ax,
    table: pd.DataFrame,
    designs: list[str],
    title: str,
    flags: dict[str, str] | None = None,
) -> None:
    """One model's grid: two bars per design, one per sample, base case centred.

    `flags` appends a marker to a design's tick -- what `adoption_rates_fisher` uses
    to say which of the separators it plots came out the wrong way round. The
    unparsed marker is not in it: that one is read off the table itself, so every
    caller gets it whether or not it thought to ask.
    """
    width = 0.38
    at = {label: index for index, label in enumerate(designs)}
    offsets = {"adopter": -width / 2, "non_adopter": width / 2, "none": 0.0}

    for sample in SAMPLE_ORDER:
        rows = table[table["sample"] == sample]
        if rows.empty:
            continue
        x = [at[label] + offsets[sample] for label in rows["design"]]
        ax.bar(
            x,
            rows["rate"],
            width=width if sample != "none" else width * 2,
            color=SAMPLE_COLOURS[sample],
            edgecolor=SURFACE,
            linewidth=0.5,
            label=SAMPLE_LABELS[sample],
            zorder=2,
        )
        ax.errorbar(
            x,
            rows["rate"],
            yerr=rows["se"],
            fmt="none",
            ecolor=INK_2,
            elinewidth=1.0,
            capsize=2.5,
            zorder=3,
        )

    # A design whose bars rest on unparsed responses is flagged on its tick, so a
    # rate computed from six answers is not read as one computed from ten.
    unparsed = set(table.loc[table["unparsed"] > 0, "design"])
    flags = flags or {}
    ticks = [f"{label}{'*' if label in unparsed else ''}{flags.get(label, '')}" for label in designs]

    ax.set_xticks(range(len(designs)))
    ax.set_xticklabels(ticks, rotation=90, fontsize=7, family="monospace", color=INK_2)
    # Headroom above 1.0 for the legend and the title to sit in. A model that saturates
    # puts a bar at 1.0 under every one of them, so the space has to be made rather
    # than borrowed -- the y ticks still stop at 1.0, which is where the scale ends.
    ax.set_ylim(0.0, 1.28)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("adoption rate", fontsize=9, color=INK_2)
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=HAIRLINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=8, colors=INK_2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(HAIRLINE)


def plot_adoption_rates(
    llm: LLMs | None = None,
    rates: pd.DataFrame | None = None,
    outfile: Path | None = None,
) -> plt.Figure:
    """The adoption rate of every design, with standard-error bars, one row per model.

    Reads whatever is in `output/pilot` unless a rate table is passed in -- so a
    half-finished run plots the designs it has. Designs keep `all_designs()`'s
    order rather than alphabetical, which puts the base case first and steps
    through the axes in the order the labels number them.

    Pass `outfile` to write it; the figure is returned either way.
    """
    if rates is None:
        rates = adoption_rates(load_results(llm))
    if rates.empty:
        raise ValueError("Nothing to plot: the rate table is empty")

    present = set(rates["design"])
    designs = [label for label in (design_label(*d) for d in all_designs()) if label in present]
    designs += sorted(present - set(designs))  # anything logged under a label we no longer generate
    models = list(dict.fromkeys(rates["llm"]))

    fig, axes = plt.subplots(
        len(models),
        1,
        figsize=(max(9.0, 0.42 * len(designs) + 2.0), 4.6 * len(models)),
        facecolor=SURFACE,
        squeeze=False,
        sharex=True,
    )
    for ax, model in zip(axes[:, 0], models):
        one = rates[rates["llm"] == model]
        reps = sorted(set(one["n"]))
        span = f"{reps[0]}" if len(reps) == 1 else f"{reps[0]}-{reps[-1]}"
        _draw_rates(ax, one, designs, f"{model}  ({span} answered repetitions per bar)")

    axes[0, 0].legend(
        frameon=False,
        fontsize=8,
        labelcolor=INK_2,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        ncols=3,
    )
    fig.text(
        0.005,
        0.005,
        "bars: share of (Y) answers   whiskers: SE of a proportion, sqrt(p(1-p)/n)"
        "   *: design with unparsed responses, excluded from its rate",
        fontsize=7,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 1.0))

    if outfile is not None:
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=200, facecolor=SURFACE, bbox_inches="tight")
        print(f"wrote {outfile}")
    return fig


# --------------------------------------------------------------------------
# Which designs separate the two samples
# --------------------------------------------------------------------------

# The level the corrected p-values are read against. Nothing here depends on it;
# it decides what `significant` says and what the report prints.
FDR_ALPHA = 0.05


def _benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    """FDR-corrected p-values, in the order they came in.

    Every design is one test of the same question, so the grid asks it 51 times per
    model: at an uncorrected 0.05 we would expect between one and two designs to
    look like separators purely by chance -- exactly the mistake that would send the
    main study off with the wrong prompt. Benjamini-Hochberg rather than Bonferroni
    because the designs share axes and are anything but independent, and because a
    pilot picking candidates can afford a controlled false-discovery rate.
    """
    order = pvalues.sort_values(kind="mergesort")
    m = len(order)
    scaled = order.to_numpy(dtype=float) * m / np.arange(1, m + 1)
    # Enforce monotonicity from the largest p downward, the standard step-up rule.
    adjusted = np.minimum.accumulate(scaled[::-1])[::-1].clip(max=1.0)
    return pd.Series(adjusted, index=order.index).reindex(pvalues.index)


def separation_floor(n_adopter: int, n_non_adopter: int) -> float:
    """The smallest p these repetition counts can produce: a perfect 2x2 split.

    Worth checking before reading any result off the grid. Ten repetitions per arm
    can reach 1.1e-5, but a 7-vs-3 split -- a 40-point difference in adoption rate --
    only reaches p = 0.18, which survives no correction at all. If the floor is
    close to alpha, the answer is more repetitions, not a different test.
    """
    return float(fisher_exact([[n_adopter, 0], [0, n_non_adopter]])[1])


def design_tests(
    rates: pd.DataFrame | None = None,
    llm: LLMs | None = None,
    alpha: float = FDR_ALPHA,
) -> pd.DataFrame:
    """Fisher's exact test per (model, design): does the adopter sample answer differently?

    One 2x2 table per design -- adopter/non-adopter against joined/did not -- tested
    exactly rather than by chi-square, because ten binary repetitions per arm put
    cells in the range where the chi-square approximation is not to be trusted. The
    test is two-sided: a design that makes the *non*-adopter household keener is as
    much a finding as the other direction, and `diff` carries the sign.

    A base case is not tested. It renders identically for both samples, so it has
    only a `none` arm and there is nothing to compare -- which is what makes the
    three of them, one per instruction, the controls for the other 51.

    Columns: the two rates and their counts, `diff` (adopter - non-adopter), the raw
    `p`, the BH-corrected `q`, `significant` (q < alpha), and `floor`, the best p
    those counts could have reached. Sorted by q, so the designs that separate the
    samples come first.
    """
    if rates is None:
        rates = adoption_rates(load_results(llm))

    rows = []
    two_arm = rates[rates["sample"].isin(("adopter", "non_adopter"))]
    for (model, design), group in two_arm.groupby(["llm", "design"], sort=False):
        arms = group.set_index("sample")
        if not {"adopter", "non_adopter"} <= set(arms.index):
            continue  # a design only half-run yet: one arm is not a comparison
        adopter, non_adopter = arms.loc["adopter"], arms.loc["non_adopter"]
        if min(adopter["n"], non_adopter["n"]) == 0:
            continue  # every repetition unparsed on one side
        table = [
            [int(adopter["adopted"]), int(adopter["n"] - adopter["adopted"])],
            [int(non_adopter["adopted"]), int(non_adopter["n"] - non_adopter["adopted"])],
        ]
        odds_ratio, p = fisher_exact(table, alternative="two-sided")
        rows.append(
            {
                "llm": model,
                "design": design,
                "n_adopter": int(adopter["n"]),
                "n_non_adopter": int(non_adopter["n"]),
                "rate_adopter": adopter["rate"],
                "rate_non_adopter": non_adopter["rate"],
                "diff": adopter["rate"] - non_adopter["rate"],
                "odds_ratio": float(odds_ratio),
                "p": float(p),
                "floor": separation_floor(int(adopter["n"]), int(non_adopter["n"])),
            }
        )

    tests = pd.DataFrame(rows, columns=[
        "llm", "design", "n_adopter", "n_non_adopter", "rate_adopter",
        "rate_non_adopter", "diff", "odds_ratio", "p", "floor",
    ])
    if tests.empty:
        return tests.assign(q=pd.Series(dtype=float), significant=pd.Series(dtype=bool))

    # Corrected within each model: the grid is one family of tests per model, and a
    # model is either the right instrument for this study or it is not.
    tests["q"] = tests.groupby("llm", sort=False)["p"].transform(_benjamini_hochberg)
    tests["significant"] = tests["q"] < alpha
    return tests.sort_values(["llm", "q", "p"], ignore_index=True)


def significance_report(tests: pd.DataFrame | None = None, alpha: float = FDR_ALPHA) -> pd.DataFrame:
    """Print the grid's verdict per model, best design first, and return the tests.

    "Best" is the design with the smallest corrected p -- the sharpest separation
    between an adopter household's circumstances and a non-adopter's. Where several
    tie at the floor, `diff` breaks the tie, and among those the leanest design wins
    on grounds the test cannot see: fewer axes is less prompt for the same signal.
    """
    if tests is None:
        tests = design_tests()
    if tests.empty:
        print("No two-arm design has been run yet: nothing to test.")
        return tests

    for model, group in tests.groupby("llm", sort=False):
        # Per design, not per model: repetitions stack design by design, so a grid
        # part-way through a pass has designs at different counts and therefore at
        # different floors. Reporting one number for the model would describe
        # whichever design is furthest behind as though it were all of them.
        best, worst = group["floor"].min(), group["floor"].max()
        hits = group[group["significant"]]
        print(f"\n{model}: {len(hits)}/{len(group)} designs separate the samples at q < {alpha}")
        span = f"{best:.2g}" if best == worst else f"{best:.2g} to {worst:.2g}, by how many repetitions each has"
        print(f"  smallest p these repetition counts can reach: {span}")
        if best > alpha:
            print("  -- that floor is above alpha: no design can come out significant. Add repetitions.")
        elif worst > alpha:
            underpowered = group[group["floor"] > alpha]
            print(
                f"  -- {len(underpowered)} design(s) cannot reach significance at their current counts, "
                f"whatever they answer: {', '.join(sorted(underpowered['design'])[:6])}"
                f"{' ...' if len(underpowered) > 6 else ''}"
            )

        shown = (hits if not hits.empty else group).head(10)
        print(f"  {'design':<10}{'adopter':>9}{'non-adopt':>11}{'diff':>8}{'p':>10}{'q':>10}")
        for _, row in shown.iterrows():
            print(
                f"  {row['design']:<10}{row['rate_adopter']:>9.2f}{row['rate_non_adopter']:>11.2f}"
                f"{row['diff']:>+8.2f}{row['p']:>10.3g}{row['q']:>10.3g}"
            )
        if hits.empty:
            print("  (none significant -- the ten closest are listed)")
        elif len(hits) > 10:
            print(f"  ... and {len(hits) - 10} more")

        # A design that separates the samples the wrong way round is not a usable
        # instrument, whatever its p-value.
        inverted = hits[hits["diff"] < 0]
        if not inverted.empty:
            print(f"  inverted (non-adopter keener): {', '.join(inverted['design'])}")

    return tests


FISHER_FIGURE = FIGURE_DIR / "adoption_rates_fisher.png"


def separating_designs(tests: pd.DataFrame, model: str) -> list[str]:
    """One model's significant designs, sharpest separation first.

    Ordered by the corrected p rather than by the grid, because on this figure the
    order *is* the result: the design at the left is the one the main study would
    be run with.
    """
    hits = tests[(tests["llm"] == model) & tests["significant"]]
    return list(hits.sort_values(["q", "p"], kind="mergesort")["design"])


def adoption_rates_fisher(
    llm: LLMs | None = None,
    rates: pd.DataFrame | None = None,
    tests: pd.DataFrame | None = None,
    alpha: float = FDR_ALPHA,
    outfile: Path | None = None,
) -> plt.Figure:
    """The base cases and only the designs that separate the samples, one row per model.

    `plot_adoption_rates` puts all 54 designs on an axis at 7-point ticks, which is
    the right figure for seeing the grid and the wrong one for seeing the answer.
    This is the answer: the three base cases -- untested, since they render
    identically for both samples, and therefore the reference the rest are read
    against -- followed by the designs whose Fisher test survived BH correction, in
    order of how sharply they separate.

    A design that separates the samples the wrong way round -- the non-adopter
    household came out keener -- is held back behind a red rule at the right, in its
    own block. It is on the figure because the test found it and hiding it would
    misreport the grid; it is behind the rule because it is not a candidate for the
    main study, and reading it in q order alongside the usable designs would put it
    forward as one.

    Pass `outfile` to write it; the figure is returned either way.
    """
    if rates is None:
        rates = adoption_rates(load_results(llm))
    if rates.empty:
        raise ValueError("Nothing to plot: the rate table is empty")
    if tests is None:
        tests = design_tests(rates=rates, alpha=alpha)
    if tests.empty:
        raise ValueError("No two-arm design has been tested yet: there is nothing to select from")

    # Grid order for the base cases -- D0, D1, D2 -- so the instruction axis reads
    # left to right, the one place on this figure where the grid's order still means
    # something.
    base_present = set(rates.loc[rates["sample"] == "none", "design"])
    base = [label for label in (design_label(*d) for d in all_designs()) if label in base_present]

    models = list(dict.fromkeys(rates["llm"]))
    # Three blocks per model: the untested reference, the usable separators, and the
    # inverted ones. Each keeps q order within itself.
    inverted = {model: set(tests.loc[(tests["llm"] == model) & tests["significant"] & (tests["diff"] < 0), "design"])
                for model in models}
    blocks = {}
    for model in models:
        hits = separating_designs(tests, model)
        blocks[model] = (base, [d for d in hits if d not in inverted[model]], [d for d in hits if d in inverted[model]])
        if not hits:
            print(f"{model}: no design separates the samples at q < {alpha}; only the base cases are plotted.")
    selected = {model: [label for block in blocks[model] for label in block] for model in models}
    widest = max(len(labels) for labels in selected.values())

    fig, axes = plt.subplots(
        len(models),
        1,
        figsize=(max(7.0, 0.78 * widest + 2.5), 4.6 * len(models)),
        facecolor=SURFACE,
        squeeze=False,
    )
    for ax, model in zip(axes[:, 0], models):
        one = tests[tests["llm"] == model]
        labels = selected[model]
        reference, usable, wrong_way = blocks[model]
        _draw_rates(
            ax,
            rates[(rates["llm"] == model) & rates["design"].isin(labels)],
            labels,
            f"{model}  ({len(usable) + len(wrong_way)}/{len(one)} designs separate the samples at q < {alpha})",
            flags={label: "†" for label in wrong_way},
        )
        # The base cases are a block, not a series: a rule between them and the
        # separators keeps the reference from being read as the first result.
        if reference and len(labels) > len(reference):
            ax.axvline(len(reference) - 0.5, color=HAIRLINE, linewidth=1.0, zorder=1)
        # The second rule is red because what it fences off is a warning: everything
        # to its right is significant and pointing the wrong way.
        if wrong_way and len(wrong_way) < len(labels):
            ax.axvline(len(labels) - len(wrong_way) - 0.5, color=WARNING, linewidth=1.4, zorder=1)

    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left", ncols=3)
    fig.tight_layout()

    if outfile is not None:
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=200, facecolor=SURFACE, bbox_inches="tight")
        print(f"wrote {outfile}")
    return fig


# --------------------------------------------------------------------------
# The DT designs: the matrices behind the decisions
# --------------------------------------------------------------------------

DT_FRAME_COLUMNS = (
    ["llm", "design", "sample", "repetition", "decision", "parsed"]
    + [f"p_{state}" for state in DT_STATES]
    + [f"uY_{state}" for state in DT_STATES]
    + [f"uN_{state}" for state in DT_STATES]
    + ["p_sum", "p_valid", "eu_y", "eu_n", "eu_margin", "coherent", "n_evidence"]
)


def dt_frame(results: pd.DataFrame | None = None, llm: LLMs | None = None) -> pd.DataFrame:
    """The elicited decision matrices, one row per DT repetition.

    Derived from the `response` column rather than logged alongside it. That is
    not only tidiness: `_append_row` writes a header only when it creates the
    file, so a column added to the log would have to be added to all 54 files,
    including the ones a later run appends more repetitions to. The raw response
    is already the record of what came back, and with the schema enforcing its
    shape, reading the matrix out of it is a `json.loads`.

    Every D2 repetition gets a row, parsed or not. `parsed` says which; `p_sum`
    and `p_valid` say whether the three probabilities are a distribution -- the
    one part of the schema strict mode could not enforce; and `coherent` says
    whether the decision the model stated is the one its own matrix implies,
    which is the question the mode exists to make askable. A tie in expected
    utility counts as coherent either way: there is nothing to violate.
    """
    if results is None:
        results = load_results(llm)
    dt_rows = results[results["design"].astype(str).str.endswith("D2")]

    rows = []
    for record in dt_rows.to_dict("records"):
        payload = parse_dt(str(record.get("response") or ""))
        row: dict[str, object] = {
            "llm": record["llm"],
            "design": record["design"],
            "sample": record["sample"],
            "repetition": record["repetition"],
            "decision": record["decision"],
            "parsed": payload is not None,
        }
        if payload is not None:
            states = payload["states"]
            probability = {state: float(states[state]["probability"]) for state in DT_STATES}
            utility = {
                action: {state: float(states[state][f"{action}_utility"]) for state in DT_STATES}
                for action in ("Y", "N")
            }
            expected = {
                action: sum(probability[state] * utility[action][state] for state in DT_STATES)
                for action in ("Y", "N")
            }
            total = sum(probability.values())
            # The stated decision comes off the payload rather than off the CSV's
            # column, so coherence is a property of the response alone.
            stated_yes = payload["decision"] == "Y"
            row.update({f"p_{state}": probability[state] for state in DT_STATES})
            row.update({f"uY_{state}": utility["Y"][state] for state in DT_STATES})
            row.update({f"uN_{state}": utility["N"][state] for state in DT_STATES})
            row["p_sum"] = total
            row["p_valid"] = bool(
                abs(total - 1.0) <= DT_PROBABILITY_TOLERANCE
                and all(0.0 <= p <= 1.0 for p in probability.values())
            )
            row["eu_y"], row["eu_n"] = expected["Y"], expected["N"]
            row["eu_margin"] = expected["Y"] - expected["N"]
            row["coherent"] = bool(expected["Y"] >= expected["N"]) if stated_yes else bool(expected["N"] >= expected["Y"])
            row["n_evidence"] = sum(len(states[state].get("evidence") or []) for state in DT_STATES)
        rows.append(row)

    return pd.DataFrame(rows, columns=DT_FRAME_COLUMNS)


def dt_tests(dt: pd.DataFrame | None = None, alpha: float = FDR_ALPHA) -> pd.DataFrame:
    """Mann-Whitney per (model, DT design): is P(beneficial) higher for the adopter sample?

    The companion to `design_tests`, on a different outcome. The decision is one
    bit per repetition, and Fisher's exact on twenty of them needs a near-perfect
    split to survive correction -- a 7-vs-3 difference in adoption rate reaches
    p = 0.18 and no further. The elicited probability of the beneficial state is
    continuous, so the same twenty repetitions carry far more of the difference
    between the two samples, if there is one. A design can separate the samples
    here and not there, and that is informative rather than contradictory: it
    says the households' circumstances moved the model's beliefs without moving
    its answer across the threshold.

    Rank-based rather than a t-test: these are bounded, often clustered
    probabilities with no reason to be normal. BH-corrected within model, as in
    `design_tests`, and `diff` is the difference in means, which carries the sign.
    """
    if dt is None:
        dt = dt_frame()

    rows = []
    usable = dt[dt["parsed"] & dt["sample"].isin(("adopter", "non_adopter"))]
    for (model, design), group in usable.groupby(["llm", "design"], sort=False):
        arms = {
            sample: group.loc[group["sample"] == sample, "p_beneficial"].to_numpy(dtype=float)
            for sample in ("adopter", "non_adopter")
        }
        if min(len(arm) for arm in arms.values()) == 0:
            continue  # a design only half-run, or one arm that never parsed
        adopter, non_adopter = arms["adopter"], arms["non_adopter"]
        pooled = np.concatenate([adopter, non_adopter])
        if np.allclose(pooled, pooled[0]):
            # Every repetition on both sides gave the same probability. There is
            # no rank information at all, and mannwhitneyu would report a tie as
            # if it were a test; p = 1 is what "these are indistinguishable" means.
            statistic, p = float("nan"), 1.0
        else:
            statistic, p = mannwhitneyu(adopter, non_adopter, alternative="two-sided")
        rows.append(
            {
                "llm": model,
                "design": design,
                "n_adopter": len(adopter),
                "n_non_adopter": len(non_adopter),
                "mean_adopter": float(adopter.mean()),
                "mean_non_adopter": float(non_adopter.mean()),
                "diff": float(adopter.mean() - non_adopter.mean()),
                "u": float(statistic),
                "p": float(p),
            }
        )

    tests = pd.DataFrame(rows, columns=[
        "llm", "design", "n_adopter", "n_non_adopter", "mean_adopter", "mean_non_adopter", "diff", "u", "p",
    ])
    if tests.empty:
        return tests.assign(q=pd.Series(dtype=float), significant=pd.Series(dtype=bool))

    tests["q"] = tests.groupby("llm", sort=False)["p"].transform(_benjamini_hochberg)
    tests["significant"] = tests["q"] < alpha
    return tests.sort_values(["llm", "q", "p"], ignore_index=True)


def dt_report(
    dt: pd.DataFrame | None = None,
    tests: pd.DataFrame | None = None,
    alpha: float = FDR_ALPHA,
) -> pd.DataFrame:
    """Print what the DT designs produced, and return the P(beneficial) tests.

    Two blocks per model. The first is whether the mode worked at all: how many
    responses carried a matrix, how many of those carried a distribution, and how
    often the stated decision agreed with the matrix's own argmax. A low
    coherence rate is not a bug in the parsing -- it is the finding that the
    model's analysis and its answer are produced by different processes, and it
    is worth knowing before any of the numbers below it are read.

    The second is which designs separate the samples on the elicited probability
    of the beneficial state, best first, alongside the adoption rates the same
    designs produced.
    """
    if dt is None:
        dt = dt_frame()
    if dt.empty:
        print("No DT design (D2) has been run yet: nothing to report.")
        return pd.DataFrame()
    if tests is None:
        tests = dt_tests(dt, alpha=alpha)

    for model, group in dt.groupby("llm", sort=False):
        parsed = group[group["parsed"]]
        print(f"\n{model}: {len(group)} DT repetition(s) across {group['design'].nunique()} design(s)")
        print(f"  matrix parsed:        {len(parsed)}/{len(group)}")
        if parsed.empty:
            print("  -- nothing parsed, so there is no matrix to report on.")
            continue
        valid = int(parsed["p_valid"].sum())
        coherent = int(parsed["coherent"].sum())
        print(f"  probabilities sum to 1 (+-{DT_PROBABILITY_TOLERANCE}): {valid}/{len(parsed)}")
        print(f"  decision agrees with its own argmax EU:  {coherent}/{len(parsed)} ({coherent / len(parsed):.0%})")
        print(f"  evidence items per response: {parsed['n_evidence'].mean():.1f} mean")
        for sample in SAMPLE_ORDER:
            arm = parsed[parsed["sample"] == sample]
            if arm.empty:
                continue
            print(
                f"  {SAMPLE_LABELS[sample]:<18} P(beneficial) {arm['p_beneficial'].mean():.3f}"
                f" +- {arm['p_beneficial'].std(ddof=1) if len(arm) > 1 else 0.0:.3f}"
                f"   EU margin {arm['eu_margin'].mean():+.2f}   n = {len(arm)}"
            )

        one = tests[tests["llm"] == model] if not tests.empty else tests
        if one.empty:
            print("  no two-arm DT design has parsed on both sides yet: nothing to test.")
            continue
        hits = one[one["significant"]]
        print(f"\n  {len(hits)}/{len(one)} DT design(s) separate the samples on P(beneficial) at q < {alpha}")
        print(f"  {'design':<10}{'adopter':>9}{'non-adopt':>11}{'diff':>8}{'p':>10}{'q':>10}")
        for _, row in (hits if not hits.empty else one).head(10).iterrows():
            print(
                f"  {row['design']:<10}{row['mean_adopter']:>9.3f}{row['mean_non_adopter']:>11.3f}"
                f"{row['diff']:>+8.3f}{row['p']:>10.3g}{row['q']:>10.3g}"
            )
        if hits.empty:
            print("  (none significant -- the ten closest are listed)")

    return tests


DT_FIGURE = FIGURE_DIR / "dt_rationality.png"

# Which way a matrix points, in the order the panel puts them. The tie is last
# because it is the rare case, and it is its own direction rather than folded into
# either side: with equal expected utilities neither answer can be inconsistent, so
# counting ties with one side would put unfalsifiable rows into that side's ratio.
EU_DIRECTIONS = (("y", "EU(Y) > EU(N)"), ("n", "EU(Y) < EU(N)"), ("tie", "EU(Y) = EU(N)"))

# The decision axis borrows the sample palette: joining is the warm colour on both
# figures, which is what keeps them readable side by side.
DECISION_COLOURS = {YES_TOKEN: SAMPLE_COLOURS["adopter"], NO_TOKEN: SAMPLE_COLOURS["non_adopter"]}

def _percent(share: float) -> str:
    """A share as whole percent, except that a nonzero one never prints as 0%.

    The tie group is two responses out of seven hundred. Rounding that to "0%"
    beside "n = 2" reads as a rendering fault rather than as a rare case.
    """
    if math.isnan(share):
        return "--"
    if 0.0 < share < 0.005:
        return "<1%"
    if 0.995 < share < 1.0:
        return ">99%"
    return f"{share:.0%}"


EU_SPLIT_COLUMNS = ("llm", "direction", "label", "n", "share", "yes", "no", "share_yes")


def eu_decision_split(dt: pd.DataFrame | None = None, llm: LLMs | None = None) -> pd.DataFrame:
    """Per (model, EU direction): how many responses point that way, and how many said (Y).

    The two questions the first panel asks, in one table. `share` is over the
    model's parsed responses -- how often the elicited matrix came out in favour of
    joining at all -- and `share_yes` is *within* the direction: of the responses
    whose own matrix favoured joining, how many went on to answer (Y). The second
    is the interesting one. A model whose matrices favour joining 93% of the time
    but who answers (Y) in only 72% of those is not being swayed by its own
    analysis, and no amount of prompt design on top of it will fix that.

    A direction no response took is dropped rather than plotted at zero, so a model
    that never produced a tie is not given an empty third group.
    """
    if dt is None:
        dt = dt_frame(llm=llm)
    parsed = dt[dt["parsed"]]

    rows = []
    for model, group in parsed.groupby("llm", sort=False):
        margin = group["eu_margin"].to_numpy(dtype=float)
        keys = pd.Series(np.where(margin > 0, "y", np.where(margin < 0, "n", "tie")), index=group.index)
        for key, label in EU_DIRECTIONS:
            arm = group[keys == key]
            if arm.empty:
                continue
            yes = int((arm["decision"] == YES_TOKEN).sum())
            no = int((arm["decision"] == NO_TOKEN).sum())
            rows.append(
                {
                    "llm": model,
                    "direction": key,
                    "label": label,
                    "n": len(arm),
                    "share": len(arm) / len(group),
                    "yes": yes,
                    "no": no,
                    # Over answered repetitions, not over `n`: a matrix can parse and
                    # its decision token still not, and dividing by `n` would report
                    # that as a (N).
                    "share_yes": yes / (yes + no) if yes + no else float("nan"),
                }
            )
    return pd.DataFrame(rows, columns=list(EU_SPLIT_COLUMNS))


def _draw_eu_split(ax, split: pd.DataFrame, title: str) -> None:
    """Per EU direction: how many responses took it, and what they then answered.

    Two bars per group, on two different denominators, which is why they are drawn
    apart rather than stacked into one. The wide one is the share of all parsed
    responses -- it sums to 1 across the groups. The narrow one beside it always
    reaches 1.0 and is split at the share of that group's answers that were (Y): it
    is a decomposition of its own group, not of the axis.
    """
    main_width, side_width, gap = 0.34, 0.20, 0.12
    # The pair is centred on the tick, not the wide bar: the label underneath names
    # the group, and both bars are in it.
    left = -(main_width + gap + side_width) / 2

    for index, row in enumerate(split.itertuples()):
        mx = index + left + main_width / 2
        ax.bar(
            mx, row.share, width=main_width, color=SAMPLE_COLOURS["none"],
            edgecolor=SURFACE, linewidth=0.5, zorder=2,
            label="share of parsed responses" if index == 0 else None,
        )
        ax.text(mx, row.share + 0.015, f"{_percent(row.share)}\nn = {row.n}", ha="center", va="bottom",
                fontsize=7.5, color=INK_2, linespacing=1.35)

        sx = index - left - side_width / 2
        yes = 0.0 if math.isnan(row.share_yes) else row.share_yes
        ax.bar(sx, yes, width=side_width, color=DECISION_COLOURS[YES_TOKEN], edgecolor=SURFACE,
               linewidth=0.5, zorder=2, label="answered (Y)" if index == 0 else None)
        ax.bar(sx, 1.0 - yes, bottom=yes, width=side_width, color=DECISION_COLOURS[NO_TOKEN],
               edgecolor=SURFACE, linewidth=0.5, zorder=2, label="answered (N)" if index == 0 else None)
        ax.text(sx, 1.015, f"{_percent(yes)} (Y)", ha="center", va="bottom", fontsize=7.5, color=INK_2)

    ax.set_xticks(range(len(split)))
    ax.set_xticklabels(list(split["label"]), fontsize=8.5, color=INK_2)
    ax.set_xlim(-0.6, len(split) - 0.4)
    # Room for the group labels above the bars and the legend above those. The
    # sidebars all end at 1.0, so the space has to be made rather than borrowed.
    ax.set_ylim(0.0, 1.45)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("share", fontsize=9, color=INK_2)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=HAIRLINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=8, colors=INK_2)
    ax.tick_params(axis="x", length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(HAIRLINE)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper right", ncols=1)


def _draw_coherence_pie(ax, parsed: pd.DataFrame, title: str) -> None:
    """One model's answers, split by decision and by whether the matrix backed it.

    A donut rather than a pie so the headline -- what share of the answers their own
    matrix implied -- can sit in the middle, where it is read first. The two
    consistent wedges are adjacent and the two inconsistent ones are adjacent, so
    the split is one arc rather than four numbers to add up.
    """
    yes, no = parsed["decision"] == YES_TOKEN, parsed["decision"] == NO_TOKEN
    coherent = parsed["coherent"].astype(bool)
    slices = [
        ("consistent (Y)", int((coherent & yes).sum()), DECISION_COLOURS[YES_TOKEN], None),
        ("consistent (N)", int((coherent & no).sum()), DECISION_COLOURS[NO_TOKEN], None),
        ("inconsistent (N)", int((~coherent & no).sum()), DECISION_COLOURS[NO_TOKEN], "////"),
        ("inconsistent (Y)", int((~coherent & yes).sum()), DECISION_COLOURS[YES_TOKEN], "////"),
    ]
    slices = [entry for entry in slices if entry[1] > 0]
    total = sum(count for _, count, _, _ in slices)
    if not total:
        raise ValueError("Nothing to plot: no parsed DT response carried a decision")

    wedges, _ = ax.pie(
        [count for _, count, _, _ in slices],
        colors=[colour for _, _, colour, _ in slices],
        labels=[f"{label}\n{count} ({count / total:.0%})" for label, count, _, _ in slices],
        startangle=90,
        counterclock=False,
        textprops={"fontsize": 8, "color": INK_2},
        wedgeprops={"width": 0.44, "edgecolor": SURFACE, "linewidth": 1.2},
        labeldistance=1.12,
    )
    # Hatched in the surface colour rather than shaded: two hues already carry the
    # decision, and a third variation of them would be one signal too many.
    for wedge, (_, _, _, hatch) in zip(wedges, slices):
        if hatch:
            wedge.set_hatch(hatch)
            wedge.set_alpha(0.72)

    agreed = int(parsed["coherent"].sum())
    ax.text(0, 0.08, f"{agreed / len(parsed):.0%}", ha="center", va="center", fontsize=20, color=INK)
    ax.text(0, -0.14, f"consistent\n{agreed}/{len(parsed)}", ha="center", va="center",
            fontsize=8, color=MUTED, linespacing=1.4)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8)


def plot_dt_rationality(
    dt: pd.DataFrame | None = None,
    llm: LLMs | None = None,
    split: pd.DataFrame | None = None,
    outfile: Path | None = None,
) -> plt.Figure:
    """Whether the DT designs' decisions follow the matrices they came with.

    Two panels per model, on the same question from two sides. The left one asks it
    conditionally -- given a matrix that favoured joining, how often did the answer
    follow, and likewise for one that favoured staying out -- which is where a model
    that ignores its analysis in one direction only shows up. The right one asks it
    once, over everything: what share of the answers were the argmax of the
    response's own expected utilities.

    Ties count as consistent on the right panel and sit in their own group on the
    left, per `dt_frame` and `eu_decision_split`. Unparsed responses are in neither:
    there is no matrix to be consistent with.

    Pass `outfile` to write it; the figure is returned either way.
    """
    if dt is None:
        dt = dt_frame(llm=llm)
    if dt.empty:
        raise ValueError("Nothing to plot: no DT design (D2) has been run")
    parsed = dt[dt["parsed"]]
    if parsed.empty:
        raise ValueError("Nothing to plot: no DT response carried a matrix")
    if split is None:
        split = eu_decision_split(dt)

    models = list(dict.fromkeys(parsed["llm"]))
    fig, axes = plt.subplots(
        len(models), 2, figsize=(11.0, 4.9 * len(models)), facecolor=SURFACE, squeeze=False,
    )
    for (left, right), model in zip(axes, models):
        one = parsed[parsed["llm"] == model]
        _draw_eu_split(left, split[split["llm"] == model], f"{model}  ({len(one)} parsed matrices)")
        _draw_coherence_pie(right, one, "decision vs. its own argmax EU")

    fig.tight_layout()

    if outfile is not None:
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=200, facecolor=SURFACE, bbox_inches="tight")
        print(f"wrote {outfile}")
    return fig


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

DESCRIPTION = "The adoption-rate pilot: one prompt-design grid per model, its rates, and which designs separate the samples."

# The short name each model answers to on the command line, next to the value and
# the enum name -- `--models gpt` is what a run is actually typed as.
MODEL_ALIASES = {"gpt": LLMs.GPT_5_4_NANO, "haiku": LLMs.HAIKU_4_5, "grok": LLMs.GROK_4_2}

DEFAULT_FIGURE = FIGURE_DIR / "adoption_rates.png"

# Where each `plot --kind` writes, unless --outfile says otherwise.
PLOT_KINDS = {"rates": DEFAULT_FIGURE, "fisher": FISHER_FIGURE, "dt": DT_FIGURE}


def resolve_model(label: str) -> LLMs:
    """`gpt`, `gpt-5.4-nano` or `GPT_5_4_NANO` -> the enum member."""
    key = label.strip()
    if key.lower() in MODEL_ALIASES:
        return MODEL_ALIASES[key.lower()]
    by_name = {llm.name.lower(): llm for llm in LLMs}
    if key.lower() in by_name:
        return by_name[key.lower()]
    return get_llm(key)  # raises with the list of valid values


def resolve_designs(labels: list[str] | None) -> list[tuple[str, str, str, str]]:
    """Design labels back into the tuples `run_pilot` takes, in grid order.

    The label is the only handle a design has outside the code -- it is what the CSV
    filenames and the plot's ticks are written in -- so `--designs A1B0C1D0` is how
    one design out of the 54 is asked for by name.
    """
    if not labels:
        return all_designs()
    by_label = {design_label(*design): design for design in all_designs()}
    wanted = [label.strip().upper() for label in labels]
    unknown = [label for label in wanted if label not in by_label]
    if unknown:
        raise ValueError(f"no such design(s): {', '.join(unknown)}. Labels look like A0B0C0D0 (see design_label)")
    # Grid order, not the order they were typed, so a partial run reads like the whole.
    return [design for label, design in by_label.items() if label in set(wanted)]


def dry_run(designs: list[tuple[str, str, str, str]], print_prompts: bool = False) -> int:
    """Render every design and call nothing: what `--live` would send, for free.

    The failure this catches is the one worth catching before paying for 105 calls: a
    sample household with a missing field, or a profiles file that was never built,
    fails here at design one rather than three designs into the grid. Both arms are
    rendered, because a field missing on the non-adopter household would otherwise
    only surface halfway through the live run. Rendering is model-independent, so
    this is the same check whichever models were asked for.
    """
    rendered = failed = 0
    for design in designs:
        label = design_label(*design)
        for sample, (ego_hhid, informer_hhid) in design_samples(*design[:3]).items():
            try:
                prompt = get_prompt(
                    *design,
                    hhid=ego_hhid if design[0] else None,
                    informer_hhid=informer_hhid if (design[1] or design[2]) else None,
                )
            except Exception as exc:  # noqa: BLE001 -- report every broken design, not the first
                print(f"  {label} {sample:<12} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
                failed += 1
                continue
            rendered += 1
            print(f"  {label} {sample:<12} {len(prompt):>5} chars")
            if print_prompts:
                print("\n".join(f"    | {line}" for line in prompt.splitlines()) + "\n")

    if failed:
        print(f"\n{failed} prompt(s) could not be rendered; nothing would have been sent.", file=sys.stderr)
        return 1
    print(f"\n{rendered} prompt(s) across {len(designs)} design(s) render. Nothing was sent: add --live to run it.")
    return 0


def main(argv: list[str] | None = None) -> int:
    # The paths are module-level because everything from `log_path` to `load_results`
    # reads them; pointing the CLI somewhere else means rebinding OUTPUT_DIR once,
    # below, rather than threading a directory through every function touching a file.
    global OUTPUT_DIR

    argv = list(sys.argv[1:] if argv is None else argv)
    # `--live` means `run --live`, and a bare invocation means `run`. `--help` is left
    # alone, or the top-level help would never list the other two commands.
    if not argv or (argv[0].startswith("-") and argv[0] not in ("-h", "--help")):
        argv.insert(0, "run")

    p = argparse.ArgumentParser(description=DESCRIPTION)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run the grid: every design on every model, reps times each")
    r.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help=f"any of {', '.join(MODEL_ALIASES)} or a full model id (default: all {len(LLMs)}; "
        "the ones with no provider wired up are reported and skipped)",
    )
    r.add_argument(
        "--designs",
        nargs="+",
        default=None,
        metavar="LABEL",
        help="design labels like A1B0C1D0 (default: the full 54-design factorial)",
    )
    r.add_argument(
        "--reps",
        type=int,
        default=1,
        help="repetitions per design per sample (default: 1). Repetitions stack: this asks for N more.",
    )
    r.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help=f"where the CSVs go (default: {OUTPUT_DIR})")
    r.add_argument(
        "--live",
        action="store_true",
        help="actually call the APIs (default: dry-run, renders every prompt, no cost, no calls made)",
    )
    r.add_argument("--print-prompts", action="store_true", help="dry run: print each prompt in full, not just its size")

    pl = sub.add_parser("plot", help="the pilot's figures: the whole grid, only the separators, or the DT matrices")
    pl.add_argument("--models", nargs="+", default=None, metavar="MODEL", help="default: every model in the logs")
    pl.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="where to read the CSVs from")
    pl.add_argument(
        "--kind",
        choices=tuple(PLOT_KINDS),
        default="rates",
        help="rates: every design that has been run. fisher: the base cases and only the designs that "
        "separate the samples. dt: whether the DT decisions follow their own matrices. (default: rates)",
    )
    pl.add_argument("--alpha", type=float, default=FDR_ALPHA, help=f"--kind fisher: the FDR level (default: {FDR_ALPHA})")
    pl.add_argument(
        "--outfile",
        type=Path,
        default=None,
        help="default: the figure for this kind (" + ", ".join(f"{k} -> {v}" for k, v in PLOT_KINDS.items()) + ")",
    )

    rp = sub.add_parser("report", help="Fisher's exact test per design: which ones separate the two samples")
    rp.add_argument("--models", nargs="+", default=None, metavar="MODEL", help="default: every model in the logs")
    rp.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="where to read the CSVs from")
    rp.add_argument("--alpha", type=float, default=FDR_ALPHA, help=f"the FDR level (default: {FDR_ALPHA})")
    rp.add_argument("--csv", type=Path, default=None, help="also write the full test table here")

    dt = sub.add_parser("dt", help="the DT designs: parse and coherence rates, and P(beneficial) per sample")
    dt.add_argument("--models", nargs="+", default=None, metavar="MODEL", help="default: every model in the logs")
    dt.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="where to read the CSVs from")
    dt.add_argument("--alpha", type=float, default=FDR_ALPHA, help=f"the FDR level (default: {FDR_ALPHA})")
    dt.add_argument("--csv", type=Path, default=None, help="also write the per-repetition matrix table here")

    a = p.parse_args(argv)

    OUTPUT_DIR = a.output_dir

    try:
        models = [resolve_model(label) for label in a.models] if a.models else None
        designs = resolve_designs(getattr(a, "designs", None))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if a.command == "run":
        llms = models if models is not None else list(LLMs)
        # Only the base case needs no household data, so only it can run from anywhere.
        needs_data = any(profile or informer or endorsement for profile, informer, endorsement, _ in designs)
        missing = [path for path in (FEATURES_PATH, PROFILES_PATH) if not path.is_file()]
        if needs_data and missing:
            print(
                f"error: {', '.join(str(path) for path in missing)} not found. These paths are relative: "
                "run this from the repository root.",
                file=sys.stderr,
            )
            return 1
        if not a.live:
            print(
                f"{len(llms)} model(s) x {len(designs)} design(s) x {a.reps} rep(s) "
                f"= {planned_calls(llms, designs, a.reps)} calls, if this were --live"
            )
            return dry_run(designs, print_prompts=a.print_prompts)
        run_pilot(llms, designs, reps=a.reps)  # prints the same plan itself
        return 0

    # The readers want one model or all of them, and `load_results` takes one.
    if models is not None and len(models) > 1:
        print(
            "error: plot, report and dt read one model at a time, or all of them if --models is omitted",
            file=sys.stderr,
        )
        return 1
    only = models[0] if models else None

    try:
        results = load_results(only, output_dir=OUTPUT_DIR)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if a.command == "dt":
        matrices = dt_frame(results)
        dt_report(matrices, alpha=a.alpha)
        if a.csv is not None:
            a.csv.parent.mkdir(parents=True, exist_ok=True)
            matrices.to_csv(a.csv, index=False)
            print(f"\nwrote {a.csv}")
        return 0

    if a.command == "plot":
        outfile = a.outfile or PLOT_KINDS[a.kind]
        try:
            if a.kind == "dt":
                plot_dt_rationality(dt_frame(results), outfile=outfile)
            elif a.kind == "fisher":
                adoption_rates_fisher(rates=adoption_rates(results), alpha=a.alpha, outfile=outfile)
            else:
                plot_adoption_rates(rates=adoption_rates(results), outfile=outfile)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    rates = adoption_rates(results)

    tests = design_tests(rates=rates, alpha=a.alpha)
    significance_report(tests, alpha=a.alpha)
    if a.csv is not None:
        a.csv.parent.mkdir(parents=True, exist_ok=True)
        tests.to_csv(a.csv, index=False)
        print(f"\nwrote {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())