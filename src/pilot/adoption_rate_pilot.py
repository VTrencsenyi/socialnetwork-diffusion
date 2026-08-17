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
from scipy.stats import fisher_exact
from tqdm.auto import tqdm

try:
    from ..llm import load_client, one_call
except ImportError:  # running as a script, not a package
    # `python src/pilot/adoption_rate_pilot.py` puts src/pilot on sys.path, not src,
    # so the sibling module is only importable once src itself is on it.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from llm import load_client, one_call  # type: ignore[no-redef]


"""
TODO:
Implement a new prompt mode 'DT-mode' on top of the others that implements a decision theoretic analysis.
We tell the agent to use all the available information and his subjective judgement to conduct a decision-theoretic analysis and fill out a decision matrix.
I.e. for the case of 'Should my household adopt the microfinance service?", given two actions {Y,N} and three state of nature describing the effects of taking a loan {beneficial,limited,harmful},
determine the subjective utilities for each action x state pair, and estimate probabilities of each of the 3 states happening. For each state, provide evidence that justifies the values.
We expect a structured response inm the format of a JSON like:

{
  "states": [
    {
      "state": "beneficial",
      "probability": 0.6,
      "Y_utility": 3,
      "N_utility": 0,
      "evidence": [
        "Household operates an income-generating activity",
        "Information came from a trusted contact"
      ]
    },
    {
      "state": "limited_effect",
      "probability": 0.25,
      "Y_utility": -1,
      "N_utility": 0,
      "evidence": [
        "The profitability of using additional capital is uncertain"
      ]
    },
    {
      "state": "harmful",
      "probability": 0.15,
      "Y_utility": -4,
      "N_utility": 0,
      "evidence": [
        "The household has limited savings to absorb repayment difficulties"
      ]
    }
  ],
  "decision": "Y"
}

Note, as this in practice conflates with MOA, disable moa when DT-mode is active.
"""


BASE_CONTEXT = """
An institution providing microfinance services has started a new programme in villages across Karnataka, India.
Their services have entered your village, and as the head of the household you have been asked to consider joining the programme.
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
MAX_OUTPUT_TOKENS = 512

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


def get_response(llm: LLMs, prompt: str) -> tuple[str, str, dict[str, int]]:
    """One call to one model: the text, the decision in it, and what it cost.

    Three of the CSV's four columns, so that a row is one call to this function.
    A response with no decision in it comes back as `PARSING_ERROR` rather than an
    exception, so the text still reaches the log -- it is the only record of what
    the model said instead, and losing it to a traceback would lose the evidence.
    Transient API errors are retried, and anything left raises.
    """
    if llm is not LLMs.GPT_5_4_NANO:
        raise NotImplementedError(
            f"{llm.value} is not wired up yet -- only {LLMs.GPT_5_4_NANO.value} answers so far."
        )
    text, usage = one_call(
        _client(PROVIDERS[llm]),
        {"model": llm.value, "input": prompt},
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
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
    use_moa: bool = False,
    hhid: int | None = None,
    informer_hhid: int | None = None,
) -> str:
    base_context = BASE_CONTEXT

    assert profile_enhancement in ["", "DEMOGRAPHIC", "NARRATIVE"], "Invalid profile enhancement option"
    assert informer_enhancement in ["", "DEMOGRAPHIC", "NARRATIVE"], "Invalid informer enhancement option"
    assert endorsement_enhancement in ["", "ENDORSEMENT"], "Invalid endorsement enhancement option"

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

    # Axis D: the MOA instruction replaces the plain one, and carries the format
    # instruction itself, so exactly one of the two ends every prompt.
    instruction = MOA_INSTRUCTION if use_moa else FORMAT_INSTRUCTION
    prompt_parts.append(instruction.format(Y=YES_TOKEN, N=NO_TOKEN))

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
    use_moa: bool = False,
) -> str:
    """`A1B0C1D0` -- one digit per axis, in the order the TODO block names them."""
    levels = {"": 0, "DEMOGRAPHIC": 1, "NARRATIVE": 2}
    return (
        f"A{levels[profile_enhancement]}"
        f"B{levels[informer_enhancement]}"
        f"C{int(bool(endorsement_enhancement))}"
        f"D{int(bool(use_moa))}"
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
    The MOA axis is not asked for: it changes the instruction, not the household.
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
    use_moa: bool = False,
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

    path = log_path(llm, design_label(profile_enhancement, informer_enhancement, endorsement_enhancement, use_moa))
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
            use_moa,
            ego_hhid,
            informer_hhid,
        )
        first = _next_repetition(path, sample)
        for repetition in range(first, first + reps):
            response, decision, usage = get_response(llm, prompt)
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
MOA_LEVELS = (False, True)


def all_designs() -> list[tuple[str, str, str, bool]]:
    """The full factorial, 3 x 3 x 2 x 2 = 36 designs, base case first."""
    return list(itertools.product(PROFILE_LEVELS, INFORMER_LEVELS, ENDORSEMENT_LEVELS, MOA_LEVELS))


def planned_calls(llms: list[LLMs], designs: list[tuple[str, str, str, bool]], reps: int) -> int:
    """How many API calls a `run_pilot` with these arguments will make.

    Not simply designs x reps: every design but the base case is run once per
    sample, and the base case only once.
    """
    per_model = sum(1 if not (a or b or c) else 2 for a, b, c, _ in designs)
    return per_model * reps * len(llms)


def run_pilot(
    llms: list[LLMs] | None = None,
    designs: list[tuple[str, str, str, bool]] | None = None,
    reps: int = 1,
) -> list[Path]:
    """Every design on every model, `reps` times each, one CSV per pair.

    Defaults to the whole study: all three models over the full 36-design grid.
    Repetitions stack the same way they do for a single config, so re-running this
    adds `reps` more of everything rather than starting over.

    A model that is not wired up yet is reported and skipped. A design that fails
    outright is reported, and the run carries on to the next one -- an afternoon of
    calls should not be lost to a single bad configuration. Both are listed again
    at the end, so nothing that went wrong is only visible in scrollback.

    The bar counts API calls rather than designs, because that is the unit the wait
    is made of: a design is one or two calls depending on whether it renders per
    sample, so 36 of them are not 36 equal steps. Each design's calls are counted
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


def _draw_rates(ax, table: pd.DataFrame, designs: list[str], title: str) -> None:
    """One model's grid: two bars per design, one per sample, base case centred."""
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
    ticks = [f"{label}*" if label in unparsed else label for label in designs]

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

    Every design is one test of the same question, so the grid asks it 34 times per
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

    The base case is not tested. It renders identically for both samples, so it has
    only a `none` arm and there is nothing to compare -- which is what makes it the
    control for the other 34.

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
        floor = group["floor"].max()
        hits = group[group["significant"]]
        print(f"\n{model}: {len(hits)}/{len(group)} designs separate the samples at q < {alpha}")
        print(f"  smallest p these repetition counts can reach: {floor:.2g}")
        if floor > alpha:
            print("  -- that floor is above alpha: no design can come out significant. Add repetitions.")

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


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

DESCRIPTION = "The adoption-rate pilot: one prompt-design grid per model, its rates, and which designs separate the samples."

# The short name each model answers to on the command line, next to the value and
# the enum name -- `--models gpt` is what a run is actually typed as.
MODEL_ALIASES = {"gpt": LLMs.GPT_5_4_NANO, "haiku": LLMs.HAIKU_4_5, "grok": LLMs.GROK_4_2}

DEFAULT_FIGURE = FIGURE_DIR / "adoption_rates.png"


def resolve_model(label: str) -> LLMs:
    """`gpt`, `gpt-5.4-nano` or `GPT_5_4_NANO` -> the enum member."""
    key = label.strip()
    if key.lower() in MODEL_ALIASES:
        return MODEL_ALIASES[key.lower()]
    by_name = {llm.name.lower(): llm for llm in LLMs}
    if key.lower() in by_name:
        return by_name[key.lower()]
    return get_llm(key)  # raises with the list of valid values


def resolve_designs(labels: list[str] | None) -> list[tuple[str, str, str, bool]]:
    """Design labels back into the tuples `run_pilot` takes, in grid order.

    The label is the only handle a design has outside the code -- it is what the CSV
    filenames and the plot's ticks are written in -- so `--designs A1B0C1D0` is how
    one design out of the 36 is asked for by name.
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


def dry_run(designs: list[tuple[str, str, str, bool]], print_prompts: bool = False) -> int:
    """Render every design and call nothing: what `--live` would send, for free.

    The failure this catches is the one worth catching before paying for 70 calls: a
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
        help="design labels like A1B0C1D0 (default: the full 36-design factorial)",
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

    pl = sub.add_parser("plot", help="the adoption rate of every design that has been run, one row per model")
    pl.add_argument("--models", nargs="+", default=None, metavar="MODEL", help="default: every model in the logs")
    pl.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="where to read the CSVs from")
    pl.add_argument("--outfile", type=Path, default=DEFAULT_FIGURE, help=f"default: {DEFAULT_FIGURE}")

    rp = sub.add_parser("report", help="Fisher's exact test per design: which ones separate the two samples")
    rp.add_argument("--models", nargs="+", default=None, metavar="MODEL", help="default: every model in the logs")
    rp.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="where to read the CSVs from")
    rp.add_argument("--alpha", type=float, default=FDR_ALPHA, help=f"the FDR level (default: {FDR_ALPHA})")
    rp.add_argument("--csv", type=Path, default=None, help="also write the full test table here")

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

    # Both readers want one model or all of them, and `load_results` takes one.
    if models is not None and len(models) > 1:
        print("error: plot and report read one model at a time, or all of them if --models is omitted", file=sys.stderr)
        return 1
    only = models[0] if models else None

    try:
        results = load_results(only, output_dir=OUTPUT_DIR)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    rates = adoption_rates(results)

    if a.command == "plot":
        plot_adoption_rates(rates=rates, outfile=a.outfile)
        return 0

    tests = design_tests(rates=rates, alpha=a.alpha)
    significance_report(tests, alpha=a.alpha)
    if a.csv is not None:
        a.csv.parent.mkdir(parents=True, exist_ok=True)
        tests.to_csv(a.csv, index=False)
        print(f"\nwrote {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())