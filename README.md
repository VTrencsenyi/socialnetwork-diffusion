# LLM Agents for Social-Network Diffusion

An agent-based model in which **the agents are LLMs**, simulating how microfinance
spread through **real village social networks** in rural Karnataka, India — and
validated against **what actually happened**.

The setting is the natural experiment behind Banerjee, Chandrasekhar, Duflo &
Jackson (2013), *"The Diffusion of Microfinance," Science* 341(6144):1236498. A
microfinance institution (BSS) entered 43 villages. Before launching, it
privately informed a set of **leaders** — the injection points — and let
participation spread by word of mouth. Because each village's social network had
been mapped ~6 months beforehand, we know both the seeds and the true outcome.
That makes it one of the few diffusion datasets with a real ground truth to score
a simulation against.

See [`instructions.md`](instructions.md) for the full task brief.

---

## Status

| Stage | State |
|---|---|
| Data loading, integrity checks, village descriptives | working |
| Household feature/persona table | working |
| Agent + memory-state scaffolding (`HH_Agent`, `DefaultState`) | working |
| Diffusion loop (`GameMaster`) + decision policies, no LLM | working |
| Descriptive & ground-truth figures | working |
| Profile rendering, both arms (`profiler.py`) | working |
| Subcaste spelling merge, pilot village (`subcaste.py`) | working |
| LLM take-up decisions, Stage I (`elicit.py`, `prompts.py`) | working, 217 tests passing; prompt wording is a draft |
| Ten-household toy sub-network, end to end (`subnetwork.py`) | working, run live on village 6 |
| Replicates + the run record (`experiment.py`, `runlog.py`) | working, not yet run live |
| Adoption-rate prompt-design pilot (`pilot/adoption_rate_pilot.py`) | prompts, runner and analysis working end to end on gpt-5.4-nano; haiku and grok raise `NotImplementedError` |
| DT prompting in the pilot (design axis `D2`) | working on gpt-5.4-nano, under a strict response schema; the schema is an OpenAI Responses feature, so the other two providers need a decision when they are wired up |
| LLM transmission (Stage II), decision-theoretic prompting (axis A2) | stubs that raise; the log schema is ready for them. The pilot's `D2` is where the state space, action set and cell semantics are being settled empirically — `elicit.py`'s A2 stays shut until they are |
| Baseline port, scoring against ground truth | **not yet built** |

`profiler.py` and `elicit.py` read `keys.json`; everything else runs without one. The
adoption-rate pilot will be the only thing here that talks to more than one provider:
it reaches all three models through the OpenAI Responses API, from the `openai`,
`claude` and `grok` blocks of the same file.

**`output/profiles/profiles_<village>.json` is the single source of truth for profiles.**
`game_master.from_village()` reads it and pushes the chosen arm into each agent with
`set_context()`, so no `output/context/context_<hhid>.txt` file is involved any more.

---

## Repository layout

```
src/
  data_loader.py   load one village: network edges, leaders, MF outcome, covariates
  tools.py         .dta -> .csv conversion; builds the household feature table
  subcaste.py      curated per-village subcaste alias maps; writes CLEANED_hh_features_<v>.csv
  agent.py         HH_Agent — one household as an LLM agent
  state.py         what an agent remembers (AgentState / DefaultState / Turn)
  game_master.py   the diffusion loop: seeds, rounds, delivery, run log
  subnetwork.py    carve a small toy village out of a real one, and run it
  policy.py        how an agent decides (dummy / never / always; BCDJ transmission)
  elicit.py        the LLM take-up decision: sampling, parsing, caching, the run artefact
  experiment.py    the same configuration S times over, and the average of it
  runlog.py        what a run writes down: one directory, tables + hash-keyed text
  prompts.py       every word the decision agent reads — the instrument, in one file
  llm.py           the provider seam: client, retries, token accounting
  pilot/
    adoption_rate_pilot.py  the adoption-rate prompt-design pilot, whole: the
                   instrument's wording, the four sample households, the four
                   prompt axes (the last of them plain / MOA / decision-theoretic),
                   the model list, the run, and the analysis — adoption rates with
                   SE bars, Fisher's exact test per design, and the elicited
                   decision matrices behind the DT designs
  profiler.py      the profile layer: static fact listings and LLM narratives
  plots.py         network and take-up figures
  test.ipynb       scratch notebook
tests/             pytest suite for agent, state and game-master invariants
docs/
  household_design.md   why each household feature exists, and how it is built
  experiment_design.md  the three-stage / two-axis experiment plan; pilot village 6 (design only, not built)
CODEBOOK.md        every file and every column in the dataset, with provenance tags
figures/           generated figures (tracked — they are a deliverable)
output/            generated data (not tracked)
  features/        hh_features_<village>.csv, one per village, plus
                   CLEANED_hh_features_<village>.csv where a subcaste alias map exists
  profiles/        profiles_<village>.json — THE artefact for anything profile-related:
                   traits, static profile, prompt, narrative profile and token counts
                   (profiles_<village>_explicit.json for the explicit leader mode)
  toy/             a carved sub-network as an ordinary village: network_<toy>.json,
                   features/hh_features_<toy>.csv, profiles/profiles_<toy>*.json
  runs/            run_v<village>_<arm>_k<k>_seed<seed>.json + .elicitation.json
                   — one run, one file; what a single `game_master` invocation writes
  experiments/     v<village>_<arm>_k<k>_S<reps>_seed<seed>/ — S replicates as a
                   directory of tables plus the prompts and responses behind them
  pilot/           the adoption-rate pilot: <llm>_<design>.csv, one row per
                   repetition; a second run appends rather than overwrites
report/            LaTeX scaffold for the technical report
```

`CODEBOOK.md` is the place to start for anything about the raw data: it documents
each column and tags where the claim comes from (`[R]` package README, `[P]` the
paper, `[L]` Stata labels, `[S]` replication scripts, `[D]` direct inspection,
`[?]` genuinely undocumented).

---

## Technical report scaffold

A LaTeX scaffold for the write-up lives in `report/`:

```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The scaffold is split into section files for the report's `Data`, `Agents`,
`Interaction model`, `Experiments`, and `Extensions` sections, with a starter
bibliography in `references.bib`.

---

## Setup

Requires **Python 3.12** (developed on 3.12.2).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Data setup

The raw data (~2 GB) is **not in this repository** — it is publicly
redistributable from the authors, so the repo links it rather than vendoring it.

1. Download the replication package **"Social Networks and Microfinance in Indian
   Villages"**, version **4.0** (dated 16 September 2013), from the Stanford
   landing page <https://web.stanford.edu/~jacksonm/Data.html>, which links the
   current copy on Harvard Dataverse.
2. Unpack it so that `datav4.0` sits inside `diffusion-science-data/` at the repo
   root:

```
diffusion-science-data/
└── datav4.0/
    ├── Data/
    │   ├── 1. Network Data/
    │   └── 2. Demographics and Outcomes/
    │       ├── household_characteristics.dta
    │       └── individual_characteristics.dta
    ├── Matlab Replication/
    │   └── India Networks/          # adj_*, key_HH_*, MF*, HHhasALeader*, inGiant*, hhcovariates*
    ├── Stata Replication/
    │   └── data/
    │       ├── panel.dta
    │       └── cross_sectional.dta
    └── README.pdf
```

That default path is `data_loader.DEFAULT_ROOT`; every CLI also takes `--root` if
you keep the data elsewhere.

The specific files the code reads:

| File | Used for |
|---|---|
| `Matlab Replication/India Networks/adj_<network>_HH_vilno_<v>.csv` | household adjacency matrix |
| `Matlab Replication/India Networks/key_HH_vilno_<v>.csv` | row index -> household ID |
| `Matlab Replication/India Networks/MF<v>.csv` | **ground truth**: household take-up |
| `Matlab Replication/India Networks/HHhasALeader<v>.csv` | **seeds**: injection points |
| `Matlab Replication/India Networks/inGiant<v>.csv` | giant-component membership |
| `Matlab Replication/India Networks/hhcovariates<v>.csv` | household covariates |
| `Data/2. Demographics and Outcomes/household_characteristics.dta` | roof/rooms/beds, electricity, latrine |
| `Data/2. Demographics and Outcomes/individual_characteristics.dta` | caste, occupation, SHG/savings, education |
| `Stata Replication/data/panel.dta` | take-up over time (adoption curve) |
| `Stata Replication/data/cross_sectional.dta` | the paper's 43-village analysis sample |

### API keys

The LLM layer reads credentials from `keys.json`, which is **gitignored**. Copy
the template and fill it in:

```bash
cp keys.example.json keys.json
```

Never commit `keys.json`.

---

## Running

All commands run from the repository root.

```bash
# List villages that have a household-level microfinance outcome
python -m src.data_loader --list

# Describe one village: network, seeds, ground-truth take-up, integrity checks
python -m src.data_loader --village 1
python -m src.data_loader --village 24 --network allVillageRelationships

# Build the household feature table (one CSV per village, into output/features/)
python -m src.tools hh-features                      # every eligible village
python -m src.tools hh-features --villages 1 24

# Merge subcaste spelling variants for a reviewed village (docs/experiment_design.md §7.1)
# Writes output/features/CLEANED_hh_features_<v>.csv, keeping the raw string in subcaste_raw
python -m src.subcaste --villages 6
python -m src.subcaste --villages 6 --drop-in -o output/features_clean   # as hh_features_6.csv too,
                                                                        # so --features-dir switches over

# Convert Stata files to CSV
python -m src.tools dta2csv path/to/file.dta

# Figures (default: figures/village_<v>_<view>.png)
python -m src.plots --village 1 --view outcome   # take-up over the real network
python -m src.plots --village 1 --view hops      # network distance from the seeds
python -m src.plots --village 1 --view panels
python -m src.plots --view takeup                # take-up by village
python -m src.plots --view corr-all              # feature correlations

# Build household profiles (two arms; see docs/household_design.md §5.1)
# Both write output/profiles/profiles_<village>.json — one artefact, keyed by hhid.
python -m src.profiler --mode facts --villages 73 67       # traits + static profile; free, no API key
python -m src.profiler --mode story --villages 73          # dry-run: shows the prompts, costs nothing
python -m src.profiler --mode story --villages 73 --live   # + the LLM narrative and token counts

# Run the diffusion simulation, mechanics only (no LLM, no API key)
python -m src.game_master --village 6 --adoption dummy --seed 0
python -m src.game_master --village 6 --adoption always   # reachability oracle
python -m src.game_master --village 6 --adoption never    # the empty run

# The pilot: LLM take-up, BCDJ transmission at (qN, qP) = (0.09, 0.45)
python -m src.game_master --village 6 --arm static             # dry run: prints one full prompt, costs nothing
python -m src.game_master --village 6 --arm static --live      # the real thing
python -m src.game_master --village 6 --arm narrative --live   # the other side of axis B
python -m src.game_master --village 6 --arm static --k 5 --seed 1 --live
python -m src.game_master --village 6 --arm static --leader-mode explicit --live   # the other profiles file
# -> output/runs/run_v6_<arm>_k<k>_seed<seed>.json  + .elicitation.json beside it

# The ten-household toy: carve a sub-network out of village 6 and run the same pilot on it
python -m src.subnetwork extract --village 6 --list                       # every leader's candidate, and why
python -m src.subnetwork extract --village 6 --leader-mode explicit       # -> output/toy/...
python -m src.subnetwork run --village 6 --leader-mode explicit           # dry run: one prompt, no cost
python -m src.subnetwork run --village 6 --leader-mode explicit --live    # ~10 calls
python -m src.subnetwork run --village 6 --adoption always --qn 1 --qp 1  # the toy's reachability oracle, free

# Replicates: the same configuration S times, averaged (src/experiment.py)
python -m src.experiment --village 6 --adoption dummy --reps 20          # free, no API key
python -m src.experiment --village 6 --arm static --reps 10              # dry run: the cost of S
python -m src.experiment --village 6 --arm static --reps 10 --live       # the pilot, ten times
python -m src.experiment --village 6 --arm static --reps 10 --refresh-elicitation --live  # re-elicit every replicate
python -m src.game_master --village 6 --arm static --reps 10 --live      # the same thing from the runner
python -m src.subnetwork run --village 6 --leader-mode explicit --reps 10 --live   # ~10-30 calls
# -> output/experiments/v6_static_k1_S10_seed0/

# The adoption-rate pilot: which prompt design separates an adopter from a non-adopter
python -m src.pilot.adoption_rate_pilot run --models gpt                    # dry run: all 105 prompts, no cost
python -m src.pilot.adoption_rate_pilot run --models gpt --print-prompts    # the same, each one in full
python -m src.pilot.adoption_rate_pilot run --models gpt --live             # 105 calls, one rep of the grid
python -m src.pilot.adoption_rate_pilot run --models gpt --reps 10 --live   # the study: 1050 calls
python -m src.pilot.adoption_rate_pilot run --designs A1B0C1D0 A2B2C1D1 --reps 5 --live   # two designs only
python -m src.pilot.adoption_rate_pilot report            # Fisher's exact per design, BH-corrected
python -m src.pilot.adoption_rate_pilot plot              # -> figures/pilot/adoption_rates.png
python -m src.pilot.adoption_rate_pilot dt                # the D2 designs: the matrices behind the decisions
# -> output/pilot/gpt_5_4_nano_<design>.csv, one row per call. Repetitions stack:
#    re-running --reps 10 asks for ten more, it does not start over.

# Read one back
python -m src.experiment show output/experiments/v6_static_k1_S10_seed0
python -m src.experiment show output/experiments/v6_static_k1_S10_seed0 --household 6003
python -m src.experiment show output/experiments/v6_static_k1_S10_seed0 --transcript --replicate 3

# Tests
python -m pytest tests/ -q
```

### The simulation

`game_master.py` runs one village for `T` rounds, where `T` defaults to the
village's last trimester in `panel.dta` — how long BSS was actually in it (5 for
village 6, 6 for village 24). Rounds are 1-based and there is no round 0: the
MFI seeding the leaders is the opening move *of* round 1.

The loop is BCDJ's `diffusion_model.m`, transliterated, because
`docs/experiment_design.md` §1.3 requires it be held constant across the stages
the LLM substitutes into:

```
contagious = the leaders, before round 1
for r in 1 .. T:
    step 1  every household informed *before this round* that has not yet
            decided decides, once, and is never asked again
    step 2  every household ever informed tries every neighbour, independently,
            at a node-level rate q_i = qP if it has joined, else qN
    step 3  whoever was reached becomes informed, and decides next round;
            advance() every agent in lockstep
then    stop -- or, with --sweep, one terminal decision round in which everyone
        informed but never asked gets their decision
```

Four properties of that loop are the model rather than incidental, and each has
a test: **take-up is one-shot**, at the moment of first hearing — a household
that declines is never asked again, however many neighbours pitch it later
(asking every round is a different estimand, and is §9.1 of the design, not the
spine); **hearing and deciding are one period apart**, so information moves one
hop per round and decisions trail it by one; **an adopter already transmits at
`qP` in the period it joins**, because step 1 updates the taker set before step 2
reads it; and **the transmission rate is node-level**, identical along every edge
out of *i*, which is exactly what Stage II is meant to be able to beat.

The one available departure from the Matlab is the terminal sweep, and it is
**off by default**. BCDJ stop at `T`, which leaves the households reached in the
final period informed but never asked — and in their code that final round of
transmission is inert, because `divergence_model.m` scores `infected` only and
never looks at `contagious`. `--sweep` closes those decisions without adding a
round of information flow (`RoundRecord.sweep` marks it, and the pre-sweep level
is still the round-`T` line of the log). It costs one paid call per household it
reaches, which is why the default is BCDJ's behaviour: on village 6 at
`(0.09, 0.45)` the sweep asked a mean 4.2 households a run, of whom 1.3 joined.

A household is asked at most once, ever, and never once it has joined — so the
call budget for a run is bounded by the number of *informed* households, not by
`T`.

`--adoption always --qn 1 --qp 1` is a useful oracle rather than a model: it
informs everything reachable, so its adopters are exactly the households within
`T` hops of a seed. On village 6 it reaches 107 of 114 — the 7 isolates are
unreachable by construction and never adopt, which is
`docs/household_design.md` §7.2's free correctness check, and it is the recall
ceiling `docs/experiment_design.md` §5.2 reports as 25 of 25 adopters.

Nothing in `game_master.py` reads `MF<v>.csv` or the `_adopted` column, and
neighbour profiles are rendered through a `usecols` allowlist so the outcome is
never loaded into the process at all. Scoring against ground truth is a separate
step that does not exist yet.

### The LLM decision layer

Stage I of `docs/experiment_design.md`: take-up is the agent's, transmission
stays at BCDJ's fitted `(qN, qP)`. Three files, one job each — `prompts.py` is
every word an agent reads, `elicit.py` samples and parses and caches,
`llm.py` is the provider.

The pilot runs the **informer-aware** information set (I-b): the prompt carries
the household's own profile, the programme description, and who told it what,
rendered from the agent's own ledger. The message itself says only that the
programme was mentioned — whether the speaker joined is left unspecified, since
mode 2a's message has no author and stating it on every edge would build BCDJ's
endorsement model into the mechanics. `--endorsement` turns it on as an explicit
assumption. The own-profile-only arm (I-a) is a
parameter that raises rather than a path that has been built, as are LLM
transmission (Stage II) and decision-theoretic prompting (axis A2).

`k` samples per decision at temperature 0.9 give `p̂` = the fraction answering
join, and one keyed Bernoulli draw turns that into the action; at the pilot's
`k = 1` the model's single answer *is* the decision. The only constraint the
instruction places on a response is that `(Y)` or `(N)` be the first or last
thing written — nothing about length, form or reasoning, because that would be
an intervention on the decision rather than on the parser.

Every decision is cached on a hash of the rendered request, so two households in
the same position share one elicitation and a later replicate re-uses whatever
its predecessors paid for. Each run writes the log and, beside it, an
`.elicitation.json` holding every prompt, every raw response and every token
count behind it.

**Open: the model does not always write the brackets.** On the first live toy run
(`gpt-5.4-nano`, temperature 0.9) two of nine decisions came back as a bare `N` —
once as the single character — and never wrote `(N)` in six attempts, so
`ElicitationError` stopped the run. It is a systematic property of those prompts,
not a transient: retrying does not fix it. `--on-parse-failure decline` is the
existing escape hatch and is what the recorded runs used; a lenient reading of
both failures was `N`, i.e. what `decline` assigned, so those runs' decisions are
unaffected — but as a rule it silently turns an unparseable *join* into a
refusal, which is a bias in the direction of the null. Two ways out, and it is a
decision for the instrument rather than a bug to patch quietly: loosen
`elicit.parse_answer` to accept a bare leading/trailing `Y`/`N`, or tighten the
wording in `prompts.SYSTEM_PROMPT`. `parse_failures` is in every run's
`elicitation` block, so the rate is on the record either way.

### Replicates, and the run record

One run of this model is a **draw, not a result**: transmission is a Bernoulli
draw per ordered edge per round, so who hears anything at all — and therefore who
is ever asked — is different every time. On village 6 with the dummy policy, five
replicates of one configuration ran from 3 to 18 adopters. The quantity the
design wants is the **adoption rate per household across replicates** and the
mean village curve, and neither exists until S > 1.

`experiment.py` runs one configuration S times. Replicate *s* is that
configuration at `seed = base_seed + s`, which keys both draws in the model
(transmission, and the Bernoulli against `p̂`); agents are rebuilt every replicate
because a ledger *is* the run.

**The elicitation is held fixed across replicates, deliberately.** One elicitor
serves the whole experiment and its cache spans it, which is what
`docs/experiment_design.md` §4.2 asks for: `p̂` is paid for once and the S
replicates cost almost nothing, so the run-to-run spread is the mechanics rather
than the model's sampling noise. Two consequences worth saying out loud: at
`k = 1` a household that reaches an identical position in two replicates makes an
identical decision, so **the spread is not a confidence interval on the LLM's
answer**; and `--refresh-elicitation` buys the other reading — every decision
re-sampled every replicate, the spread including the model's own variance, at S
times the cost. On village 6 the cache is worth having: across 4 replicates
against a stub model, replicate 0 paid for 81 decisions and replicate 3 for 34.

A replicate that raises stops the experiment by default (`--continue-on-error` to
carry on), and either way it is **not averaged into anything** — a run that
stopped in round 3 has a real adopter set that is simply not the quantity the
other replicates measured. `households.csv` writes its denominator into every row
for the same reason.

#### What a run writes down

One directory per experiment. Three kinds of file, one job each: **tables** for
the things evaluation counts, **hash-keyed JSONL** for the text that cost money,
and a **document** for a human.

```
output/experiments/v6_static_k1_S10_seed0/
  manifest.json          what was run, per replicate, and the aggregate
  rounds.csv             replicate × round: asked, joined, messages, informed, adopters
  curves.csv             round: mean and sd of the informed and adopted shares
  households.csv         household: adoption_rate across replicates  ← the headline
  decisions.csv          replicate × household: the one decision, and when
  messages.csv           replicate × round × edge: who told whom
  messages.legend.json   msg_id → the text that passed along the edge
  elicitations.csv       replicate × decision: prompt_sha, p̂, k, the k answers
  prompts.jsonl          prompt_sha → the exact request sent to the model, once
  calls.jsonl            every call: the raw response, what it parsed to, what it cost
  transcript.md          one replicate in reading order, for a human
```

Three properties this shape is for, none of which the single-run JSON has:

- **A crash must not cost the calls.** Every stream is opened at the start and
  appended to as the run goes, so a run that dies in replicate 7 keeps the six
  that are paid for. The old artefact was written only at the end — and since the
  README's open question about missing brackets is a failure that *aborts a run*,
  that is not a theoretical concern.
- **The unit of analysis is a row.** `households.csv` and `rounds.csv` load with
  `pd.read_csv` and need no bespoke parser.
- **Text is stored once, keyed by hash.** A prompt appears in `prompts.jsonl`
  once however many decisions and replicates it served — the same
  `sha256(request)[:16]` `elicit.py` caches on. Every attempt is in `calls.jsonl`,
  including the responses that failed to parse and were retried, because those
  are the evidence for the bracket question rather than noise to be dropped.

The files divide into **facts of the simulation** — rounds, decisions, messages,
households — which every policy produces, LLM or not, and the **instrument** —
elicitations, prompts, calls — which exists only where a model was asked
something. A `--adoption dummy` run writes the first group and leaves the second
empty, which is a true statement about that run rather than a gap in it.

The join keys are the same everywhere, which is what makes the split cost
nothing:

```
decisions      (replicate, round, hh_id)
elicitations   (replicate, round, hh_id) -> prompt_sha
prompts        prompt_sha
calls          prompt_sha, sample, attempt
messages       (replicate, round, src, dst)
```

That last line is what **Stage II** needs. When transmission becomes the model's
decision rather than a Bernoulli draw, the inform prompt and its response go into
the same `prompts.jsonl` and `calls.jsonl`, and the elicitation row carries
`kind = "inform"` with `target_id` set to the household being told — two columns
that Stage I leaves empty and Stage II fills in. Nothing about the schema
changes; `elicit.LLMTransmission`'s docstring states the three calls it will have
to make, and a test asserts the format takes them.

`show` is what hash-keyed storage owes a human: `--household 6003` reconstitutes
every prompt that household was given and every answer it gave, and
`--transcript` renders a replicate round by round. Both are built from the
written files alone, so what they can show is exactly what the directory holds.

### The toy sub-network

`subnetwork.py` carves an **induced** subgraph out of a real village and writes it
as an ordinary village of its own, so the whole pipeline — the same prompts, the
same elicitor, the same BCDJ transmission, the same `T` — runs end to end on ten
households for about a tenth of the calls. It is a fixture for reading a whole
run by hand, not a village: ten households carry no estimate of anything.

The selection rule, in full: start from a leader household, then repeatedly add
the nearest household that is not a leader and that already has an edge into the
set (BFS distance from the leader, ties by hhid). Everything after the first
joins by an edge, so the result is connected — the useful form of "no isolates".
Other leaders are skipped rather than stopping the growth, which is what keeps
the seed set at exactly one in a village where 22 of 114 households are leaders.
Which leader seeds it, when `--leader` is not given: among the leaders that yield
a full-size set, keep those whose eccentricity inside the subgraph is at least 2
(so the seed is not adjacent to everyone and information needs more than one
hop), then take the one with the most induced edges. `--list` prints the whole
table so the choice can be checked rather than taken on trust.

**Nothing in the selection reads the outcome** — adjacency and `has_leader` only,
with the feature table read through a three-column `usecols` allowlist. Choosing
a fragment on `_adopted` would be selecting the sample on the dependent variable,
and a test asserts the same carve comes out of a feature table with the column
deleted.

Household facts and personas are copied from the parent verbatim; `row`, `degree`
and `in_giant` are recomputed on the induced graph, because a stale degree column
would describe a village that is not the one being run. `hhid` is never
renumbered — it is the join key for the profiles and the only honest provenance
for a row. `T` is inherited from the parent's `panel.dta`.

The carve of village 6 at `--size 10` is toy village **906**: seed 6003, 21 edges,
degree 2/4.2/8, one household two hops out, ground-truth take-up 1 of 10.

`--view` accepts `outcome`, `hops`, `panels`, `all`, `takeup`, `horizon`, `corr`,
`corr-all`. `--network` selects which of the 12 surveyed relationship layers (or
the two combined layers) to use as the diffusion channel.

---

## Notes on the data

Two things worth knowing before reading any number out of this repo:

- **75 villages numbered 1–77; villages 13 and 22 do not exist.** 43 of the 75
  are the paper's analysis sample. `data_loader --list` reports 49 villages with
  a household-level outcome, six of which (10, 28, 37, 41, 66, 77) are *outside*
  that published sample — do not mix them in without saying so.
- **Two different denominators for take-up.** `panel.dta` is computed on the full
  village census; `MF<v>.csv` is computed on the network sample (~46% of
  households were individually surveyed). They do not agree, and
  `check_consistency()` will flag it. The paper's outcome uses the
  `cross_sectional.dta` denominator.

---

## What is not committed

| Path | Why |
|---|---|
| `diffusion-science-data/` | ~2 GB; publicly re-downloadable (see Data setup) |
| `papers/` | published PDFs, not ours to redistribute |
| `output/` | generated — rebuild features with `python -m src.tools hh-features`, profiles with `python -m src.profiler` |
| `keys.json` | secret |

`figures/` **is** tracked, since the figures are a deliverable that the write-up
refers to.

---

## Source

Banerjee, A., Chandrasekhar, A. G., Duflo, E., & Jackson, M. O. (2013). The
Diffusion of Microfinance. *Science*, 341(6144), 1236498.
Data: <https://web.stanford.edu/~jacksonm/Data.html>
