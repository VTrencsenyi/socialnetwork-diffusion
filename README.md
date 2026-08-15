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
| Agent + memory-state scaffolding (`HH_Agent`, `DefaultState`) | working, 21 tests passing |
| Descriptive & ground-truth figures | working |
| LLM decision layer, diffusion loop, baseline, validation | **not yet wired** |

`keys.example.json` is in place for the LLM layer; no module reads it yet.

---

## Repository layout

```
src/
  data_loader.py   load one village: network edges, leaders, MF outcome, covariates
  tools.py         .dta -> .csv conversion; builds the household feature table
  agent.py         HH_Agent — one household as an LLM agent
  state.py         what an agent remembers (AgentState / DefaultState / Turn)
  plots.py         network and take-up figures
  test.ipynb       scratch notebook
tests/             pytest suite for agent + state invariants
docs/
  household_design.md   why each household feature exists, and how it is built
CODEBOOK.md        every file and every column in the dataset, with provenance tags
figures/           generated figures (tracked — they are a deliverable)
output/            generated hh_features_<village>.csv (not tracked)
```

`CODEBOOK.md` is the place to start for anything about the raw data: it documents
each column and tags where the claim comes from (`[R]` package README, `[P]` the
paper, `[L]` Stata labels, `[S]` replication scripts, `[D]` direct inspection,
`[?]` genuinely undocumented).

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

# Build the household feature table (one CSV per village, into output/)
python -m src.tools hh-features                      # every eligible village
python -m src.tools hh-features --villages 1 24

# Convert Stata files to CSV
python -m src.tools dta2csv path/to/file.dta

# Figures (default: figures/village_<v>_<view>.png)
python -m src.plots --village 1 --view outcome   # take-up over the real network
python -m src.plots --village 1 --view hops      # network distance from the seeds
python -m src.plots --village 1 --view panels
python -m src.plots --view takeup                # take-up by village
python -m src.plots --view corr-all              # feature correlations

# Tests
python -m pytest tests/ -q
```

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
| `output/` | generated — rebuild with `python -m src.tools hh-features` |
| `keys.json` | secret |

`figures/` **is** tracked, since the figures are a deliverable that the write-up
refers to.

---

## Source

Banerjee, A., Chandrasekhar, A. G., Duflo, E., & Jackson, M. O. (2013). The
Diffusion of Microfinance. *Science*, 341(6144), 1236498.
Data: <https://web.stanford.edu/~jacksonm/Data.html>
