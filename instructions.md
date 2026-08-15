# Technical Assessment: LLM Agents for Social-Network Diffusion

## Objective

Build a small **agent-based model in which the agents are LLMs**, and use it to
simulate how a new financial product (microfinance) spreads through a **real
village social network**. Then do the thing that matters most to us: **check your
simulation against what actually happened.**

This is a deliberately open task. We are far more interested in how you reason —
about the model, the data, and especially the validation — than in how much you
build.

---

## Why this task

This is a stripped-down version of what we do at Electric Twin. We instantiate
populations of LLM agents from real data about real people, let them interact,
and use the collective behaviour to answer questions a survey can't. The hard
part is never getting agents to *do something* — it's knowing whether what they
did is *right*. This dataset is a rare case where there is a ground-truth answer
to compare against, which is exactly why we like it for this exercise.

---

## The Data

Use the **"Social Networks and Microfinance in Indian Villages"** data behind
*Banerjee, Chandrasekhar, Duflo & Jackson (2013), "The Diffusion of
Microfinance," Science*.

- Stanford landing page: https://web.stanford.edu/~jacksonm/Data.html
- The most current version is on Harvard Dataverse (linked from that page),
  with a README and replication code.

The original study: a microfinance institution entered 43 villages. Before
launching, it privately informed a set of **leaders** (teachers, shopkeepers,
self-help-group heads) — the *injection points* — and let participation spread
by word of mouth. The researchers had mapped each village's social network
beforehand, so they could study how take-up diffused. Roughly speaking, *who
the first informed people were* mattered as much as *how many* there were.

The bundle contains, per village:

- **Network data** — adjacency matrices for several relationship types
  (e.g. who borrows from / lends to / visits / takes advice from whom).
- **Household and individual attributes** — caste, religion, occupation,
  education, age, gender, dwelling characteristics, etc.
- **Microfinance participation** — which households actually joined (your
  ground truth).
- **Injection points** — which households were informed first (your seeds).

**Read the dataset's own README first.** Orienting yourself in unfamiliar
research data, and stating clearly which files and variables you used (and which
you ignored, and why), is part of the assessment. Exact schemas are documented
there — do not assume, check.

> Scope note: there are dozens of villages and some networks have hundreds of
> households. Running LLM agents over all of them is neither necessary nor
> sensible. **Pick one or two villages** (or a principled subsample) and keep
> the simulation tractable. Reasoning explicitly about this trade-off is a plus,
> not a corner you're cutting.

---

## The Task

### 1. Build the agents
Instantiate each household (or individual) in your chosen village(s) as an LLM
agent whose persona is grounded in its real attributes. An agent should be able
to decide, in character, whether it would join the microfinance programme given
what it knows and who has told it about it.

### 2. Wire up the network and run the diffusion
Use the **real network edges** as the channels along which information and
influence travel. Seed the simulation at the **empirical injection points**.
Step the model forward: informed/adopting agents can talk to their neighbours,
and neighbours decide whether to adopt. Record who adopts and when.

### 3. Validate against ground truth — the core of the task
Compare your simulated outcome to the **actual** microfinance take-up. That can
be the final adoption rate, the adoption *curve* over time, *which* households
adopt, the role of network position, or all of the above — you choose the
comparisons, but justify them. Be honest about where the model is right, where
it is wrong, and what you think is driving the gap.

### 4. Compare to a simple baseline
Stand your LLM model next to at least one **non-LLM baseline** (e.g. a simple
threshold / independent-cascade contagion model on the same network and seeds).
The question we want you to answer: **does making the agents LLMs actually buy
us anything here, and how would you know?**

### 5. (Stretch — reasoning, not delivery) Transfer
Briefly: is there comparable network + adoption + demographic data for a
**Western population**? We suspect there isn't much. We don't expect you to find
a perfect equivalent — we want your reasoning about what would and wouldn't
transfer, and what you'd need to commission to do this properly elsewhere.

---

## Deliverables

| # | Deliverable | Format |
|---|-------------|--------|
| 1 | Simulation code | `.py` / notebook |
| 2 | A short write-up (≈1–2 pages) of approach, results, and what you concluded | `.md` / `.pdf` |
| 3 | At least one figure comparing simulated vs. actual diffusion | image |
| 4 | README: how to run it, and what data files it expects | `.md` |

The write-up is as important as the code. Write it for a smart colleague who
wasn't in the room — lead with what you found and how much you trust it.

---

## What we're evaluating

- **Scientific judgement** — did you validate honestly against ground truth,
  pick sensible comparisons, and own the model's limits? (This carries the most
  weight.)
- **Modelling sense** — is the agent/decision design and the diffusion
  mechanism reasonable and clearly motivated?
- **Engineering** — does it run, is it readable, did you handle scale and LLM
  cost sensibly?
- **Communication** — can a non-technical stakeholder follow what you did and
  why it matters?

---

## Notes / constraints

- **Effort:** aim for roughly one focused day. A small, working, well-validated
  slice beats a sprawling unfinished system. If you run out of time, say what
  you'd do next.
- **LLM:** we use Claude (Anthropic) in production, so we'd love to see you use
  it, but use whatever you're fastest in. Agent loops make many calls — be
  mindful of cost (cache shared context, subsample, cap timesteps). Tell us
  roughly what your run cost.
- **Assumptions:** you will have to make some. State them. Documented
  assumptions are fine; silent ones are not.
- **AI tools:** use Cursor / Claude Code / Copilot freely — we do. We care about
  the thinking and the judgement, which they don't do for you.
- **Questions:** if something is genuinely blocking, ask. Knowing when to ask is
  itself a signal.
