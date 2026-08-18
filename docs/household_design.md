# `Household`: design plan for the data component

Status: the reduced feature set of §4 is decided and built —
`tools.build_household_features()` writes it to `output/features/hh_features_<village>.csv`. The
persona rendering, the decision loop and the prompt are still open. Scope: **what data a
household agent carries and why** — not yet how it decides, talks, or is prompted.

Everything below is checked against the v4.0 bundle rather than assumed. Counts in
parentheses are what the files actually contain.

---

## 1. What the object is for

`Household` is the agent. Its data serves three consumers with different needs, and the
whole design turns on keeping them apart:

| Consumer | Needs | Cost of getting it wrong |
|---|---|---|
| **Persona** — text handed to the LLM | a small set of human-legible, decision-relevant attributes | tokens are spent per agent per timestep; noise dilutes signal |
| **Mechanics** — the diffusion loop | network position, seed status, mutable state | misalignment silently produces meaningless results |
| **Evaluation** — post hoc | the privileged target, plus covariates to slice by | leakage makes the validation worthless |

**The one non-negotiable:** the target must be *structurally* unable to reach the prompt,
not merely omitted by convention. Concretely: outcome fields live behind a private
attribute, `to_persona()` renders from an explicit allowlist of field names, and a test
asserts the rendered string is byte-identical for two households that differ only in
their outcome. "We were careful" is not a control; a test is.

---

## 2. A finding that amends the spec: there is no household-level trimester

The brief asks for "which trimester it joined (or 0 if it did not)". **That variable does
not exist in the public bundle.**

- The only household-level outcome is `MF<V>.csv` — a plain 0/1 vector, `n` rows. BCDJ's
  own replication code loads exactly this file as `TakeUp`
  (`Matlab Replication/GMMDiffusion/Main_models_1_3.m:106`).
- Timing exists **only at village level**, in `Stata Replication/data/panel.dta`, where
  `t` indexes trimesters since the MFI entered and `dynamicMF_empirical` is the observed
  cumulative take-up. A trimester is 4 months: `T = ceil(months/4) + 1`
  (`Main_models_1_3.m:57`). Villages run 3–11 trimesters; our candidates 6–9.
- `household_characteristics` has no microfinance column, despite what README §3.4 says
  (already noted in `data_loader.py`).

So the honest encoding is a single column:

```
_adopted: int   # 0/1, privileged, from MF<V>.csv -- and that is the whole of it
```

An earlier draft carried `joined_trimester` alongside it, `0` for non-adopters and unknown
for adopters, to preserve the API the brief asked for. That column is **gone**: it would
have been empty for every household that actually joined, which is precisely the half a
timing variable exists to describe. A column that is unknown exactly where it matters is
worse than no column — it invites a later evaluation to fill the blanks with `0` and
silently treat every adopter as a non-adopter. The *simulation* will still produce a
per-household adoption time; it simply has nothing per-household to be scored against.

**Consequence for validation** — this splits the target in two, and each half is scored
against a different file:

- *Who* adopts → household-level, against `MF<V>.csv`. Confusion matrix, AUC, take-up by
  degree band and by distance-to-seed.
- *When* adoption happens → village-level only, against the panel curve. Per-household
  simulated trimesters are aggregated to a curve and compared in shape.

`Village.adoption_curve()` already returns `empirical_rescaled` for exactly this, so the
two targets sit on one population. Simulated adoption times therefore live in the
simulation's own state, aggregated to a curve for scoring — not as a column of the
household table, which holds only what the data can support.

---

## 3. The dominant constraint: the individual survey is a non-random half

This is the finding that should drive the feature set, and it is easy to walk past.

`individual_characteristics` covers only **36.5–54.0% of network households** (median 46.8%).
`hhSurveyed == 1` is exactly equivalent to appearing in that file — verified, 0
disagreements across all 75 villages — so one flag tells you which agents can have a rich
persona. Two ways that subsample is not random:

1. **Degree.** Surveyed households are far better connected. In all **43/43** analysis
   villages, surveyed mean degree exceeds non-surveyed by more than 2×; mean ratio
   **3.41×** (range 2.67–5.14). Village 73: 16.5 vs 5.9. This is a design artefact, not a
   discovery — the network was elicited *from* surveyed households, so non-surveyed ones
   appear only as named alters and their degree reflects one-sided reporting.
2. **The outcome itself.** Take-up is higher among surveyed households: pooled 19.3% vs
   16.9%; per-village gap mean **+2.4pp**, positive in 30/49 villages, one-sample
   *t* = 2.77, *p* = 0.008. Restricted to non-leaders: +2.9pp, *p* = 0.003.

Put together: the households we can describe richly are also the better-connected and
more-likely-to-adopt ones. If persona richness varies with `hhSurveyed`, then any accuracy
the LLM model shows is confounded with network position and with the target — and the
headline question ("does making the agents LLMs buy anything?") becomes unanswerable.

**Three rules follow, and they are the core of this proposal:**

- **R1 — Uniform schema.** Every household carries every field. Individual-derived fields
  get an explicit `unknown`, and the persona template *says* "not known" rather than
  dropping the line. Otherwise prompt length itself is a covariate correlated with the
  target.
- **R2 — Headline model runs on the universally-available block only** (household file +
  derived size + network position). The individual-derived block is a second arm, run as
  an ablation. That is what isolates "richer persona → better fit" from "better-connected
  household → more exposure". Under the reduced set adopted in §4 that ablation is two bits
  and one categorical — `has_shg`, `has_savings` and `subcaste` (§4.3b). Narrow enough to
  interpret, unlike the seven-field version it replaces, but no longer so narrow that a
  null result would be uninformative. If the individual-derived arm fits better, the
  categorical and the bits have to be separated before the win is attributed.
- **R3 — Never select features by correlation with `MF<V>.csv`.** Inclusion is justified
  by coverage, by theoretical relevance to a borrowing decision, and by BCDJ's own
  choices — never by peeking at the outcome. Otherwise we hand-fit the personas to the
  answer and the validation is circular.

---

## 4. Feature inventory

### 4.0 The adopted set

Thirteen fields, all of them either universally available or explicitly flagged as
survey-only. This is deliberately narrower than the inventory that follows: §4.5 records
what was dropped and what the drop costs, so the reduction is a decision on the record
rather than an omission.

| Field | Source | Coverage | Note |
|---|---|---|---|
| `religion` | `hohreligion` | 100%, 0 NaN | Persona realism only. **Zero within-village variance in villages 73, 67, 62, 68** (all 100% Hinduism); 40 of 75 villages are single-religion. Not a feature — say so before, not after. |
| `rooms` | `room_no` | 100% | BCDJ covariate. |
| `beds` | `bed_no` | 100% | BCDJ covariate, but **`beds == 0` for 50.8% of the bundle** (70% in village 73, 75% in village 68) — near-binary in practice. It means no cot or bedstead, a real poverty marker; the persona must word it so an LLM does not read it as a broken record. |
| `capita` | derived, §4.4 | 98.9–100% in the candidates | Household size. Not in any published file. |
| `rooms_per_capita`, `beds_per_capita` | `hhcovariates<V>.csv` cols 5–6 | as `capita` | BCDJ covariates. Exactly `rooms / capita` and `beds / capita`, so **carry them for the baseline, not for the prompt** — three columns for one fact costs tokens per agent per timestep. |
| `electricity` | `electricity` | 100% (3 NaN in 14,904) | Ordinal 0/1/2 — §4.1. |
| `own_latrine` | `latrine` | 100% (3 NaN) | Binary, **not** the 3-level ladder it looks like — §4.2. **`"None"` is a genuine category, no latrine, not a missing value.** Verified against raw Stata codes: 10,930 households genuinely have none. A naive `isna`-plus-string check reports 73% missing and throws away the sharpest wealth marker in the file. |
| `has_leader` | `leader` | 100% | The seed set. Not privileged (§4.7). |
| `surveyed` | `hhSurveyed` | 100% | Research artefact — defines the R2 arm, **never rendered into a persona**. |
| `subcaste` | `individual.subcaste` | surveyed households only | The social identity an agent can actually reason with — §4.3b. Raw strings, 12–17 levels per village, deliberately not normalised. |
| `occupation_head` | `individual.occupation` | surveyed households only | Raw free text, deliberately not bucketed — §4.3c. Unlike `subcaste`, kept for narrative variance in generated biographies, not as a clean categorical. |
| `has_shg`, `has_savings` | `shgparticipate`, `savings` | surveyed households only | Tri-state. Which household members' answers these aggregate is a run parameter — §4.3. |

Rendered by `tools.build_household_features()` to `output/features/hh_features_<village>.csv`, one
row per household, row *i* = adjacency row *i*. Unknown is empty in the CSV — never `0`,
never `False` — which is why every integer column is a nullable `Int64` and the two survey
flags are a nullable `boolean`.

### 4.1 Electricity is genuinely ordinal

Coded `0` none → `1` government connection → `2` private connection. Checked bundle-wide
against three independent wealth proxies:

| electricity | n | mean rooms | mean beds | owns a latrine |
|---|---|---|---|---|
| No | 1,141 | 1.617 | 0.341 | 4.5% |
| Yes, Government | 4,565 | 1.948 | 0.477 | 8.8% |
| Yes, Private | 9,195 | 2.659 | 1.096 | 37.3% |

Monotone on all three, so the integers are a gradient rather than a convenience coding.
Two caveats to carry: it is a connection *type* — government schemes target BPL households
— and **collapsing to yes/no is strictly worse**, because 92.3% of the bundle is "yes" and
the binary merges the *larger* of the two contrasts (Govt→Private, +0.71 rooms) while
keeping the smaller (None→Govt, +0.33).

### 4.2 Latrine is not — hence `own_latrine`

The obvious coding (0 none → 1 common → 2 owned) fails its own premise:

| latrine | n | mean rooms | mean beds |
|---|---|---|---|
| None | 10,930 | 2.095 | 0.612 |
| **Common** | **84** | **2.024** | **0.571** |
| Owned | 3,887 | 3.117 | 1.520 |

`Common` sits *below* `None` on both proxies, not between them — a shared latrine marks a
poor, dense settlement, not a rung up from open defecation. And it barely exists: 84
households bundle-wide, **1 in village 73, 0 in villages 75 and 68**, at most 15 in any
village. So the middle rung is both empty and in the wrong place, and the only contrast
the data supports is binary: `own_latrine = 1` for Owned, `0` for None **or** Common.

### 4.3 `has_shg` / `has_savings`: whose answer this is

`shgparticipate` and `savings` are individual-level, and the individual file holds
**2.46 interviewed adults per household** against a recovered size near 5 (village 73:
2.31 vs 5.31). There is therefore no "all members" option available — only a choice of
which interviewed adults to read. The column names are deliberately scope-neutral: which
adults were read is a run parameter, not a property of the schema, so the CSV keeps the
same header whichever way `--shg-scope` is set and the choice is recorded here rather than
smuggled into a column name. The choice is not cosmetic:

| scope | v73 SHG | v67 SHG | v73 savings | v67 savings | coverage of surveyed hh |
|---|---|---|---|---|---|
| head alone | 2.3% | 7.1% | 55.8% | 42.9% | 90–91% |
| **head + spouse (default)** | **27.7%** | **54.8%** | **61.7%** | **75.3%** | 100% |
| every interviewed member | 30.9% | 63.4% | 67.0% | 84.9% | 100% |

Head-only guts the variable: heads are 90.6% male (6,033 of 6,657) and SHGs are women's
groups, so the head's personal answer is near-constant zero on the one field closest to the
product. Head-plus-spouse keeps almost all the variance and covers essentially every
surveyed household, so `couple` is the default; `--shg-scope {couple,head,any}` switches
it. Whichever is used must be stated with the results — the three scopes disagree by a
factor of ten on SHG membership, which is more than most of the effects we will be
measuring.

Two further properties, both handled in code rather than by convention:

- **Tri-state, not boolean.** `Yes` → true, `No` → false, and *only* `Do not know` /
  `Refuse to say` (50 records bundle-wide) → unknown. Collapsing refusals to "no" would
  invent a measurement the survey did not make.
- **`surveyed == 1` is exactly "appears in the individual file"** — verified, 0
  disagreements across all 75 villages — so the two flags are present for 100% of surveyed
  households and absent for 100% of the rest. There is no partial-coverage middle case to
  reason about, which is what makes the R2 split clean.

The R1 warning applies with full force here. These two fields are the *entire* difference
between a surveyed and a non-surveyed persona under the reduced set, and they are the two
most product-adjacent variables in the bundle: SHG membership is the closest existing
analogue of microfinance, and SHG heads sit in the injection set. So the line must always
be rendered, as "not known" where it is not known, never dropped — otherwise prompt
*structure* itself becomes a covariate that correlates with degree (3.41×) and with the
target (+2.4pp).

### 4.3b `subcaste`, and why it displaces `caste`

`caste` is the administrative classification — OBC, SC, ST, General. It is the right
variable for a regression and the wrong one for an agent: those four labels are
bureaucratic categories, not identities anyone reasons *from*, and an LLM handed "OBC" has
almost nothing to work with. `subcaste` is the name a household would actually give —
Vokkaliga, Adi Karnataka, Lingayath, Kuruba — which is both legible and the level at which
social boundaries in this setting are drawn. So `subcaste` is in and `caste` is out.

What the data supports:

- **0% blank** among respondents, bundle-wide. (The 37%-blank figure elsewhere in these
  notes belongs to `occupation`, not here.)
- **434 levels bundle-wide, but 17 in village 73 and 12 in village 67.** The unusable
  granularity that got it excluded in the first draft is a property of the pooled bundle;
  within one village it is an ordinary categorical.
- **Coverage 100% of surveyed households** under the head-then-anyone rule of §4.3's
  neighbouring code: 90–91% of surveyed households have a head row, and reading the first
  interviewed member for the rest closes the gap. Near-constant within a household anyway
  — 865 of 6,901 carry more than one distinct string.
- **Stronger relative homophily than `caste`**, which is the variance argument made
  concrete:

| village | same-subcaste edges | random-mixing baseline | ratio | (same for `caste`) |
|---|---|---|---|---|
| 73 | 42.9% | 20.2% | **2.12×** | 1.59× |
| 67 | 51.3% | 22.4% | **2.29×** | 2.02× |

**The catch, which is orthographic and must not be papered over.** The same group is
spelled several ways inside a single village: village 67 has `LINGAYATH` (20 households)
next to `LINGAYATHA` (9), `GOWDA` (5) next to `GOWDAS`, and `KORACHA` / `KORACH` /
`KORASARU`; village 73 has `NAYAK` / `NAYAKA` / `NAIK`, `BRAMANA` / `BRAMHINA` / `BRAHMIN`,
`THIGALA` / `THIGALARU`; bundle-wide, `VOKKALIGA` (4,204) and `VAKKALIGA` (637) are the
same people. So village 73's 15 household-level levels are more like 9 or 10 real groups.
Two consequences:

1. The homophily figures above are a **lower bound** — splitting one group across two
   spellings can only reduce the measured same-group share.
2. An agent shown `LINGAYATHA` beside a neighbour's `LINGAYATH` may read two communities
   where there is one.

The column is left **raw** regardless, because the alternatives are worse: a prefix or
edit-distance heuristic would merge `ADI KARNATAKA` with the wrong things sooner or later,
and it would do so silently. With ≤17 levels per village, eyeballing a curated alias map is
minutes of work and is auditable — see §7.4, where that map now exists for the pilot village
and is applied downstream rather than in this builder, so what ships from here stays the raw
string and the inflation stays visible.

### 4.3c `occupation_head`, and why it is *not* the same reversal as `subcaste`

§4.5 originally dropped `occupation` bundle-wide: 1,479 distinct free-text strings, 37%
blank. That verdict does not flip the way `subcaste`'s did. Restricted to heads within one
village, the numbers look better but the underlying problem is unchanged:

| village | heads | distinct occupation strings | blank |
|---|---|---|---|
| 73 | 87 | 45 | 10% |
| 67 | 85 | 25 | 19% |

`subcaste`'s bundle-wide 434 levels collapsed to 12–17 per village because most of that
count was spelling variants of a closed vocabulary of ~15 caste names. Occupation's levels
do not collapse the same way: `AGRICULTURE LABOUR`, `AGRICULTURE COOLIE`, `AGRICULTURIST`
and `AGRICULTURE` are genuinely different answers, not four spellings of one job, and the
long tail is real work, not noise — the single most obvious bucket ("contains AGRI") covers
only 29% of village 73's heads and 55% of village 67's, leaving 19–38 distinct leftover
strings per village. A coarse taxonomy would need the keyword heuristics §4.5 already
rejected once, for the same reason: cost in code and reviewer trust exceeded the signal for
a clean categorical.

The column is in anyway, **raw and unbucketed, same head-then-anyone rule as `subcaste`**
(`_occupation` in `tools.py`, mirroring `_subcaste`) — but for a different job than
`subcaste` does. It is not meant to be a variable a model or a regression conditions on; it
is meant to widen the narrative material available when generating a household's biography,
where the LLM benefits from `STONE CUTTER` or `AGRICULTURE COOLIE` sitting alongside
`rooms`/`beds`/`religion` even though the raw string is too granular and too surveyed-only
to use as a clean feature. Two properties carried over from `_subcaste` matter here too:

- **Blank means "does not work", not "unmeasured, ask again" — so it is written out as
  `"no work"`, not left blank.** A blank `occupation` agrees with `workflag == No` for 1,021
  of 1,028 blank heads bundle-wide, so `_occupation` cross-checks `workflag` rather than
  trusting the blank string alone: blank-and-`workflag == No` becomes the literal string
  `"no work"`; blank-and-anything-else (occupation blank but `workflag == Yes`, or
  `workflag` itself unrecorded — 4 of 4,892 surveyed households bundle-wide) is a genuine
  data gap and stays `pd.NA`. Writing `"no work"` as a real string rather than an empty cell
  is what makes it a distinct, interpretable level instead of something that reads as
  missing once it round-trips through CSV.
- **Surveyed-only, same population as `subcaste`.** The same 46–54%-of-households ceiling
  applies, and the same R1 rule: render "not known" honestly for the rest, never blank as if
  it were a genuine "no occupation."

**What it costs.** `subcaste` is survey-only, so the survey block is now two bits *plus* a
15-level categorical — by far the largest persona difference between a surveyed and a
non-surveyed household in this design, and therefore the sharpest test of R1 so far. The
line must be rendered for everyone, "not known" where it is not known. It also re-widens
the asymmetry §3 warns about: the R2 ablation is no longer "two bits", and if the
individual-derived arm now fits better, subcaste's contribution has to be separated from
`has_shg`'s rather than reported as one effect.

### 4.4 `capita` and the per-capita columns, recovered for ~100% of households

Household size is in no published file, but it is recoverable exactly.
`hhcovariates<V>.csv` is BCDJ's own six-column covariate matrix — tab separated, Stata `.`
for missing — and decoding it gives:

| col | is |
|---|---|
| 1, 2 | `room_no`, `bed_no` — exact match |
| 3 | electricity code (1 Private, 2 Government, 3 None) — exact |
| 4 | latrine code (1 Owned, 2 Common, 3 None) — exact |
| 5, 6 | rooms **per capita**, beds **per capita** |

So `capita = rooms / c5 = beds / c6`, and c5/c6 are kept as features in their own right —
they are two of BCDJ's six, so the non-LLM baseline can run on **exactly the covariate
matrix the paper used**, which is most of the reason the reduced set is worth having.
Verification:

- Columns 1–4 reproduce the household file **exactly in 48 of 49 villages** (of those, four
  — 3, 39, 45, 50 — differ on a single row out of 221–292), and match at 1.000 in all five
  candidate villages of §6.
- The two independent recoveries (from rooms, from beds) agree in **100%** of households
  where both are defined, and are integer-valued in 100% of recoveries.
- The beds route alone covers only **25–54%**, because `beds == 0` makes `beds_per_capita`
  a true `0` rather than a divisor. The rooms route carries the recovery; `capita` is
  unknown only where rooms = 0 *and* beds = 0.
- Size recovered for 100% of households in 32 of 49 villages, and ≥85.3% in all (worst:
  village 20 at 85.3%). Both recommended villages in §6 are ≥98.9% (village 73: 2
  households unknown; village 67: 1). Sanity: recovered size ≥ number of surveyed members
  in 99.0% of cases.

Worth the effort because household size is first-order for a borrowing decision
(dependents, labour supply) and this is the **only** route to it for the non-surveyed
half — i.e. it strengthens the universal arm that R2 depends on.

> **Exclude village 48.** Its `hhcovariates48.csv` does not align with village 48's
> household file (`room_no` matches 8.8%) nor with any other same-length village's. Size
> is unrecoverable there. It *is* in the paper's 43-village sample, so this needs stating
> rather than silently skipping. The builder catches this generically — a rooms-column
> match below 50% blanks `capita` and both per-capita columns and emits a warning, rather
> than hard-coding village 48 as a special case.

### 4.5 What the reduction drops, and what that costs

These were all defensible inclusions. They are out because every additional field is paid
for per agent per timestep, and because — for the individual-file ones — every additional
field widens the surveyed/non-surveyed asymmetry that §3 says is the dominant constraint.
The costs, honestly:

| Dropped | Source | What it was worth | What dropping it costs |
|---|---|---|---|
| `caste` | individual, head | The strongest social-boundary variable in this setting (OBC 56%, SC 25%, General 12%, ST 6%), near-constant within household (only 255/6,901 have >1 distinct value). **Must** come from the individual file: `castesubcaste` in the household file is 100% blank in 25 of 75 villages, villages 1–5 included. | **Superseded rather than lost** — `subcaste` (§4.3b) carries the same social boundary at a level an agent can reason with, and with stronger relative homophily (2.1–2.3× vs 1.6–2.0×). Same-caste edges run 62.7% against a 39.3% random-mixing baseline in village 73 and 72.2% against 35.7% in village 67; re-derive `caste` from the individual file if a post-hoc slice by the administrative category is wanted. Note either way that homophily is already baked into the observed edge list, so what a social attribute adds is the agent weighing *who* told it — not the routing. |
| `n_women_18_57` | individual, count | BSS lent to women's groups, so this is the closest thing in the bundle to product *eligibility*. | The drop most worth reconsidering. Note if it is restored that `resp_gend` is an unlabelled 1/2 (2 = female, 9,416 of 16,984 respondents), so it needs a stated decode assumption rather than a guess. |
| `education` | individual, head | 16 ordered levels, 0% missing among respondents; plausible driver of understanding a financial product. | Collapsible to ~4 bands, but surveyed-only, and its plausible effect on take-up is not separable from caste and occupation at this sample size. |
| ~~`occupation`~~ | individual, head | Income type and volatility drive loan demand. | 1,479 distinct free-text strings, 37% blank, bundle-wide. **Re-added as `occupation_head`, raw and unbucketed, in §4.3c** — not as a clean categorical (that verdict stands), but as narrative material for generated biographies. |
| `head_age` | individual, head | Life-cycle borrowing demand. | Surveyed-only, and weakly identified against `capita`. |
| `tenure` | `ownrent` | Full coverage; tenure ⇒ collateral and stability, directly germane to a loan. | A real loss on the universal side. 13 rows carry junk labels (`"6.0"`, `"0.0"`) and 6 are truly NaN. Restore this one first if the universal block turns out too thin. |
| `roof` | `rooftype1..5` | Full coverage, standard asset-index component. | Not one-hot in practice (per `CODEBOOK.md`, 662 households have none set and 67 have two — the source question is "circle all that apply"), so it needed a `multiple`/`unknown` escape hatch that carries no meaning to an agent. |
| `crowding` | derived | Better-scaled poverty proxy than rooms or size alone. | None: it is `1 / rooms_per_capita`, which is now a column in its own right. |

### 4.6 Excluded outright. Grouped by reason, because the reasons differ.

**Redacted at publication — literally no information:**
`native_name`, `native_taluk`, `movecontact_name` are the string
`"data has been removed for publication"`.

**Too sparse to condition on** (percentages are of *respondents*, so the share of all
households is worse): `movecontact` 96.6%, `movecontact_res` 98.1%, `movecontact_hhid/pid`
98.5%, `otherlang` 99.8%, `res_time_mths` 81.3%, `work_outside_freq` 78.7%, `shg_no`
79.0%, `savings_no` 60.7%, `native_district`/`native_type`/`res_time_yrs`/`movereason`
~56%. Migration history is genuinely interesting for the social-capital angle, but there
is nothing here to work with.

**Near-constant within a village — costs tokens, buys no differentiation:**
`mothertongue`, `speakother`, and the six language dummies
(`kannada`/`tamil`/`telugu`/`hindi`/`urdu`/`english`). Our candidate villages have 1–5
distinct mother tongues, mostly Kannada-dominant. *Conditional exclusion:* if the chosen
village turns out to be linguistically mixed, reinstate `mothertongue` only — check per
village, do not assume.

**Unusable granularity:** ~~`subcaste`~~ — **this exclusion was wrong and is reversed in
§4.3b.** The 434-level count is a property of the pooled bundle; within a single village
there are 12–17 levels and 0% blank, which is an ordinary categorical. `privategovt` (269
free-text levels, 37% blank, overlaps `occupation`). `rooftypeoth` (free text, ~always
blank).

**Inconsistent units, poor return:** `work_freq` / `work_freq_type` /
`work_outside_freq` — 37–79% missing across three incompatible units (days per week,
hours per day, hours per week). Marginal signal over `occupation` does not justify the
cleaning.

**Weak variance despite genuine relevance:** `electioncard` (86% yes), `rationcard` (99%
yes), `rationcard_colour`. Card colour really is a Karnataka poverty indicator (BPL/APL),
but 16% blank across 40 spellings and little within-village spread. `rationcard_colour`
stays out of the table; re-derive it from the individual file if an independent poverty
check is ever wanted.

**Identifiers — keep for joins, never in the prompt:** `pid`, `resp_id`, `hhid`,
`adjmatrix_key`, `HHnum_in_village`. A numeric id in a prompt invites the model to invent
orderings and rankings that do not exist.

**Research artefacts the household could not know about itself:** `hhSurveyed` (keep on
the object — it defines the R2 ablation arm and is needed to interpret coverage) and
`in_giant` (keep for sample definition; BCDJ restrict to the giant component).

### 4.7 Privileged. Never in a prompt, under any circumstance.

- `_adopted` (§2) — the only privileged column, and the only household-level outcome that
  exists. The leading underscore is the convention in the CSV too: anything so prefixed is
  evaluation-only.
- **Privileged by proxy:** anything computed from the outcome vector — most importantly
  *neighbours' real adoption*. During the run an agent may see its neighbours' **simulated**
  states; if it ever sees their real ones the whole exercise collapses. Worth a separate
  assertion, because this is the leak that would look most like success.
- `has_leader` is **not** privileged. It is the seed assignment, known ex ante, and an agent
  legitimately knows the MFI spoke to it. Carry BCDJ's caveat forward: it marks who
  the MFI *could* have informed, not who verifiably was.

### 4.8 How distinct the reduced personas actually are

The reduction has a floor: strip enough fields and households stop being distinguishable,
at which point adoption heterogeneity can only come from network position. Distinct
feature vectors, candidate villages:

| | v73 (n=174) | v67 (n=193) |
|---|---|---|
| dwelling block (rooms, beds, capita, electricity, own_latrine) | 96 distinct, largest cluster 10 | 126 distinct, largest 8 |
| + religion | 96 — no change | 126 — no change |
| + has_leader, surveyed, has_shg, has_savings | 134 | 168 |
| + subcaste | 145 | 174 |
| surveyed households alone, without → with subcaste | 80 → 91 of 94 | 84 → 90 of 93 |
| non-surveyed households alone, dwelling block | 52 / 80; **59% share a vector** | 80 / 100; 34% share |

Three things follow. Personas are adequately distinct **once the flags are in** — and
`religion` adds literally nothing, as §4.0 says. `subcaste` adds real separation on top
(village 73's surveyed households go from 80 distinct vectors to 91 of 94), which is the
variance it was included for. But over the non-surveyed half, which is 46–50% of the
village, more than half of village 73's households are byte-identical to at least one
other household, so those agents differ only by who talks to them — and adding `subcaste`
widens that gap rather than closing it, because it lands entirely on the surveyed side.
That is arguably the mechanism we want to isolate; it also means the headline question
("does making the agents LLMs buy anything?") gets answered mostly on the surveyed half.
State it as a property of the design, not as something discovered afterwards.

---

## 5. Field layout

Grouped by consumer, so the separation in §1 is visible in the type. This is the schema
`tools.build_household_features()` writes to `output/features/hh_features_<village>.csv`:

```
Household
  identity      village, row (adjacency index), hhid, hh_num
  network       degree, has_leader, in_giant, neighbours   # mechanics, mostly not persona
  base          religion, rooms, beds, capita,
                rooms_per_capita, beds_per_capita,
                electricity, own_latrine                   # §4.0, always present
  survey        surveyed: bool
                subcaste,                                  # raw string, surveyed only
                occupation_head,                            # raw string, surveyed only, narrative use only -- §4.3c
                has_shg, has_savings                       # tri-state, surveyed only
  state         informed_at, adopted_at, told_by           # mutable, simulation
  _outcome      _adopted                                   # PRIVILEGED
```

`neighbours` and the `state` block are the simulation's, not the table's — the CSV is
static input, and `row` is the join back to the adjacency matrix.

### 5.1 Two profile arms, one grounding, one artefact

The profile is the experimental manipulation, so there are two of them and `profiler.py`
builds both from the *same* `render_traits()` output, into **one JSON per village**:
`output/profiles/profiles_<village>.json`, keyed by hhid.

```
"73001": {
  "traits":            {...},   # the twelve disclosed fields, labelled
  "static_profile":    "...",   # the `facts` arm, rule-based, no LLM
  "prompt":            {"model", "instructions", "input"},
  "narrative_profile": "...",   # the `story` arm, what the LLM returned
  "usage":             {...}    # what this household cost, in tokens
}
```

`--mode facts` writes the first two keys and stops; `--mode story` writes all five, so a
story-mode file is a strict superset. Keeping `traits` and `static_profile` beside the
narrative is the point: the two arms are then *provably* built from the same twelve facts,
rather than from two files that have to be trusted to agree. `prompt` is the whole request
as the model receives it, so what the model was told is recoverable from the artefact
rather than from whichever version of the source produced it.

> **This file supersedes the context `.txt` files.** An earlier iteration also wrote
> `output/context/<mode>/context_<hhid>.txt`, because that is what `agent.HH_Agent.context`
> reads. That is withdrawn — a second on-disk copy of the same text is one that can drift.
> `agent.py` and `game_master.py` still expect context files and have **not** been changed;
> wiring them to read the JSON is a separate job and is the next thing owed here.

| | `facts` | `story` |
|---|---|---|
| what the agent carries | the twelve labelled lines of §5.2 | a ~95–115 word narrative |
| written by | a pure function of the row | an LLM, once per household, then frozen |
| cost | none | one call per household, nothing per round |
| length across the surveyed split | fixed by construction | varies — measured, not constrained |

The `story` arm exists because of Concordia's claim that agents with distinct biographies,
memories and plans "behave systematically differently from one another" (Vezhnevets et al.).
What we are testing is whether that survives the biography being *grounded in real survey
data* rather than authored — which is exactly the constraint Concordia does not operate
under, and the reason this is a finding rather than a demo.

Two things this arrangement is careful about:

- **The leakage control is written once and covers both arms.** `build_request()` takes a
  `dict[str, str]`, never a row, so the model is structurally unable to see a column the
  fact listing does not contain. §1's non-negotiable therefore costs one test, not two.
- **The neighbour view is rule-based in both arms.** The mode switch applies to a
  household's own profile and not to what it is told about others. A neighbour profile is
  attached per edge per round — ~8,400 times in village 24 — and is the *non*-cacheable tail
  of that prompt, where the profile is the cacheable prefix and is paid for once. It is also
  the honest information model: you know your neighbour's house, work and community, not
  their interior life.

**Narrative length varies with `surveyed`, and that is left alone.** A surveyed household
has four more facts to write from, so its narrative runs longer — measured at roughly 6
words on a ~110-word budget over 40 households of village 73 (Welch *t* ≈ 3). That is a real
consequence of knowing more about a household, not an artefact to engineer away, so nothing
constrains it. `profiler.balance_report()` prints the gap on every live run purely so the
number is on the record: `surveyed` tracks degree (3.41×) and take-up (+2.4pp) per §3, so
anything co-varying with it must be quoted with the results rather than found afterwards.
The `facts` arm is fixed-length by construction and cannot show the effect at all, which is
part of why it is the headline arm and the story is the ablation.

### 5.2 Representation

Four decisions that matter more than they look:

- **A plain fact listing, not a rendered persona.** What `profiler.render_traits()` hands
  the model is one labelled survey field per line — twelve of them for every household:
  `religion, rooms, beds, capita, rooms_per_capita, beds_per_capita, electricity,
  own_latrine, subcaste, occupation_head, has_shg, has_savings` — with `not known` where
  the value is missing, and no interpretation on top. The labels do the disambiguating the
  column names cannot (`has_shg` → "member of a savings self-help group", `has_savings` →
  "has a bank or savings account", `own_latrine` → owns one vs. shared/none), because that
  meaning is in the codebook, not in the data. The prose — and any judgement about what
  two rooms for six people amounts to — is the LLM's to write.

  *This amends an earlier decision here*, which was to pre-digest the dwelling fields into
  a within-village percentile ("more crowded than most households here") and to state
  `beds = 0` in words rather than as a figure. Both put our reading of the data into the
  prompt ahead of the model's, and the crowding percentile in particular made the rendering
  depend on the whole village rather than the row. Derived features like that are out at
  this stage; the raw figures go in and the identity `rooms = capita × rooms_per_capita`
  is simply stated three times, which is the price of not interpreting it for the model.
- **Do not state degree numerically.** "You have 17 friends" reads to an LLM as an
  instruction to be influential, which manufactures the network effect we are trying to
  measure. Network position should reach the agent only through who actually talks to it.
  Keep `degree` on the object for evaluation.
- **`has_leader` is not a disclosed field, but it is not wasted either.** It never appears
  as a flag of its own. It is read in exactly one place: when a leader household's
  `occupation_head` is unknown, the occupation line becomes "works in a role that entails
  leadership in the village" — which is what BSS's designation means (teachers,
  shopkeepers, SHG leaders). A leader household that already has an occupation keeps it,
  and renders identically to a non-leader household. That keeps the seed assignment from
  reading as a status the agent should act on, while still using the one thing it tells us
  about the ~46% of leader households the individual survey never reached.
- **The whole prompt, and what it cost, go in the output.** `biographies_<village>.json`
  records `prompt: {model, instructions, input}` — the system instructions as well as the
  facts — so what the model was told is recoverable from the artefact rather than from
  whichever version of the source produced it. Alongside it, `usage: {input_tokens,
  output_tokens, total_tokens, cached_input_tokens, reasoning_tokens}` as the API reported
  it, per household. Per household rather than per village because the persona is the
  thing we pay for once per agent per timestep, so its token size is the number that
  scales the whole simulation; `profiler.total_usage()` aggregates when a total is what
  is wanted.

Also: the persona string is immutable, so build it once per household and cache it. Only
the dynamic block (who told me what, which trimester) varies per call. That is the main
lever on LLM cost.

---

## 6. Village selection

Filtering the 43 analysis villages on: `n ≤ 210`, caste available, size recoverable,
`hhcovariates` intact, non-trivial non-leader take-up:

| village | n | ind. coverage | caste blank | take-up (all / non-leader) | trimesters | deg srv/not | isolates |
|---|---|---|---|---|---|---|---|
| **73** | 174 | **54.0%** | 6% | 17.2% / 15.6% | 7 | 16.5 / 5.9 | 4 |
| **75** | 172 | 50.6% | 3% | 23.3% / 23.7% | 7 | 17.4 / 5.6 | 8 |
| 67 | 193 | 48.2% | 0% | 29.5% / 29.4% | 7 | 15.5 / 5.8 | **2** |
| 62 | 190 | 49.5% | 1% | 17.4% / 16.1% | 6 | 13.3 / 4.3 | 4 |
| 68 | 153 | 45.8% | 0% | 14.4% / 13.5% | 7 | 15.0 / 5.2 | 3 |

Recommend **73 as primary** (best individual coverage, so the R2 ablation has the most to
work with; take-up 17.2% is near the sample median) and **67 as the contrast** (very
different take-up at 29.5%, caste fully present, almost no isolates). A second village
with a different base rate is what stops us tuning to one number.

Note village **1** is the obvious default and a poor choice here: `castesubcaste` 100%
blank, individual coverage only 41.8%. The existing figures use it; worth redoing.

The ranking predates the reduction and is left as computed. Caste availability is no
longer a feature requirement (§4.5), so it now reads as what it always partly was — a
proxy for individual-file coverage, plus a precondition for the post-hoc homophily check.
Nothing in the ordering changes: 73 and 67 lead on the criteria that still bind.

---

## 7. Open decisions

1. **Household vs individual as the agent.** Recommend household: the outcome is
   household-level, the networks are household-level (`adj_*_HH_*`), and BCDJ model it
   that way. The cost is real though — the person who joins is usually a woman, while the
   survey respondent is often the male head (6,033 of 6,657 heads are male). Under the
   reduced set the patch for this is `--shg-scope couple` (§4.3), which reads the spouse's
   answer as well as the head's; `n_women_18_57` was the fuller patch and is dropped
   (§4.5).
2. **Sample restriction.** BCDJ prune to the giant component before estimating. If we
   follow, we drop 2–19 isolates depending on village and our denominator stops matching
   `takeup_all`. Recommend keeping isolates in (they are genuine non-adopters and a free
   correctness check: a sane model must never have them adopt) and reporting both.
3. **`leader` as "informed".** Inherited from BCDJ, but it means the seed set is
   over-stated by an unknown amount. Assumption to state, not resolve.
4. **Normalising `subcaste` spellings.** ~~The column ships raw (§4.3b)~~ — **settled for the
   pilot village, still open elsewhere.** The column still ships raw from
   `build_household_features()`, and `src/subcaste.py` applies a hand-curated per-village
   alias map on top, writing `output/features/CLEANED_hh_features_<village>.csv` with the raw
   string kept in a `subcaste_raw` column so the merge stays reversible — this section's own
   condition. Village 6 is the only village reviewed so far: 28 household-level levels → 21,
   10 of 44 rows rewritten, measured homophily 2.05× → 2.14×. `alias_map()` raises for an
   unreviewed village rather than returning an empty map, so nothing else is silently
   half-cleaned. The map claims orthography only — plurals, honorific suffixes,
   transliterations, initialisms — and records the merges it deliberately did *not* make, with
   reasons, in `subcaste.KEPT_APART`. See `docs/experiment_design.md` §7.1 for the numbers and
   the three limits that follow (only 36% of village 6's edges have a subcaste at both ends).
   What remains open: the map for every other village, and whether the *headline* arm reads
   the cleaned table or the raw one — the raw column surviving makes that a testable
   manipulation rather than a preference (experiment design §9).

---

## Appendix: how each claim here was checked

Profiling scripts under `$CLAUDE_JOB_DIR/tmp` — coverage and missingness by column,
`hhcovariates` decoding against the household file for all 49 MF villages, `latrine`
label-vs-code check on raw Stata codes, surveyed/non-surveyed degree and take-up gaps with
a *t*-test, and the village ranking in §6 (`profile.py`, `profile2.py`, `profile4.py`,
`pick.py`); then, for the reduction, the electricity and latrine wealth-proxy tables of
§4.1–4.2, the shg/savings scope comparison of §4.3, the subcaste level counts, coverage and
homophily of §4.3b, the caste homophily baselines of §4.5 and the distinct-vector counts of
§4.8 (`eval_reduced.py`, `eval3.py`, `eval4.py`). The §4.8 counts and the §4.3 scope table
are re-derived from the built CSVs, not from the profiling scripts, so the doc and the
output cannot drift apart unnoticed.

These are throwaway. Three of the checks are worth keeping and should move into
`data_loader.check_consistency`: the `hhcovariates` alignment test (currently duplicated
inside `tools._covariates`), the electricity monotonicity test — if it ever fails on a
subsample the ordinal coding is no longer licensed — and the `surveyed == in individual
file` equivalence that §4.3 relies on.

**One check we deliberately do not run:** nothing here is validated against `MF<V>.csv`.
Per R3, no field was kept or dropped by looking at its correlation with the outcome, and
the ceiling this feature set implies for household-level prediction is a post-hoc finding
to report, never an input to the design.
