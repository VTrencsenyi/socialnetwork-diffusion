# Codebook — `diffusion-science-data/datav4.0`

Replication package for **Banerjee, Chandrasekhar, Duflo & Jackson (2013), "The Diffusion of
Microfinance", *Science* 341(6144):1236498**. Package version 4.0, dated 16 September 2013
(per `README.pdf`).

This codebook documents **every type of data file** in the package and **every column** in it.

---

## 0. How to read this document

### Provenance tags

Every non-obvious statement below carries a tag saying where it comes from. Nothing here is guessed;
where a fact could not be established from the sources it is marked **`[?]`** explicitly.

| Tag | Source |
|-----|--------|
| `[R]` | `datav4.0/README.pdf` shipped with the package |
| `[P]` | The *Science* paper (`papers/Banerjee-…-Science-2013.pdf`) |
| `[L]` | Stata variable label / value label embedded in the `.dta` file itself |
| `[S]` | A replication script in the package (`Stata Replication/do files/*.do`, `Matlab Replication/**/*.m`) |
| `[D]` | Derived by direct inspection of the data files. Each `[D]` claim below states the check that was run, so it can be re-verified. |
| `[?]` | **Not documented in the README, not stated in the paper, and not determinable from the replication scripts.** Treat as unknown. |

### Package-wide conventions

* **75 villages**, numbered `1`–`77`; **village 13 and village 22 do not exist** in the data. `[R]`, confirmed `[D]`
* **43 of the 75 villages** are the ones the microfinance institution (BSS) eventually entered, and are
  the analysis sample of the paper. `[P]`, `[R]` The 43 are:
  `1–4, 6, 9, 12, 15, 19–21, 23–25, 29, 31–33, 36, 39, 42, 43, 45–48, 50–52, 55, 57, 59, 60, 62, 64, 65, 67, 68, 70–73, 75` `[S]` (`Main_models_1_3.m`, variable `vills`; identical to the village list in `cross_sectional.dta` `[D]`)
* **Two units of observation.** Almost every file exists in an *individual* and a *household* version.
  Diffusion is modelled at the **household** level, because that is the unit at which microfinance
  participation is decided. `[P]`
* **Setting.** 75 villages in rural southern Karnataka, India. A full **household census** was run in
  every village; a **detailed individual survey** (which is where the network questions live) was run
  on a stratified random subsample — about 46% of households per village `[P]` (14 904 households
  censused, 6 901 of them individually surveyed = 46.3% `[D]`).
* **Timing.** Network data were collected ~6 months *before* BSS entered the villages; participation
  data were collected up to early 2011. `[P]`

### ID scheme `[D]` — exact, verified on all rows

```
hhid = village * 1000 + HHnum_in_village        (verified: holds for all 14 904 households)
pid  = hhid    * 100  + resp_id                 (verified: holds for all 16 984 individuals)
```

So `pid = 100201` ⇒ village 1, household 2, roster line 1. `[R]` gives this example informally; the
arithmetic relation above is `[D]`.

---

## 1. File inventory

| Directory | File pattern | Count | Unit | Format |
|---|---|---|---|---|
| `Data/1. Network Data/Adjacency Matrices/` | `adj_<relation>_vilno_##.csv` | 14 relations × 75 villages = 1050 | individual | headerless CSV matrix |
| `Data/1. Network Data/Adjacency Matrices/` | `adj_<relation>_HH_vilno_##.csv` | 14 × 75 = 1050 | household | headerless CSV matrix |
| `Data/1. Network Data/Adjacency Matrix Keys/` | `key_vilno_##.csv` | 75 | individual | headerless single column |
| `Data/1. Network Data/Adjacency Matrix Keys/` | `key_HH_vilno_##.csv` | 75 | household | headerless single column |
| `Data/2. Demographics and Outcomes/` | `household_characteristics.dta` | 1 | household | Stata (19 cols × 14 904 rows) |
| `Data/2. Demographics and Outcomes/` | `individual_characteristics.dta` | 1 | individual | Stata (48 cols × 16 984 rows) |
| `Stata Replication/data/` | `cross_sectional.dta` | 1 | village | Stata (15 cols × 43 rows) |
| `Stata Replication/data/` | `panel.dta` | 1 | village × period | Stata (16 cols × 325 rows) |
| `Stata Replication/do files/` | `makeScience*.do` | 5 | — | Stata scripts |
| `Matlab Replication/India Networks/` | `adjacencymatrix.mat` | 1 | household | MATLAB 43×1 cell array |
| `Matlab Replication/India Networks/` | `hhcovariates##.csv` | 49 | household | headerless **tab**-separated, 6 cols |
| `Matlab Replication/India Networks/` | `MF##.csv` | 49 | household | headerless single column |
| `Matlab Replication/India Networks/` | `HHhasALeader##.csv` | 49 | household | headerless **tab**-separated, 2 cols |
| `Matlab Replication/India Networks/` | `inGiant##.csv` | 43 | household | headerless single column |
| `Matlab Replication/India Networks/` | `Omega_abs##.csv`, `Omega_rel##.csv` | 43 each | household × household | headerless CSV matrix |
| `Matlab Replication/India Networks/` | `DOmega_abs##.csv`, `DOmega_rel##.csv` | 43 each | household × household | headerless CSV matrix |
| `Matlab Replication/India Networks/` | `NOmega_abs##.csv`, `NOmega_rel##.csv` | 43 each | household × household | headerless CSV matrix |
| `Matlab Replication/India Networks/` | `leaders.zip` | 1 | — | zip of the 49 `HHhasALeader##.csv` |
| `Matlab Replication/`, `Matlab Replication/GMMDiffusion/` | `*.m` | 10 | — | MATLAB source |
| `Survey Instruments/` | `household.doc`, `individual.doc`, `village*.pdf` | 5 | — | questionnaires |

Village coverage differs by file family (all `[D]`, by enumerating filenames):

* **75 villages**: everything under `Data/`.
* **49 villages** (`hhcovariates`, `MF`, `HHhasALeader`): the 43 analysis villages **plus** `10, 28, 37, 41, 66, 77`. The 6 extras are not used by any script in the package; **why they are included is not documented `[?]`**.
* **43 villages** (`inGiant`, all six `*Omega_*` families, `adjacencymatrix.mat`): the analysis sample.

> **Note on `.csv` files in `Data/2. Demographics and Outcomes/` and `Stata Replication/data/panel.csv`.**
> `household_characteristics.csv`, `individual_characteristics.csv` and `panel.csv` are **not part of the
> original package** — they are exports created later inside this project (file mtimes are 2025-08-14,
> versus 2013–2014 for every original file) `[D]`. They carry the same columns as the corresponding
> `.dta`, but **labelled variables are exported as their text labels, not their numeric codes**
> (e.g. `religion` is `"HINDUISM"`, `resp_status` is `"Head of Household"`) `[D]`. Unlabelled numeric
> variables such as `resp_gend` stay numeric. Prefer the `.dta` files for analysis.

---

## 2. `Data/1. Network Data` — adjacency matrices and keys

### 2.1 The 14 relations

Respondents in the individual survey named other villagers along **12 network dimensions** `[P]`, `[R]`.
Two composite networks are also shipped.

| File stem | Survey question — "who do you…" | `[R]` / `[P]` |
|---|---|---|
| `borrowmoney` | …borrow money from | `[R]`, `[P]` |
| `lendmoney` | …lend money to | `[R]`, `[P]` |
| `keroricecome` | …borrow kerosene or rice from | `[R]` ("borrow kerosene or rice from"); `[P]` "borrow material goods (kerosene, rice, etc.)" |
| `keroricego` | …lend kerosene or rice to | `[R]`, `[P]` |
| `giveadvice` | …give advice to | `[R]`, `[P]` |
| `helpdecision` | …help with a decision / get advice from | `[R]` "help with a decision"; `[P]` "those from whom the respondent gets advice" |
| `medic` | …obtain medical advice from | `[R]`, `[P]` |
| `nonrel` | …engage socially with (non-relatives) | `[R]` "engage socially with"; `[P]` "nonrelatives with whom the respondent socializes" |
| `rel` | …are related to (kin in the village) | `[R]`, `[P]` |
| `templecompany` | …go to temple with (temple, church or mosque) | `[R]`, `[P]` |
| `visitgo` | …visit in another's home | `[R]` |
| `visitcome` | …invite to one's home | `[R]` |
| `allVillageRelationships` | **union** of the 12 above | `[R]`; verified exactly for all 75 villages at both levels `[D]` |
| `andRelationships` | **intersection** of the 12 above | `[R]`; verified exactly for all 75 villages at both levels — including the 13 differently-indexed household files, once re-indexed as described in §7.2 `[D]` |

The `keroricecome`/`keroricego` and `visitcome`/`visitgo` direction convention (who is the subject of
the verb) is stated only loosely in `[R]`; the matrices are symmetric in any case, so direction is
not recoverable from the data. `[?]`

### 2.2 `adj_<relation>_vilno_##.csv` — individual-level adjacency matrix

* **Shape**: `N_i × N_i`, where `N_i` is the number of lines in `key_vilno_##.csv` (verified for all 1050 files `[D]`).
* **No header row and no header column** `[R]`. Row `k` and column `k` refer to the same individual.
* **Lines end in a trailing comma**, so a naive parse yields one extra all-empty column `[D]`.
* **Values**: `0`/`1` only, in all 1050 files `[D]`.
* **Symmetry**: symmetric in all 1050 files `[D]`. "The networks are undirected (each matrix is symmetric)" `[R]`.
* **Diagonal**: **not always zero.** 69 of the 75 villages contain at least one self-loop in at least one individual-level relation; the union network `allVillageRelationships` alone contains 319 self-loops across the 75 villages `[D]`. Not mentioned in `[R]` or `[P]`; **cause is undocumented `[?]`** (plausibly a respondent naming someone who resolved to themselves). Zero them out if your analysis assumes a simple graph.
* **Population**: rows cover **every individual in the household census** of the village, not only survey respondents. Example: village 1 has 843 rows but only 203 individually-surveyed people `[D]`. Non-respondents appear because they were *named* by respondents — in village 1's union network their mean degree is 5.5 vs 16.2 for respondents `[D]`.

### 2.3 `adj_<relation>_HH_vilno_##.csv` — household-level adjacency matrix

Same format as §2.2, with `N_h × N_h` = number of lines in `key_HH_vilno_##.csv` — **except for 13
`andRelationships_HH` files, which are indexed by raw `HHnum_in_village` instead and need re-indexing;
see §7.2** `[D]`.

* Construction: *"A relationship between households exists if any household members indicated a relationship with members from the other household."* `[R]`
* Values `0`/`1`, symmetric, **diagonal always zero** — verified for all 1050 files `[D]`.
* Rows cover **all censused households** in the village, surveyed or not. In village 1: 182 households, of which 76 were individually surveyed; non-surveyed households have mean degree 4.0 vs 17.3 for surveyed ones, and 6.6% of them are isolates `[D]`.

### 2.4 `key_vilno_##.csv` — individual key

Single column, no header. Line `k` gives the **`pid`** of the individual occupying row/column `k` of
every individual-level adjacency matrix for that village. `[R]`, verified `[D]`

* 69 441 lines across the 75 files `[D]`.
* Sorted ascending within a village `[D]`.
* Reverse lookup: `individual_characteristics.dta` carries `adjmatrix_key`, which is exactly this row number — verified that `key_vilno_v[adjmatrix_key − 1] == pid` for every row `[D]`. `[R]` states this is provided "to make this process easier".

### 2.5 `key_HH_vilno_##.csv` — household key

Single column, no header. Line `k` gives the **`HHnum_in_village`** (*not* the full `hhid`) of the
household occupying row/column `k` of every household-level adjacency matrix. `[D]` — verified that
the set of key values equals the set of `HHnum_in_village` for that village.

* 14 904 lines across the 75 files — exactly one per censused household `[D]`.
* Always ascending, in all 75 files `[D]`.
* ⚠️ **`HHnum_in_village` is not always the consecutive run `1…N`.** In **13 villages** some household
  numbers were skipped in the census, so the key jumps: e.g. village 1 has 182 households but numbers
  run to 183 (number 11 is absent); village 61 has 122 households numbered up to 127 `[D]`. The 13
  villages are **1, 3, 4, 6, 18, 20, 21, 23, 26, 45, 58, 61, 71**, and they are exactly the villages
  affected by the `andRelationships_HH` indexing quirk in §7.2 `[D]`.
* Consequently `adjmatrix_key` is the **rank** of `HHnum_in_village` within the village, not the
  household number itself — verified for all 75 villages `[D]`. The two coincide for 91.8% of
  households overall `[D]`. **Always join through the key file or through `adjmatrix_key`; never assume
  row `k` is household number `k`.**

---

## 3. `Data/2. Demographics and Outcomes`

### 3.1 `household_characteristics.dta` — 14 904 rows × 19 columns

*"Demographic information about a household's home (roof type, number of rooms, latrine type, etc.)"*
`[R]`. Administered to **every** household in every one of the 75 villages `[R]`. One row per household;
`hhid` is unique `[D]`.

Column labels below are the **verbatim Stata variable labels** `[L]`; the leading numbers are the
question numbers in `Survey Instruments/household.doc`.

| Column | Type | Label `[L]` | Codes / notes |
|---|---|---|---|
| `village` | int8 | Village number | 1–77, excl. 13 & 22 |
| `adjmatrix_key` | int16 | Household's row/column number in adjacency matrix | 1-based row index into `adj_*_HH_vilno_##.csv` `[R]`. Range 1–356 `[D]` |
| `HHnum_in_village` | int16 | Household number that corresponds to key file | the value that appears in `key_HH_vilno_##.csv` `[D]` |
| `hhid` | int32 | Household id | `= village*1000 + HHnum_in_village` `[D]` |
| `hohreligion` | int8 | 2.0 What is the religion of the household head? | `1 HINDUISM`, `2 ISLAM`, `3 CHRISTIANITY` `[L]`. No missings `[D]` |
| `castesubcaste` | string | 3.2 What is your caste? | Despite the name this holds the **caste category**, not a subcaste. Observed values `[D]`: `OBC` (5 517), `SCHEDULE CASTE` (2 584), `GENERAL` (1 371), `SCHEDULE TRIBE` (618), `MINORITY` (359), blank (4 455). The instrument offers `OBC / SCHEDULE CASTE / GENERAL / SCHEDULE TRIBE / OTHER` `[S: household.doc]`; **`MINORITY` is not one of the printed options and its mapping is undocumented `[?]`** |
| `rooftype1` | int8 | 3.1 (1 Thatch) What type of roofing material does your house have? | 0/1 dummy |
| `rooftype2` | int8 | 3.1 (2 Tile) … | 0/1 dummy |
| `rooftype3` | int8 | 3.1 (3 Stone) … | 0/1 dummy |
| `rooftype4` | int8 | 3.1 (4 Sheet) … | 0/1 dummy |
| `rooftype5` | int8 | 3.1 (5 RCC) … | 0/1 dummy |
| `rooftypeoth` | string | 3.1 (OTHER) … | free text, 669 non-blank `[D]`; e.g. `MUD`, `MUD THATCH`, `JANTIGE`. Question is *circle all that apply* `[S: household.doc]`, and indeed 67 households have two roof dummies set and 662 have none `[D]` |
| `room_no` | int8 | 3.2 How many rooms does your house have? | 0–20 `[D]`. (The instrument reuses "3.2" for both caste and rooms — that duplication is in the questionnaire itself `[S: household.doc]`) |
| `bed_no` | int8 | 3.3 How many beds/cots does your house have? | 0–50 `[D]`; 109 households report 0 in village 1 alone |
| `electricity` | float64 | 3.4 Does this house have electricity? | `1 Yes, Private` (9 195), `2 Yes, Government` (4 565), `3 No` (1 141), missing 3 `[L]`,`[D]` |
| `latrine` | float64 | 3.5 What type of latrine does your house have? | `1 Owned` (3 887), `2 Common` (84), `3 None` (10 930), missing 3 `[L]`,`[D]` |
| `ownrent` | float64 | 3.6 The house that this household occupies is… | `1 OWNED` (13 369), `2 OWNED BUT SHARED` (143), `3 RENTED` (735), `4 LEASED` (23), `5 GIVEN BY GOVERNMENT` (615), `-888 REFUSE TO SAY`, `-999 DO NOT KNOW` `[L]`. **Two out-of-label values occur in the data: `6` (12 rows) and `0` (1 row)** `[D]`. `6` most likely corresponds to the instrument's `OTHER (SPECIFY)` option `[S: household.doc]` but this is **not confirmed `[?]`**; `0` is **undocumented `[?]`**. 6 rows missing |
| `hhSurveyed` | int8 | Dummy for if this household was surveyed | 1 = the household also received the *individual* questionnaire. Sums to 6 901 (46.3%) `[D]`, matching "about 46% of all households per village" `[P]`. The set of `hhid` in `individual_characteristics.dta` is exactly the set with `hhSurveyed == 1` `[D]` |
| `leader` | int8 | Household contains a leader | 1 = the household contains one of BSS's pre-designated "leaders" (teachers, shopkeepers, self-help-group leaders) — the **injection points** of the diffusion model `[P]`. Sums to 1 838 over 75 villages, **1 157 over the 43 analysis villages** `[D]`; the paper reports "nearly 1140 leaders throughout the 43 villages" `[P]`. Leader status is **not** conditional on being surveyed: 854 leader households have `hhSurveyed == 0` `[D]` |

> **There is no microfinance-participation variable in this file.** `[R]` §3.4 claims it *"has … a dummy
> that indicates whether anyone in the household became a microfinance client"* — **no such column
> exists in v4.0** `[D]`. Take-up is only available from `Matlab Replication/India Networks/MF##.csv`
> (§5.3), for 49 of the 75 villages.

### 3.2 `individual_characteristics.dta` — 16 984 rows × 48 columns

*"Individual demographic information (age, caste, religion, language, occupation, etc.) … conducted
among a little under half of households, and also asked for social network information"* `[R]`.

Administered to the head of household, the spouse of the head, any other women aged 18–50 who are
permanent residents, and the spouses of those women `[S: individual.doc]`.

**Missing-value convention** (used throughout, from the questionnaire and value labels): `-999` =
*Do not know*, `-888` = *Refuse to say* `[L]`, `[S: individual.doc]`. These are stored as ordinary
negative numbers, **not** as Stata missing — filter them out before averaging.

| Column | Type | Label `[L]` | Codes / notes |
|---|---|---|---|
| `village` | int8 | Village number | |
| `adjmatrix_key` | int16 | Person's row/column number in adjacency matrix | 1-based row index into `adj_*_vilno_##.csv` `[R]`; verified `[D]` |
| `pid` | int32 | Person id to match up with key file | `= hhid*100 + resp_id` `[D]`. **One `pid` is duplicated: `6109803` appears twice (village 61)** `[D]` — undocumented `[?]` |
| `hhid` | int32 | Household ID | joins to `household_characteristics.dta` |
| `resp_id` | int8 | Resp_Id | the respondent's line number on the household roster grid `[S: individual.doc]`, range 1–29 `[D]` |
| `resp_gend` | int8 | 0.3 Gender | **`1 MALE`, `2 FEMALE`** — the `.dta` carries **no value label** for this variable, the coding is read off the questionnaire `[S: individual.doc]`. Counts: 7 568 male, 9 416 female `[D]` |
| `resp_status` | int8 | 0.4 Status | `1 Head of Household`, `2 Spouse of Head of Household`, `3 Other` `[L]` |
| `age` | int8 | 1.0 How old are you now? | 10–99 `[D]` |
| `religion` | float64 | 2.0 What is your religion? | `1 HINDUISM`, `2 ISLAM`, `3 CHRISTIANITY`, `-999 Do not know` `[L]` |
| `caste` | float64 | 3.0 What is your caste? | `1 SCHEDULED CASTE`, `2 SCHEDULED TRIBE`, `3 OBC`, `4 GENERAL`, `-999 DO NOT KNOW` `[L]`. **Note this ordering differs from the household questionnaire's caste codes** (`OBC=1, SC=2, GENERAL=3, ST=4`) `[S]` |
| `subcaste` | string | 3.1 What is your subcaste? | free text, 434 distinct values `[D]`; e.g. `VOKKALIGA`, `ADI KARNATAKA`, `LINGAYATH`, `KURUBA`. Spelling is not standardised (`VOKKALIGA`/`VAKKALIGA` both occur) |
| `mothertongue` | string | 4.0 What is your mother tongue? | 7 values `[D]`: `KANNADA`, `TELUGU`, `TAMIL`, `URDU`, `HINDI`, `MARATI`, `MALAYALAM` |
| `speakother` | int8 | 4.1 Do you speak any other languages? | `1 Yes`, `2 No` `[L]` |
| `kannada` | int8 | 4.2 Also speaks Kannada (dummy) | `0 No`, `1 Yes` `[L]` |
| `tamil` | int8 | 4.2 Also speaks Tamil (dummy) | `0 No`, `1 Yes` |
| `telugu` | int8 | 4.2 Also speaks Telugu (dummy) | `0 No`, `1 Yes` |
| `hindi` | int8 | 4.2 Also speaks Hindi (dummy) | `0 No`, `1 Yes` |
| `urdu` | int8 | 4.2 Also speaks Urdu (dummy) | `0 No`, `1 Yes` |
| `english` | int8 | 4.2 Also speaks English (dummy) | `0 No`, `1 Yes` |
| `otherlang` | string | 4.2 Also speaks other language | free text, only 30 non-blank `[D]` |
| `educ` | int8 | 5.0 What is the highest level of education you achieved? | `1`–`9` = 1st–9th standard, `10 S.S.L.C.`, `11 1ST P.U.C.`, `12 2ND P.U.C.`, `13 UNCOMPLETED DEGREE`, `14 DEGREE OR ABOVE`, `15 OTHER DIPLOMA`, `16 NONE` `[L]`. **Not monotone in years of schooling** — `16 NONE` sorts above a degree |
| `villagenative` | int8 | 6.0 Is this village your native home? | `1 Yes`, `2 No` `[L]` |
| `native_name` | string | 6.1 What is your native home? | free text |
| `native_type` | string | 6.1 What is your native home? (Type) | `VILLAGE` / `TOWN` / `CITY` `[D]` |
| `native_taluk` | string | 6.2 In what taluk is your native home? | free text |
| `native_district` | string | 6.3 In what district is your native home? | free text |
| `res_time_yrs` | float64 | 6.4 How long have you lived in this village? (Years) | 0–75, plus `-999 Do not know` `[L]`,`[D]` |
| `res_time_mths` | float64 | 6.4 … (Months) | 0–45 `[D]` — values above 12 occur, so this is **not** strictly a within-year remainder `[?]` |
| `movereason` | string | 6.5 Why did you move to this village? | free text, 23 distinct `[D]`; mostly `MARRIAGE`, `WORK`, `EDUCATION` |
| `movecontact` | float64 | 6.6 Did you have a contact in the village when you moved here? | `1 Yes`, `2 No` `[L]` |
| `movecontact_res` | float64 | 6.7 Does this person (your contact) still live here? | `1 Yes`, `2 No` `[L]` |
| `movecontact_hhid` | float64 | 6.7 Name of person (contact) | **holds the contact's `hhid`**, range 1007–77153 `[D]` |
| `movecontact_pid` | float64 | 6.7 Name of person (contact) | range 1–55 `[D]` — this is a **roster line number, not a full `pid`**; combine as `movecontact_hhid*100 + movecontact_pid` to get a `pid`. **This reconstruction is inferred from the ID scheme, not documented `[?]`** |
| `movecontact_name` | string | 6.7 Name of person (contact) | free text |
| `workflag` | float64 | 7.0 Did you work last week? | `1 Yes`, `2 No` `[L]` |
| `work_freq` | float64 | 7.1 How much time did you spend working last week? | 0–49 `[D]`; **units given by `work_freq_type`** |
| `work_freq_type` | string | 7.1 … (Unit) | `DAYS PER WEEK` / `HOURS PER DAY` / `HOURS PER WEEK` `[D]` — **not comparable across rows without conversion** |
| `occupation` | string | 7.2 What is your occupation? | free text, 1 478 distinct, uncleaned `[D]` |
| `privategovt` | string | 7.3 For whom do you work? | free text, 268 distinct `[D]` — despite the name it is **not** a clean private/government dummy (`OWN`, `LAND LORD`, `OTHERS`, … all occur) |
| `work_outside` | float64 | 7.4 Do you travel outside the village for work? | `1 Yes`, `2 No`, `-999 Do not know` `[L]`; **value `0` also occurs and is unlabelled `[?]`** `[D]` |
| `work_outside_freq` | float64 | 7.5 How many nights a month are you away from the village? | 0–30, plus `-888`/`-999` `[L]`,`[D]` |
| `shgparticipate` | float64 | 24.0 Do you currently participate in an SHG or other savings group? | `1 Yes`, `2 No`, `-999 Do not know` `[L]` |
| `shg_no` | float64 | 24.1 How many SHGs do you participate in? | 1–4, plus `-888` `[L]`,`[D]` |
| `savings` | int16 | 26.0 Do you have a bank or savings account? | `1 Yes`, `2 No`, `-888`, `-999` `[L]` |
| `savings_no` | float64 | 26.1 How many bank or savings accounts do you have? | 1–8, plus `-999` `[D]`; **no value label attached** |
| `electioncard` | float64 | 28.0 Do you have an election card? | `1 Yes`, `2 Missing`, `3 No`, `-888`, `-999` `[L]`. Note the unusual `2 = Missing` in the middle of the scale |
| `rationcard` | float64 | 29.0 Do you have a ration card? | `1 Yes`, `2 Missing`, `3 No`, `-888`, `-999` `[L]` |
| `rationcard_colour` | string | 29.1 What color is your ration card? | free text, 39 distinct `[D]`; `YELLOW`, `GREEN`, `BLUE`, `RED`, plus scheme names `APL`, `BPL`, `AKSHAYA` |

Questions numbered 8–23, 25 and 27 of the individual questionnaire (including the network-name
rosters themselves) are **not released as columns** — the network modules are released only in
already-constructed adjacency-matrix form. `[D]`, consistent with `[R]`

---

## 4. `Stata Replication`

### 4.1 `cross_sectional.dta` — 43 rows × 15 columns (one row per analysis village)

*"Leader centrality data … and village-level information … as well as the microfinance take-up rate
in each village"* `[R]`. Used by `makeScienceFigure2.do`, `makeScienceTable3.do`,
`makeScienceTableS2.do`, `makeScienceTableS3.do` `[S]`.

All eight `*_leader` variables are **averages over the set of leader households in the village**
`[P]` (Table 3 caption: "measures of centrality—averaged over the set of leaders").

| Column | Label `[L]` | Notes |
|---|---|---|
| `village` | *(no label)* | the 43 analysis villages |
| `mf` | Microfinance take-up rate (non-leader households) | **Verified exactly** `[D]`: equals `mean(MF##.csv)` over households with `HHhasALeader##.csv == 0`. This is the paper's dependent variable — "the microfinance participation rate of nonleader households in a village" `[P]` |
| `degree_leader` | Average degree of leaders (corrected) | *"corrected"* refers to the correction for missing data from the ~46% sampling `[P]`: "we corrected some of our measures for missing data". **The exact correction formula is not given in the package or the main paper `[?]`** — see the paper's supplementary materials |
| `eigenvector_centrality_leader` | Eigenvector centrality of leaders | |
| `between_centrality_leader` | Between centrality of leaders | betweenness centrality |
| `bonacich_centrality_leader` | Bonacich centrality with parameter 0.8*1/lambda1 | Katz–Bonacich with decay `0.8/λ₁(g)` |
| `decay_centrality_leader` | Decay centrality with p = 0.18 | |
| `diffusion_centrality_leader` | Sum of (A/lambda1)^t with flexible T | The paper's new measure: `DC(g,q,T) = Σ_{t=1..T} (qg)^t · 1` with `q = 1/λ₁(g)` and `T` set to the number of trimesters the village was exposed to BSS (6.6 on average) `[P]`, Eq. 5 |
| `closeness_centrality_leader` | Closeness centrality | |
| `communication_centrality_leader` | Using full diffusion model (qN,qP,etc.) to compute centrality | The paper's model-based measure — simulated fraction of the village that ends up informed/participating when a given node is the only initially informed one `[P]` |
| `numHH` | Number of households | **Verified exactly** `[D]`: number of rows for that village in `household_characteristics.dta` |
| `fractionLeaders` | Fraction of nodes that are leaders | **Verified exactly** `[D]`: `mean(leader)` over the village's households |
| `savings` | Average savings | ⚠️ **This is the village mean of the raw 1/2-coded `savings` variable, so HIGHER = LESS saving.** Range 1.354–1.837, mean 1.613 `[D]`. Best-fitting reconstruction (mean of `savings` over surveyed individuals, excluding `-999`/`-888`) gives corr 0.996 / max abs diff 0.041 — **close but not an exact match, so the precise aggregation is not fully recoverable `[?]`** |
| `shgparticipate` | Average self-help group participation | **Verified exactly** `[D]`: fraction of the village's surveyed individuals with `shgparticipate == 1` (Yes), denominator = *all* surveyed individuals. Range 0.014–0.354 `[D]` |
| `fracGM_survey` | Fraction GM caste | ⚠️ **The label is misleading — this is not a fraction.** Its range is 1.068–3.030 (mean 2.51) `[D]`, i.e. it is the village mean of the raw `caste` code (`1 SC, 2 ST, 3 OBC, 4 GENERAL`). Best-fitting reconstruction (mean over surveyed individuals excluding `-999`) gives corr 0.9993 / max abs diff 0.076 — **close but not exact `[?]`**. The paper calls this control "caste composition" `[P]` |

The `.do` files refer to these as `sav`, `shg`, `fracGM` and `fractionL` — those are Stata's
unambiguous-abbreviation forms of `savings`, `shgparticipate`, `fracGM_survey`, `fractionLeaders` `[S]`.

### 4.2 `panel.dta` — 325 rows × 16 columns (village × trimester)

*"The empirical and simulated microfinance take up rate across villages over time"* `[R]`. Used by
`makeScienceTable2.do` `[S]`.

* 43 villages, unbalanced: `t` runs `0 … T_v` where `T_v` ranges 2–10 `[D]`.
* Both take-up variables are exactly `0` at `t = 0` for every village `[D]`.
* The 12 village-level columns are **time-invariant and numerically identical to `cross_sectional.dta`** `[D]`. `mf` and `degree_leader` are **not** in this file.

| Column | Label `[L]` | Notes |
|---|---|---|
| `village` | *(no label)* | |
| `t` | Time (Trimesters) | A trimester = 4 months `[S]` (`Main_models_1_3.m`: `T = ceil(TMonths./4) + 1`). The number of model periods equated to trimester 1 is village-specific and estimated as the number of simulated periods needed to reach the village's observed trimester-1 participation `[P]` |
| `dynamicMF_empirical` | Empirical TakeUp | Observed cumulative participation rate in the village at period `t`. Contains `NaN` in the final period for some villages `[D]` |
| `dynamicMF_simulated` | Simulated Takeup | Mean simulated participation rate over 1 000 simulations of the fitted diffusion model, with period 1 normalised to the empirical period-1 rate `[P]`. `makeScienceTable2.do` further **shifts this series in time per village** before regressing (`g scale = simulatedtakeup_t1 - 1; replace … = …[_n+scale]`) `[S]` |
| `eigenvector_centrality_leader` … `communication_centrality_leader` (7 cols) | as in §4.1 | time-invariant |
| `numHH`, `fractionLeaders` | as in §4.1 | time-invariant |
| `savings` | (mean) savings | same caveat as §4.1 |
| `shgparticipate` | (mean) shgparticipate | same as §4.1 |
| `fracGM_survey` | (mean) caste | ⚠️ the panel label confirms this is a **mean of the caste code**, not a fraction |

### 4.3 `do files/` — what each script produces `[S]`

| File | Output |
|---|---|
| `makeScienceFigure2.do` | Figure 2 panels A/B/C — take-up vs. leader degree / communication centrality / diffusion centrality. Uses `cross_sectional.dta` |
| `makeScienceTable2.do` | Table 2, time-series validation: `areg dynamicMF_empirical dynamicMF_simulated_adjust i.t if t>0, absorb(village) clust(village)`, with and without demographic controls interacted with `t`. Uses `panel.dta` |
| `makeScienceTable3.do` | Table 3, take-up vs. leader centralities with controls `numHH sav shg fracGM fractionLeader`. Uses `cross_sectional.dta` |
| `makeScienceTableS2.do` | Table S2, explaining the average centrality of leaders |
| `makeScienceTableS3.do` | Table S3, panels A (no controls) / B (`numHH` only) / C (full controls) |

Requires the user-written `outreg2` and `xi3` commands; the `.do` files expect the `.dta` files to be
in the working directory (they `use "cross_sectional.dta"`, not a path) `[S]`.

---

## 5. `Matlab Replication`

Everything in this folder is at the **household** level. Row `k` of every per-village file
corresponds to row/column `k` of that village's household adjacency matrix, i.e. to
`key_HH_vilno_##.csv` line `k` — **verified** by matching `hhcovariates##.csv` against
`household_characteristics.dta` sorted by `adjmatrix_key` `[D]`.

### 5.1 `India Networks/adjacencymatrix.mat`

MATLAB `.mat` containing a single variable `X`: a **43 × 1 cell array**, `X{g}` being the household
adjacency matrix of the `g`-th village in the `vills` list `[S]`, `[D]`.

* `uint8`, values `{0,1}`, symmetric, zero diagonal `[D]`.
* **`X{g}` is restricted to the giant component** — it equals
  `adj_allVillageRelationships_HH_vilno_v.csv` sub-setted to the rows/columns where
  `inGiant_v.csv == 1`. **Verified exactly** for the villages checked `[D]`. E.g. village 1: 182
  households in the full network, 175 in `X{1}`.
* Consequence: `X{g}` is **smaller** than every other per-village file, which are all of full village
  length. The scripts prune the others with `inGiant` immediately after loading `[S]`.

### 5.2 `India Networks/hhcovariates##.csv` — 49 villages, 6 tab-separated columns, no header

*"Number of rooms in a household, number of beds, whether the household has private/government/no
electricity, whether the household has own/common/no latrine, number of rooms per capita and number
of beds per capita"* `[R]`. Used as the covariate matrix `Z` in the logit for participation `[S]`
(`Z{counter} = W{counter}(:,1:6)`).

| Col | Meaning | Verification |
|---|---|---|
| 1 | Number of rooms | **exactly equals** `room_no` `[D]` |
| 2 | Number of beds/cots | **exactly equals** `bed_no` `[D]` |
| 3 | Electricity: `1` private, `2` government, `3` none | **exactly equals** `electricity` `[D]` |
| 4 | Latrine: `1` owned, `2` common, `3` none | **exactly equals** `latrine` `[D]` |
| 5 | Rooms per capita | `= col1 / household size` `[D]` |
| 6 | Beds per capita | `= col2 / household size` `[D]` |

Columns 5 and 6 imply an identical, integer household size (`col1/col5 == col2/col6`, verified) `[D]`.
The **household-size variable itself is not shipped anywhere in the package** `[D]`, but it can be
recovered as `round(col1/col5)` for households with at least one room. The paper describes `X` in the
participation logit as "quality of access to electricity, quality of latrines, number of beds, number
of rooms" `[P]`, consistent with these columns.

### 5.3 `India Networks/MF##.csv` — 49 villages, 1 column, no header

*"Microfinance participation data"* `[R]`. `0`/`1` per household `[D]`; `1` = the household joined
BSS's microfinance programme. This is the **only** source of take-up at household level in the
package (see the README discrepancy in §7). Matched to the household census from BSS's administrative
records `[P]`.

Loaded as `TakeUp{}` and used to form the village empirical rate
`EmpRate = mean(TakeUp(~leaders))` `[S]` — i.e. **the denominator is non-leader households**, matching
`mf` in `cross_sectional.dta` `[D]`.

### 5.4 `India Networks/HHhasALeader##.csv` — 49 villages, 2 tab-separated columns, no header

*"Whether a household member is a village leader"* `[R]`.

| Col | Meaning |
|---|---|
| 1 | Row index, `1 … N` — simply the household's position in the village adjacency matrix `[D]` |
| 2 | `0`/`1` leader dummy. **Exactly equals** `household_characteristics.dta`'s `leader`, in `adjmatrix_key` order `[D]`. The script reads only this column: `leaders{counter} = templeaders(:,2)` `[S]` |

`leaders.zip` is a zip archive of these same 49 files `[D]`; it contains no additional data.

### 5.5 `India Networks/inGiant##.csv` — 43 villages, 1 column, no header

*"The giant component data"* `[S]` (`Main_models_1_3.m` comment). `0`/`1` per household; `1` = the
household lies in the **giant (largest) connected component** of the village household network `[D]`.
Length = full village household count; `sum(inGiant_v)` equals the dimension of `X{g}` — **verified** `[D]`.
Every other per-village vector/matrix is pruned by this mask before estimation `[S]`.

### 5.6 The six `*Omega_*##.csv` families — 43 villages each, `N × N` comma-separated matrices

These are the **endorsement weighting matrices** used only by the endorsement models (models 2 and 4).
`Main_models_2_4.m` maps the filenames onto the three weighting schemes named in `endorsement_model.m`
`[S]`:

| File family | Loaded as | Scheme (comment in `endorsement_model.m`) |
|---|---|---|
| `Omega_abs##` / `Omega_rel##` | `Omega_E` | **Eigenvector centrality** weighting |
| `DOmega_abs##` / `DOmega_rel##` | `Omega_D` | **Degree** weighting |
| `NOmega_abs##` / `NOmega_rel##` | `Omega_N` | **Naive** (unweighted) |

The entries themselves — **verified numerically for villages 1, 6 and 45** `[D]`, on the full-village
household network `g = adj_allVillageRelationships_HH_vilno_##`:

* The **support** of every `Ω` matrix is exactly the edge set of `g` (`Ω[i,j] ≠ 0 ⟺ g[i,j] = 1`) `[D]`.
* **`NOmega_abs` is `g` itself** (all weights = 1) `[D]`.
* **`DOmega_abs[i,j] = degree(j)`** for `i~j` `[D]`.
* **`Omega_abs[i,j] = eigenvector centrality of j`** for `i~j` — correlation 1.000000 against the
  unit-norm principal eigenvector of `g`, ratio coefficient of variation ~1e-5 `[D]`.
* **`*_rel` is the row-normalised version of the corresponding `*_abs`** (each non-zero row sums to 1) —
  exact `[D]`.

In other words `Ω[i,j]` weights *j*'s endorsement as seen by *i*, by *j*'s importance. `_abs` = raw
weight, `_rel` = share. The model uses them as
`regressor = diag(Ω·(transmissionHist ∘ z)) ./ diag(Ω·transmissionHist)` — the weighted fraction of the
people who informed *i* who themselves participated `[S]`, corresponding to `F_it` in Eq. 1 of the
paper `[P]`.

`Main_models_2_4.m` selects the family with a hard-coded switch `relative = 1` (i.e. `_rel`) `[S]`. The
labels "abs"/"rel" are **never spelled out in `[R]` or `[P]`**; the meaning above is derived
arithmetically `[D]`.

Dimensions are the **full** village household count (not giant-component-pruned); the script prunes
them after loading `[S]`.

### 5.7 MATLAB source files `[S]`

| File | Role |
|---|---|
| `GMMDiffusion/Main_models_1_3.m` | Driver for the pure information models. `modelType = 1` (`qN = qP = q`) or `3` (`qN ≠ qP`). Grid-searches `(qN, qP)`, computes the simulated-method-of-moments criterion, optionally block-bootstraps |
| `GMMDiffusion/Main_models_2_4.m` | Driver for the information + **endorsement** models (`λ` free). `modelType = 2` or `4`. Adds the `Ω` matrices and a `lambda` grid |
| `GMMDiffusion/diffusion_model.m` | One simulation run of the information-only model |
| `GMMDiffusion/endorsement_model.m` | One simulation run of the endorsement model, in three parallel variants (E = eigenvector-, D = degree-, N = naive-weighted) |
| `GMMDiffusion/divergence_model.m`, `divergence_endorsement_model.m` | Average simulated moments over `S` runs and return the divergence `D(G,m)` from the empirical moments |
| `GMMDiffusion/moments.m` | The 5 moments (version 1): (1) share of takers with no taking neighbours, (2) take-up in the neighbourhood of *participating* leaders, (3) take-up in the neighbourhood of *non-participating* leaders, (4) covariance of taking with the share of taking neighbours, (5) — see file. Matches the moment list in `[P]` |
| `GMMDiffusion/simulaton_model3.m` | Generates the simulated take-up series that becomes `dynamicMF_simulated` in `panel.dta` *(name is misspelled in the package)* |
| `breadth.m`, `breadthdistRAL.m` | Breadth-first search / reachability + distance matrices (attributed to Olaf Sporns, Indiana University) |

Hard-coded settings worth knowing `[S]`: `S = 75` simulations per grid point; `timeVector =
'trimesters'`; `TMonths` = per-village months of BSS exposure (a 43-long literal vector inside both
`Main_models_*.m`); `T = ceil(TMonths/4) + 1`.

> **Known quirk in the shipped script**: `Main_models_2_4.m` line 146 assigns
> `Omega_D{counter} = Omega_N{counter}(...)` — the degree-weighted matrix is overwritten by the naive
> one during the giant-component pruning step `[S]`. Whether this is a typo or intentional is
> **not documented `[?]`**. It means the "D" results from an unmodified run of that script are
> naive-weighted, not degree-weighted.

---

## 6. `Survey Instruments`

| File | Content |
|---|---|
| `household.doc` | Household questionnaire — "TO BE ASKED OF ANY ADULT MEMBER OF THE HOUSEHOLD". Source for the `household_characteristics` codes (roof 3.1, rooms 3.2, beds 3.3, electricity 3.4, latrine 3.5, tenure 3.6, religion 2.0, caste 3.2) `[S]` |
| `individual.doc` | Individual questionnaire ("SOCIAL NETWORKS"). Contains the 12 network name-generator modules and the demographic questions numbered 0.0–29.1. Source for the `resp_gend` coding `[S]` |
| `village.pdf`, `village-horizontal.pdf`, `villageelderl.pdf` | Village-level questionnaires (village leadership, presence of NGOs and SHGs, etc. `[P]`) |

Both `.doc` files are bilingual English/Kannada. **`[R]` §4 refers to these as `household-final.doc`
and `individual-final.doc`; the actual filenames are `household.doc` and `individual.doc`** `[D]`.

**No dataset derived from the three village-level PDF questionnaires is included in this package** `[D]` —
the village-module data referenced in `[P]` are not released here.

---

## 7. Known problems, discrepancies and open questions

Collected in one place. Each is either a documented mismatch or an explicitly-flagged unknown.

1. **README overstates `household_characteristics.dta`.** `[R]` §3.4 says the file contains a
   microfinance-client dummy. It does not — the file has 19 columns and none of them is take-up `[D]`.
   Use `Matlab Replication/India Networks/MF##.csv`, which covers only 49 of the 75 villages.

2. **13 `andRelationships_HH` matrices are indexed differently from every other matrix — but they are
   usable.** These files are larger than their key file `[D]`:
   `adj_andRelationships_HH_vilno_` **1** (183 vs 182), **3** (294 vs 292), **4** (241 vs 239),
   **6** (115 vs 114), **18** (232 vs 230), **20** (158 vs 156), **21** (205 vs 202),
   **23** (256 vs 254), **26** (127 vs 126), **45** (223 vs 221), **58** (179 vs 178),
   **61** (127 vs 122), **71** (299 vs 298). All other **2 087** adjacency files match their key
   exactly `[D]`.

   **Cause, established `[D]`:** these 13 files are indexed by **raw `HHnum_in_village`, `1…max`**,
   whereas every other household matrix is indexed by **rank** (i.e. by the key file). They are
   precisely the 13 villages with gaps in `HHnum_in_village` (§2.5), and the matrix dimension equals
   `max(HHnum_in_village)` in every case.

   **Fix, verified for all 13 `[D]`:** sub-set rows and columns by `key_HH_vilno_##.csv - 1`. The result
   equals the intersection of the 12 relation matrices exactly, and the dropped rows/columns (the
   skipped household numbers) are entirely zero:
   ```python
   idx = key_HH - 1
   AND_fixed = AND_raw[np.ix_(idx, idx)]   # == elementwise min of the 12 relation matrices
   ```
   Whether the different indexing was intentional is still **undocumented `[?]`**.

3. **Self-loops in individual-level networks.** 69 of 75 villages have at least one; 319 in the
   individual union network alone `[D]`. Household-level matrices are clean `[D]`. Undocumented `[?]`.

4. **Duplicate `pid`.** `pid = 6109803` appears on two rows of `individual_characteristics.dta` `[D]`.
   Undocumented `[?]`.

5. **`fracGM_survey` is mislabelled.** Labelled "Fraction GM caste" but valued 1.07–3.03 — it is a mean
   of the caste code, not a fraction `[D]`. `panel.dta`'s label, "(mean) caste", is the accurate one.

6. **`savings` is reverse-signed.** It is a mean of a `1 = Yes / 2 = No` variable, so larger values mean
   *less* saving `[D]`. Easy to get backwards in a regression.

7. **`savings` and `fracGM_survey` do not reproduce exactly** from the released individual file
   (corr 0.996 and 0.9993 respectively, max abs deviations 0.041 and 0.076) `[D]`. `mf`, `numHH`,
   `fractionLeaders` and `shgparticipate` **do** reproduce exactly `[D]`. The residual discrepancy is
   **unexplained `[?]`** — the two village-level files were likely built from a slightly different
   vintage of the individual data.

8. **Out-of-label numeric codes.** `ownrent` takes values `6` (12 rows, probably the questionnaire's
   `OTHER (SPECIFY)` `[?]`) and `0` (1 row, **unknown `[?]`**); `work_outside` takes a `0` that has no
   label `[?]` `[D]`.

9. **Two different caste codings.** The household questionnaire uses `OBC=1, SC=2, GENERAL=3, ST=4`,
   the individual questionnaire uses `SC=1, ST=2, OBC=3, GENERAL=4` `[S]`. `castesubcaste` is stored as
   text and `caste` as the individual coding, so no silent collision occurs — but do not reuse a mapping
   across the two files.

10. **`degree_leader` "(corrected)".** The missing-data correction for the ~46% sampling rate is
    referenced but not specified in the package or the main paper `[?]`; see the paper's supplementary
    materials.

11. **Six extra villages** (`10, 28, 37, 41, 66, 77`) have `hhcovariates`/`MF`/`HHhasALeader` files but
    no `inGiant` or `Ω` files and appear in no script `[D]`. Their status is **undocumented `[?]`**.

12. **`Main_models_2_4.m` overwrites `Omega_D` with `Omega_N`** (§5.6) `[S]`, `[?]`.

13. **README filename drift**: `household-final.doc` / `individual-final.doc` vs. the shipped
    `household.doc` / `individual.doc` `[D]`.

14. **Leftover LibreOffice lock files** `.~lock.household_characteristics.csv#` and
    `.~lock.individual_characteristics.csv#` sit in `Data/2. Demographics and Outcomes/`. They are not
    data and can be deleted `[D]`.

---

## 8. Joining the pieces

```
                      key_vilno_##.csv[k]  ──►  pid
individual adjacency ─────────────────────────────────────►  individual_characteristics.dta
   row/col k                              (or: adjmatrix_key column, same thing)

                      key_HH_vilno_##.csv[k]  ──►  HHnum_in_village
household adjacency ──────────────────────────────────────►  household_characteristics.dta
   row/col k                              hhid = village*1000 + HHnum_in_village

household adjacency ──────────────────────────────────────►  hhcovariates##.csv[k]
   row/col k          (same row order, verified)              MF##.csv[k]
                                                              HHhasALeader##.csv[k]
                                                              inGiant##.csv[k]
                                                              Omega_*##.csv[k, ·]

adjacencymatrix.mat X{g}  =  adj_allVillageRelationships_HH_vilno_v  restricted to  inGiant_v == 1

individual ──► household:   individual_characteristics.hhid  =  household_characteristics.hhid
household  ──► village:     village column, or hhid // 1000
```

Worked example from `[R]`, re-verified `[D]`: in `adj_templecompany_vilno_1.csv`, row 5 / column 8 is
`1`; `key_vilno_1.csv` line 5 is `100201` and line 8 is `100204`; both `pid`s resolve in
`individual_characteristics.dta` (village 1, household 1002).

---

*Compiled from `datav4.0/README.pdf`, the *Science* 2013 paper, the package's Stata and MATLAB
replication scripts, the two `.doc` survey instruments, and direct inspection of the data files.
Every `[D]` claim above was produced by a check run against this copy of the package.*
