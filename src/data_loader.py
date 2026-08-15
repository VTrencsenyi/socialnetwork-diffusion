"""Load and present the ground-truth data for a single village.

Data source: "The Diffusion of Microfinance" replication bundle (datav4.0),
Banerjee, Chandrasekhar, Duflo & Jackson (2013), Science.

This module deliberately reads from a small, explicitly chosen set of files.
See `docs/data_notes.md` for which files are used, which are ignored, and why.

Files used, per village V
-------------------------
Data/1. Network Data/Adjacency Matrices/adj_<net>_HH_vilno_V.csv
    n x n symmetric 0/1 household adjacency matrix. Row i corresponds to
    adjmatrix_key i+1.
Data/1. Network Data/Adjacency Matrix Keys/key_HH_vilno_V.csv
    n rows; row i gives the HHnum_in_village of adjacency row i.
Data/2. Demographics and Outcomes/household_characteristics.dta
    All 14,904 households in 75 villages. Dwelling attributes, caste,
    religion, hhSurveyed flag, and the `leader` dummy (our seeds).
Matlab Replication/India Networks/MFV.csv
    n rows of 0/1: did any member of this household take up microfinance.
    THIS IS THE HOUSEHOLD-LEVEL GROUND TRUTH.
Matlab Replication/India Networks/HHhasALeaderV.csv
    n rows of "<idx>\t<0|1>". Redundant with household_characteristics.leader;
    loaded only to cross-check.
Matlab Replication/India Networks/inGiantV.csv
    n rows of 0/1: is this household in the giant component. (43 villages only.)
Stata Replication/data/panel.dta
    Village x time take-up. `dynamicMF_empirical` is the observed adoption
    curve; `dynamicMF_simulated` is BCDJ's own structural model's prediction,
    which we keep as a published benchmark.
Stata Replication/data/cross_sectional.dta
    Village-level final take-up `mf` plus leader centrality measures.

Three traps, all verified empirically across the 43 analysis villages
--------------------------------------------------------------------
1. `cross_sectional.mf` is take-up among NON-LEADER households only. It is
   reproduced exactly from MFV.csv for all 43 villages.
2. `panel.dynamicMF_empirical` is take-up among ALL households.
3. `panel.dta` and MFV.csv are not on the same household population. The
   panel is computed on the full village census; MFV.csv is aligned to the
   network sample, which drops a handful of households. For 31 of 43
   villages the two agree exactly; for the other 12 the final values differ
   by 1-4 households (worst case village 57: panel 35/212 = 16.51% vs
   MF57.csv 31/208 = 14.90%). This is a property of the data, not an error,
   so it is reported as a warning rather than raised.

   Consequence for validation: if you score *who* adopts against MFV.csv but
   score the adoption *curve* against the panel, your two targets disagree
   slightly. `Village.adoption_curve()` therefore also returns the panel
   rescaled onto the MFV.csv population, so the curve supplies timing (its
   shape) and MFV.csv supplies the level.

Note also that the v4.0 README (section 3.4) states that
household_characteristics.dta contains a microfinance dummy. It does not.
The only household-level outcome in the bundle is MFV.csv.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "diffusion-science-data" / "datav4.0"

# The 12 surveyed relationships plus the union ("all") and intersection ("and").
NETWORK_TYPES = (
    "borrowmoney",
    "giveadvice",
    "helpdecision",
    "keroricecome",
    "keroricego",
    "lendmoney",
    "medic",
    "nonrel",
    "rel",
    "templecompany",
    "visitcome",
    "visitgo",
    "allVillageRelationships",
    "andRelationships",
)

# Villages with a household-level MF outcome file but excluded from the
# paper's 43-village analysis sample (no inGiant / cross_sectional row).
EXTRA_MF_VILLAGES = (10, 28, 37, 41, 66, 77)


class DataError(RuntimeError):
    """Raised when the loaded files fail an integrity check."""


# --------------------------------------------------------------------------
# Cached whole-bundle reads (the .dta files cover all villages)
# --------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _households_all(root: str) -> pd.DataFrame:
    df = pd.read_stata(Path(root) / "Data/2. Demographics and Outcomes/household_characteristics.dta")
    return df.astype({"village": "int32", "adjmatrix_key": "int32", "HHnum_in_village": "int32"})


@lru_cache(maxsize=None)
def _individuals_all(root: str) -> pd.DataFrame:
    df = pd.read_stata(Path(root) / "Data/2. Demographics and Outcomes/individual_characteristics.dta")
    return df.astype({"village": "int32"})


@lru_cache(maxsize=None)
def _panel_all(root: str) -> pd.DataFrame:
    return pd.read_stata(Path(root) / "Stata Replication/data/panel.dta").astype({"village": "int32"})


@lru_cache(maxsize=None)
def _cross_sectional_all(root: str) -> pd.DataFrame:
    return pd.read_stata(Path(root) / "Stata Replication/data/cross_sectional.dta").astype({"village": "int32"})


# --------------------------------------------------------------------------
# Village container
# --------------------------------------------------------------------------


@dataclass
class Village:
    """Everything needed to simulate and validate diffusion in one village.

    All array-valued fields are length n and share a single index convention:
    position i == adjacency matrix row i == adjmatrix_key i + 1.
    """

    village: int
    network_type: str
    adjacency: np.ndarray  # (n, n) symmetric 0/1, zero diagonal
    households: pd.DataFrame  # n rows, ordered by adjmatrix_key
    mf: np.ndarray  # (n,) 0/1 ground-truth adoption
    leader: np.ndarray  # (n,) 0/1 seeds
    in_giant: np.ndarray | None  # (n,) 0/1, None for the 6 extra villages
    hh_key: np.ndarray  # (n,) HHnum_in_village per adjacency row
    panel: pd.DataFrame | None  # t, dynamicMF_empirical, dynamicMF_simulated
    cross_section: pd.Series | None
    individuals: pd.DataFrame  # individual survey rows for this village
    warnings: list[str] = field(default_factory=list)

    # -- basic properties -------------------------------------------------

    @property
    def n(self) -> int:
        return self.adjacency.shape[0]

    @property
    def degree(self) -> np.ndarray:
        return self.adjacency.sum(axis=1)

    @property
    def n_edges(self) -> int:
        return int(self.adjacency.sum() // 2)

    @property
    def in_analysis_sample(self) -> bool:
        """True for the 43 villages used in the published analysis."""
        return self.cross_section is not None

    # -- the two ground-truth rates --------------------------------------

    @property
    def takeup_all(self) -> float:
        """Take-up over all households. Matches panel.dynamicMF_empirical."""
        return float(self.mf.mean())

    @property
    def takeup_nonleader(self) -> float:
        """Take-up over non-leader households. Matches cross_sectional.mf."""
        return float(self.mf[self.leader == 0].mean())

    @property
    def takeup_leader(self) -> float:
        return float(self.mf[self.leader == 1].mean())

    # -- timing -----------------------------------------------------------

    def adoption_curve(self) -> pd.DataFrame | None:
        """Observed adoption over time, plus BCDJ's simulated curve.

        Columns
        -------
        t                    period index (staggered MFI entry, so villages
                             differ in length; 3-11 periods)
        empirical            panel.dta as published, on the full village
                             census denominator
        empirical_rescaled   the same curve rescaled so its final value equals
                             this village's MFV.csv take-up. Use this when
                             comparing timing against a simulation that runs
                             on the network sample, so that curve and
                             household-level target share one population.
        n_adopters_rescaled  empirical_rescaled * n, i.e. the implied number
                             of adopting households at each t
        simulated            BCDJ's own structural model (a benchmark, not
                             ground truth)
        """
        if self.panel is None or not len(self.panel):
            return None
        df = self.panel.copy()
        emp = df.dynamicMF_empirical
        final = emp.dropna()
        if len(final) and final.iloc[-1] > 0:
            df["empirical_rescaled"] = emp * (self.takeup_all / float(final.iloc[-1]))
        else:
            df["empirical_rescaled"] = emp
        df["n_adopters_rescaled"] = df.empirical_rescaled * self.n
        return df[["t", "dynamicMF_empirical", "empirical_rescaled", "n_adopters_rescaled", "dynamicMF_simulated"]]

    # -- network helpers --------------------------------------------------

    def neighbours(self, i: int) -> np.ndarray:
        """Adjacency-row indices of household i's neighbours."""
        return np.flatnonzero(self.adjacency[i])

    def components(self) -> list[np.ndarray]:
        """Connected components, largest first, as arrays of row indices."""
        n = self.n
        seen = np.zeros(n, dtype=bool)
        out = []
        for start in range(n):
            if seen[start]:
                continue
            stack, comp = [start], []
            seen[start] = True
            while stack:
                i = stack.pop()
                comp.append(i)
                for j in self.neighbours(i):
                    if not seen[j]:
                        seen[j] = True
                        stack.append(j)
            out.append(np.array(sorted(comp)))
        return sorted(out, key=len, reverse=True)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_village(
    village: int,
    root: Path | str = DEFAULT_ROOT,
    network_type: str = "allVillageRelationships",
    check: bool = True,
) -> Village:
    """Load one village. Raises DataError if the files do not line up."""
    root = Path(root)
    if not root.exists():
        raise DataError(f"data root not found: {root}")
    if network_type not in NETWORK_TYPES:
        raise DataError(f"unknown network_type {network_type!r}; expected one of {NETWORK_TYPES}")

    net_dir = root / "Data/1. Network Data"
    mat_dir = root / "Matlab Replication/India Networks"

    adj_path = net_dir / "Adjacency Matrices" / f"adj_{network_type}_HH_vilno_{village}.csv"
    key_path = net_dir / "Adjacency Matrix Keys" / f"key_HH_vilno_{village}.csv"
    mf_path = mat_dir / f"MF{village}.csv"
    if not adj_path.exists():
        raise DataError(f"no adjacency matrix for village {village}: {adj_path}")
    if not mf_path.exists():
        raise DataError(
            f"village {village} has no household-level microfinance outcome (MF{village}.csv). "
            "Only 49 of the 75 surveyed villages have one."
        )

    adjacency = pd.read_csv(adj_path, header=None).to_numpy(dtype=np.int8)
    hh_key = pd.read_csv(key_path, header=None).to_numpy(dtype=np.int32).ravel()
    mf = pd.read_csv(mf_path, header=None).to_numpy(dtype=np.int8).ravel()

    # HHhasALeader<V>.csv is tab separated: "<row index>\t<leader 0/1>"
    leader_file = pd.read_csv(mat_dir / f"HHhasALeader{village}.csv", sep="\t", header=None)
    leader = leader_file[1].to_numpy(dtype=np.int8)

    giant_path = mat_dir / f"inGiant{village}.csv"
    in_giant = (
        pd.read_csv(giant_path, header=None).to_numpy(dtype=np.int8).ravel() if giant_path.exists() else None
    )

    households = (
        _households_all(str(root))
        .query("village == @village")
        .sort_values("adjmatrix_key")
        .reset_index(drop=True)
    )
    individuals = _individuals_all(str(root)).query("village == @village").reset_index(drop=True)

    panel = _panel_all(str(root)).query("village == @village").sort_values("t").reset_index(drop=True)
    panel = panel[["t", "dynamicMF_empirical", "dynamicMF_simulated"]] if len(panel) else None

    cs = _cross_sectional_all(str(root)).query("village == @village")
    cross_section = cs.iloc[0] if len(cs) else None

    v = Village(
        village=village,
        network_type=network_type,
        adjacency=adjacency,
        households=households,
        mf=mf,
        leader=leader,
        in_giant=in_giant,
        hh_key=hh_key,
        panel=panel,
        cross_section=cross_section,
        individuals=individuals,
    )
    if check:
        v.warnings = check_consistency(v, leader_file[0].to_numpy(dtype=np.int32))
    return v


def check_consistency(v: Village, leader_row_ids: np.ndarray | None = None) -> list[str]:
    """Verify every alignment assumption this loader relies on.

    Raises DataError on anything that would make the data silently wrong --
    a shape mismatch or an off-by-one between the adjacency matrix and the
    outcome vector would produce a plausible-looking but meaningless
    validation result.

    Returns a list of warnings for discrepancies that are real properties of
    the published data and must be reported rather than fixed.
    """
    n = v.adjacency.shape[0]

    if v.adjacency.shape[0] != v.adjacency.shape[1]:
        raise DataError(f"v{v.village}: adjacency is not square: {v.adjacency.shape}")
    if not np.array_equal(v.adjacency, v.adjacency.T):
        raise DataError(f"v{v.village}: adjacency is not symmetric")
    if set(np.unique(v.adjacency)) - {0, 1}:
        raise DataError(f"v{v.village}: adjacency is not binary")
    if v.adjacency.diagonal().any():
        raise DataError(f"v{v.village}: adjacency has self-loops")

    for name, arr in [("mf", v.mf), ("leader", v.leader), ("hh_key", v.hh_key)]:
        if len(arr) != n:
            raise DataError(f"v{v.village}: {name} has {len(arr)} rows, adjacency has {n}")
    if v.in_giant is not None and len(v.in_giant) != n:
        raise DataError(f"v{v.village}: in_giant has {len(v.in_giant)} rows, adjacency has {n}")
    if len(v.households) != n:
        raise DataError(f"v{v.village}: household_characteristics has {len(v.households)} rows, adjacency has {n}")

    # adjmatrix_key must be the dense 1..n row index. HHnum_in_village is NOT:
    # it has gaps where households were dropped, so joining on it misaligns.
    if not np.array_equal(v.households.adjmatrix_key.to_numpy(), np.arange(1, n + 1)):
        raise DataError(f"v{v.village}: adjmatrix_key is not 1..{n}")
    if leader_row_ids is not None and not np.array_equal(leader_row_ids, np.arange(1, n + 1)):
        raise DataError(f"v{v.village}: HHhasALeader row ids are not 1..{n}")
    if not np.array_equal(v.hh_key, v.households.HHnum_in_village.to_numpy()):
        raise DataError(f"v{v.village}: key file does not match HHnum_in_village")

    # Two independent copies of the seed set must agree.
    if not np.array_equal(v.leader, v.households.leader.to_numpy().astype(np.int8)):
        raise DataError(f"v{v.village}: HHhasALeader disagrees with household_characteristics.leader")

    # The published cross-section must be reproducible from MF<V>.csv exactly.
    # It is, for all 43 analysis villages -- so a mismatch here is a real error.
    warnings: list[str] = []
    if v.cross_section is not None:
        if not np.isclose(v.takeup_nonleader, float(v.cross_section.mf), atol=1e-4):
            raise DataError(
                f"v{v.village}: non-leader take-up {v.takeup_nonleader:.5f} != cross_sectional.mf "
                f"{float(v.cross_section.mf):.5f}"
            )
        if not np.isclose(v.leader.mean(), float(v.cross_section.fractionLeaders), atol=1e-4):
            raise DataError(f"v{v.village}: leader fraction disagrees with cross_sectional.fractionLeaders")
        if int(v.cross_section.numHH) != n:
            raise DataError(f"v{v.village}: cross_sectional.numHH {int(v.cross_section.numHH)} != {n}")

    # The panel is on the full village census, MF<V>.csv on the network
    # sample. They coincide for 31 of 43 villages and differ slightly for the
    # rest. Report, do not raise.
    if v.panel is not None and len(v.panel):
        final = v.panel.dynamicMF_empirical.dropna()
        if len(final) and not np.isclose(float(final.iloc[-1]), v.takeup_all, atol=1e-4):
            pf = float(final.iloc[-1])
            warnings.append(
                f"panel final take-up {pf:.2%} != MF{v.village}.csv take-up {v.takeup_all:.2%} "
                f"({pf - v.takeup_all:+.2%}, approx {abs(pf - v.takeup_all) * n:.1f} households). The panel "
                "is computed on the full village census, MF.csv on the network sample. Use "
                "adoption_curve().empirical_rescaled to put timing and household outcome on one population."
            )
    return warnings


def available_villages(root: Path | str = DEFAULT_ROOT) -> list[int]:
    """Villages that have a household-level microfinance outcome."""
    mat_dir = Path(root) / "Matlab Replication/India Networks"
    return sorted(int(p.stem[2:]) for p in mat_dir.glob("MF*.csv"))


# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------


def _bar(frac: float, width: int = 24) -> str:
    filled = int(round(frac * width))
    return "#" * filled + "." * (width - filled)


def _rule(title: str = "", width: int = 78) -> str:
    if not title:
        return "-" * width
    return f"-- {title} " + "-" * max(0, width - len(title) - 4)


def describe(v: Village) -> str:
    """Human-readable summary of one village's ground truth."""
    L = []
    n = v.n
    lead = v.leader == 1
    nonlead = ~lead

    L.append("=" * 78)
    L.append(f"VILLAGE {v.village}  |  network: {v.network_type}  |  {n} households")
    if not v.in_analysis_sample:
        L.append("  ! NOT in the published 43-village analysis sample (no inGiant / cross_sectional).")
    L.append("=" * 78)

    if v.warnings:
        L.append(_rule("DATA WARNINGS"))
        for w in v.warnings:
            L.append("  ! " + w)
        L.append("")

    # ---- ground truth -----------------------------------------------------
    L.append(_rule("GROUND TRUTH: microfinance take-up (MF%d.csv)" % v.village))
    L.append(f"  adopters                  {int(v.mf.sum()):4d} / {n}")
    L.append(f"  take-up, all households   {v.takeup_all:6.2%}   {_bar(v.takeup_all)}   <- panel.dta denominator")
    L.append(
        f"  take-up, non-leaders      {v.takeup_nonleader:6.2%}   {_bar(v.takeup_nonleader)}   "
        "<- cross_sectional.dta denominator (the paper's outcome)"
    )
    L.append(f"  take-up, leaders          {v.takeup_leader:6.2%}   {_bar(v.takeup_leader)}")
    if v.cross_section is not None:
        L.append(f"  cross_sectional.mf        {float(v.cross_section.mf):6.2%}   (reproduced: OK)")

    # ---- seeds ------------------------------------------------------------
    L.append("")
    L.append(_rule("SEEDS: leader households (injection points)"))
    deg = v.degree
    L.append(f"  leaders                   {int(lead.sum()):4d} / {n}  ({lead.mean():.1%})")
    L.append(f"  mean degree, leaders      {deg[lead].mean():6.2f}")
    L.append(f"  mean degree, non-leaders  {deg[nonlead].mean():6.2f}")
    L.append(
        "  NOTE: `leader` marks who the MFI *could* have informed, not who was verifiably told. "
        "BCDJ treat all leaders as informed; we inherit that assumption."
    )

    # ---- network ----------------------------------------------------------
    L.append("")
    L.append(_rule("NETWORK"))
    comps = v.components()
    giant = comps[0]
    L.append(f"  edges                     {v.n_edges:4d}   density {v.n_edges / (n * (n - 1) / 2):.3f}")
    L.append(f"  degree  mean {deg.mean():5.2f}  median {np.median(deg):5.1f}  min {deg.min():3d}  max {deg.max():3d}")
    L.append(f"  isolates (degree 0)       {int((deg == 0).sum()):4d}")
    L.append(f"  components                {len(comps):4d}   largest {len(giant)} ({len(giant) / n:.1%} of households)")
    if v.in_giant is not None:
        agree = (v.in_giant == 0) | np.isin(np.arange(n), giant)
        L.append(f"  inGiant{v.village}.csv agrees      {'yes' if agree.all() else 'NO — investigate'}")

    # ---- adoption curve ---------------------------------------------------
    L.append("")
    curve = v.adoption_curve()
    if curve is not None:
        L.append(_rule("ADOPTION CURVE over time (panel.dta)"))
        L.append("     t   empirical                  rescaled    adopters   BCDJ model")
        for _, r in curve.iterrows():
            emp, res, na, sim = (
                r.dynamicMF_empirical,
                r.empirical_rescaled,
                r.n_adopters_rescaled,
                r.dynamicMF_simulated,
            )
            emp_s = f"{emp:6.2%} {_bar(emp, 20)}" if pd.notna(emp) else "     - " + " " * 20
            res_s = f"{res:7.2%}" if pd.notna(res) else "      - "
            na_s = f"{na:7.1f}" if pd.notna(na) else "      - "
            sim_s = f"{sim:7.2%}" if pd.notna(sim) else "      - "
            L.append(f"  {int(r.t):4d}   {emp_s}   {res_s}   {na_s}   {sim_s}")
        L.append(
            "  `empirical` is as published; `rescaled` is anchored to this village's MF.csv total, so"
        )
        L.append(
            "  curve and household outcome share one population. `BCDJ model` is the authors' own"
        )
        L.append("  structural simulation — a published benchmark to beat, not ground truth.")
    else:
        L.append(_rule("ADOPTION CURVE"))
        L.append("  no panel data for this village — final take-up only, no timing.")

    # ---- attribute coverage ----------------------------------------------
    L.append("")
    L.append(_rule("ATTRIBUTE COVERAGE for persona construction"))
    hh = v.households
    surveyed = hh.hhSurveyed.to_numpy().astype(bool)
    n_ind_hh = v.individuals.hhid.nunique()
    L.append(f"  household attributes      {n:4d} / {n}  (100%) — dwelling, caste, religion")
    L.append(f"  hhSurveyed == 1           {int(surveyed.sum()):4d} / {n}  ({surveyed.mean():.1%})")
    L.append(
        f"  households in individual survey {n_ind_hh:4d} / {n}  ({n_ind_hh / n:.1%}) — "
        f"{len(v.individuals)} individuals"
    )
    L.append("    -> age, education, occupation, savings, SHG participation exist for this subset only.")
    L.append("  per-attribute coverage (household file, all n):")
    for col in ("hohreligion", "castesubcaste", "room_no", "bed_no", "electricity", "latrine", "ownrent"):
        s = hh[col]
        blank = int(s.isna().sum() + (s.astype(str).str.strip() == "").sum())
        flag = "  <- unusable here" if blank == n else ("  <- partial" if blank else "")
        L.append(f"    {col:<16} missing/blank {blank:4d} ({blank / n:5.1%}){flag}")
    if int((hh.castesubcaste.astype(str).str.strip() == "").sum()) == n:
        L.append(
            "    castesubcaste is empty for this village (30% of households bundle-wide). Caste for "
            "the surveyed subset must come from individual_characteristics.caste instead."
        )

    # ---- a first look at signal ------------------------------------------
    L.append("")
    L.append(_rule("FIRST LOOK: does adoption track network exposure to seeds?"))
    lead_nbrs = v.adjacency[:, lead].sum(axis=1)
    L.append("  leader-neighbours   households   take-up")
    for lo, hi, label in [(0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 100, "3+")]:
        m = nonlead & (lead_nbrs >= lo) & (lead_nbrs <= hi)
        if m.sum():
            L.append(f"    {label:>3}               {int(m.sum()):5d}      {v.mf[m].mean():6.2%}  {_bar(v.mf[m].mean(), 18)}")
    L.append("  (non-leader households only; the crude version of what the model must reproduce)")
    L.append("=" * 78)
    return "\n".join(L)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--village", type=int, default=1, help="village number (default: 1)")
    p.add_argument("--network", default="allVillageRelationships", choices=NETWORK_TYPES)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="path to datav4.0")
    p.add_argument("--list", action="store_true", help="list villages with an MF outcome and exit")
    p.add_argument("--no-check", action="store_true", help="skip integrity assertions")
    a = p.parse_args(argv)

    if a.list:
        vs = available_villages(a.root)
        print(f"{len(vs)} villages have a household-level MF outcome:")
        print("  " + " ".join(str(x) for x in vs))
        print(f"  of which not in the published analysis sample: {list(EXTRA_MF_VILLAGES)}")
        return 0

    v = load_village(a.village, root=a.root, network_type=a.network, check=not a.no_check)
    print(describe(v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
