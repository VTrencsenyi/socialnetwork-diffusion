"""Curated per-village subcaste alias maps -- `docs/household_design.md` §7.4.

`tools.build_household_features()` writes `subcaste` exactly as the survey
stored it, because normalising it is a judgement call and a string-similarity
heuristic would make that judgement silently (§4.3b). This module is the
judgement, written down: a hand-checked map of raw spelling -> canonical
spelling, one per village, applied by a script so the merge is reproducible and
reversible rather than a hand-edited CSV.

    python src/subcaste.py --villages 6 73
    -> output/features/CLEANED_hh_features_6.csv
       output/features/CLEANED_hh_features_73.csv

The cleaned file is the feature table byte-for-byte, except that `subcaste`
holds the canonical spelling and a new `subcaste_raw` column immediately after
it holds what the survey said. Keeping the raw string is §7.4's condition for
the map landing at all: the merge stays checkable and undoable, and nothing
downstream has to trust this file over the original.

Four rules, applied in this order, and they are the whole of what is claimed:

1. **Orthography only.** Two strings merge when they are spellings of the same
   name -- a plural or honorific suffix (`KURUBAS`, `NAYAKARU`), a transliteration
   difference (`VAKKALIGA`/`VOKKALIGA`, `BAJANTRI`/`BAJANTHRI`), an initialism
   (`A.K` -> `ADI KARNATAKA`), or a dropped syllable (`JENUKURBAS`). Two *names*
   for one community do not merge, however sure the ethnography is; those go in
   `KEPT_APART` with the reason, so the decision is on the record either way.
2. **`caste` is a cross-check, not a constraint.** The individual file's
   administrative caste (OBC / SC / ST / General) is an independent read on
   whether two strings name one group, and 11 of village 6's 14 merges plus all
   3 of village 73's agree with it exactly. It is not treated as binding,
   because it is not always a clean partition of the raw strings to begin with:
   in village 6 the *single* spelling `VOKKALIGA` is reported as GENERAL by two
   respondents and OBC by ten, and `ROMAN CATHOLIC` spans GENERAL, `DO NOT KNOW`
   and blank. The three merges that cross a caste label (`VOKKALIGA`,
   `BALAJIGAS`, `BUDUGA JANGAMA`) are named in `KEPT_APART`'s notes with the
   reading taken. Village 73 is the easier case and shows what the cross-check
   looks like when it works: there each of the 17 spellings carries exactly one
   caste, so every merge below either agrees with it or is forbidden outright.
3. **The canonical form is attested.** It is always a string that occurs in this
   village, so the map can be checked against the raw file without knowing how
   the name "should" be spelled. Ties break toward the singular over the plural,
   and then toward the fuller spelling.
4. **Silence is pass-through.** A string not in the map is returned unchanged,
   never guessed at. `report()` lists what fell through so an unreviewed village
   is visible rather than silently half-cleaned.

Missing codes are handled here too: `-999` is the survey's missing marker and
becomes `pd.NA` rather than a subcaste of its own.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Survey non-answers, and only the documented ones. `DO NOT KNOW` / `REFUSE TO
# SAY` are the labelled forms used elsewhere in the individual file; `-999` is
# the raw code and is the one that actually occurs in `subcaste` (once, in
# village 6). Nothing else is treated as missing -- `OTHERS` and the like are
# real answers a respondent gave, and blanking them here would be exactly the
# silent judgement this module exists to avoid.
MISSING_CODES = frozenset({"", "-999", "-999.0", "NA", "N/A", "DO NOT KNOW", "REFUSE TO SAY"})

# village -> canonical spelling -> the raw spellings merged into it.
#
# Village 6: 49 distinct strings over 110 interviewed individuals (plus `-999`),
# which collapse to 28 groups; at household level, 28 raw strings over 44
# surveyed households collapse to 21. Counts in the comments are individuals,
# from `individual_characteristics.csv`, and are what fixed each canonical form.
ALIASES: dict[int, dict[str, tuple[str, ...]]] = {
    6: {
        # Vokkaliga / Vakkaliga is the same community bundle-wide (4,204 vs 637)
        # -- household_design.md §4.3b. Here 12 vs 2, both GENERAL and OBC rows.
        "VOKKALIGA": ("VAKKALIGA",),
        # Naik / Naika / Nayak / Nayaka, plus the Kannada honorific plural
        # `-ru`. All ST, all one name. The largest group in the village once
        # merged (20 individuals, 8 households) and the clearest single case
        # for cleaning at all: five spellings, no ambiguity.
        "NAYAKA": ("NAIK", "NAIKA", "NAYAK", "NAYAKARU"),
        "BAJANTHRI": ("BAJANTRI",),
        # Singular attested (2) though the plural is commoner (3) -- rule 3.
        "KURUBA": ("KURUBAS",),
        # Distinct from KURUBA: Jenu Kuruba is a separate ST community, not a
        # spelling of the OBC Kuruba. Only the dropped-syllable pair merges.
        "JENUKURUBAS": ("JENUKURBAS",),
        # Only plural forms are attested (`-s` 2, `-ru` 1), so the canonical is
        # the commoner plural rather than an invented singular -- rule 3.
        "BALAJIGAS": ("BALAJIGARU",),
        "LINGAYATH": ("LINGAYAT", "LINGAYATS"),
        "KORACHA": ("KORACHARU",),
        "BUDUGA JANGAMA": ("BUDAGA JANGAMA",),
        "HOLIYA": ("HOLIYARU",),
        # The initialism case, and the reason the map is worth having at all:
        # `A.K` is unreadable to an agent and identical to `ADI KARNATAKA` to a
        # reader who knows the setting. Both SC, one household each.
        "ADI KARNATAKA": ("A.K",),
        # Chennadasar, three ways, all SC. `KHANNADAS` is the weakest merge in
        # the map: it differs from `CHANNADAS-` by its first consonant (K vs
        # CH), not by a suffix. Kept because the remaining eight characters
        # agree, the caste agrees, and there is no Karnataka SC community the
        # `K` spelling would otherwise name -- but it is the first line to
        # revisit if the map is ever wrong.
        "CHANNADASARU": ("CHANNDASAR", "KHANNADAS"),
        # Madivala, Agasa and Dhobi are the Kannada, Kannada-regional and Hindi
        # names of the washerman caste; here they appear as qualifiers on one
        # head word, so the merge is on the shared stem rather than on the
        # synonymy. All OBC.
        "MADIVAL": ("MADIVAL AGASAR", "MADIVAL DHOBI"),
        # Sayyid, three transliterations, all OBC. `SAIHADH` is a phonetic
        # spelling rather than a suffix variant -- weaker than `SAYYAD`, and
        # noted as such.
        "SYED": ("SAYYAD", "SAIHADH"),
    },
    # Village 73: 17 distinct strings over 217 interviewed individuals (no
    # `-999`), which collapse to 12 groups; at household level, 15 raw strings
    # over 94 surveyed households collapse to 11, and singletons fall from 8 to
    # 3. Counts in the comments are individuals, from
    # `individual_characteristics.csv`, as in village 6.
    #
    # Unlike village 6, `caste` is a *clean* partition of the raw strings here:
    # every one of the 17 spellings carries exactly one administrative caste, so
    # rule 2's cross-check either agrees with a merge or forbids it outright, and
    # all three merges below agree with it exactly.
    73: {
        # The same Naik / Nayak / Nayaka bundle as village 6, three spellings
        # instead of five, all SCHEDULED TRIBE. `NAYAKA` is the canonical there
        # too, which matters: the two villages' cleaned tables are read side by
        # side, and a group that is NAYAKA in one and NAYAK in the other would be
        # two groups to anything comparing them. Rule 3 is satisfied
        # independently -- `NAYAKA` is attested here (1 individual, 1 household)
        # and is the fuller spelling -- so the cross-village agreement is a
        # consequence of the rule, not a thumb on it. 20 individuals once
        # merged, 8 households.
        "NAYAKA": ("NAIK", "NAYAK"),
        # Brahmin, three transliterations, all GENERAL. No plural and no
        # initialism to break the tie, so this falls to the same reading as
        # `BALAJIGAS` in village 6: frequency decides, and `BRAMANA` has 4 of the
        # 6 individuals against one each for the others. `BRAMHINA` is arguably
        # the fuller spelling -- it keeps both the H-cluster and the final -A --
        # and is the one line here to revisit if the canonical looks wrong; note
        # that `BRAHMIN`, the spelling a reader would reach for first, is
        # deliberately *not* canonical, because choosing it would be knowing how
        # the name should be spelled rather than reading what this village said.
        # `BRAHMIN` occurs in the individual file only, so no household row turns
        # on it.
        "BRAMANA": ("BRAHMIN", "BRAMHINA"),
        # Kannada honorific plural `-ru`, exactly as `KORACHA`/`KORACHARU` in
        # village 6, both OBC. Singular attested (2) and commoner than the plural
        # (1), so rule 3's two tiebreaks agree.
        "THIGALA": ("THIGALARU",),
        # Not merged, and not for want of a candidate: `BHAJANTHRI` (1, SC) is
        # the same washerman-adjacent musician caste that village 6 spells
        # `BAJANTHRI`/`BAJANTRI`, but it is the only spelling attested here, so
        # rule 4 returns it unchanged and rule 3 forbids importing village 6's
        # canonical. The consequence is real and is the price of per-village
        # maps: `BAJANTHRI` in village 6 and `BHAJANTHRI` in village 73 are one
        # community under two strings, and anything pooling villages must merge
        # them itself rather than trust either cleaned file to have done it.
        # `ADI KARNATAKA` (64, SC), `VANNIKULA` (64, OBC), `VOKKALIGA` (33, OBC),
        # `BHOVI` (18, SC) and `LINGAYATH` (2, GENERAL) likewise each occur in a
        # single spelling -- village 6's `A.K` and `VAKKALIGA` have no analogue
        # here -- so the four largest groups in the village are untouched by this
        # map and the merges only ever reach its tail.
    },
}

# Pairs a reader would reasonably expect to be merged, and deliberately are not.
# Recorded because leaving a merge out is as much a decision as making one, and
# an undocumented omission looks like an oversight.
KEPT_APART: dict[int, tuple[tuple[str, str, str], ...]] = {
    6: (
        (
            "NAYAKA",
            "WALMIKI",
            "Karnataka lists the ST community as 'Valmiki Nayaka', so these are "
            "arguably one group -- but 'Walmiki' is a second name, not a spelling "
            "of 'Nayaka'. Rule 1 stops here. One individual, no household.",
        ),
        (
            "BALAJIGAS",
            "CHITTI BANAJIGA",
            "Balajiga and Banajiga are the same trading-caste name, but this "
            "string carries a qualifier ('Chitti') that the others do not, and "
            "merging it would drop a distinction the respondent made.",
        ),
        (
            "SYED",
            "SHEIKH",
            "Both Muslim lineage names in the same OBC block, and neither is a "
            "spelling of the other.",
        ),
        (
            "MUSLIMS",
            "SHIYA",
            "'Muslims' is a religion-level answer and 'Shiya' a sect; neither is "
            "a subcaste name, and collapsing them would invent a group the survey "
            "did not record. Left as given -- see the note in the design doc on "
            "what this column is and is not.",
        ),
        (
            "MADIVAL",
            "AGARARU",
            "One letter from 'Agasaru', which would put it in the Madival group, "
            "but the reading is a guess and the string is intact as it stands.",
        ),
        (
            "BUDUGA JANGAMA",
            "BUDAGA JANGAMA",
            "These two ARE merged (rule 1), and so are VOKKALIGA/VAKKALIGA and "
            "BALAJIGAS/BALAJIGARU. Listed here because all three cross a caste "
            "label -- ST/SC for the Jangamas, GENERAL/OBC for the other two. The "
            "GENERAL/OBC pair is the same self-reporting drift that already splits "
            "the single spelling VOKKALIGA (2 GENERAL, 10 OBC), so it says nothing "
            "about the merge; Budga Jangam is ST in Karnataka's list, so the SC "
            "row is read as respondent error rather than as a second group.",
        ),
    ),
    73: (
        (
            "NAYAKA",
            "VALMIKI NAYAKA",
            "The hardest call in this village, and it is settled by consistency "
            "with village 6's WALMIKI entry above rather than by ethnography: "
            "Karnataka lists the ST community as 'Valmiki Nayaka', and this "
            "string carries the canonical head word verbatim, so the argument for "
            "merging is stronger here than it was there. It is still refused. "
            "Rule 1 admits suffixes, transliterations, initialisms and dropped "
            "syllables, not a qualifying prefix; MADIVAL AGASAR -> MADIVAL is the "
            "one qualified merge in the map and it absorbs a *synonym* of its head "
            "word, where 'Valmiki' is a second name. Merging would also drop a "
            "distinction the respondent made, which is exactly why CHITTI BANAJIGA "
            "stayed out of BALAJIGAS. 2 individuals, 2 households; subcaste_raw "
            "makes the opposite reading a one-line change.",
        ),
        (
            "NAYAKA",
            "BYADAR NAYAKA",
            "Same shape as VALMIKI NAYAKA and refused for the same reason, with "
            "one addition that makes it the clearer refusal of the two: Byadar / "
            "Bedar is a distinct ST community in Karnataka's list, not a synonym "
            "of Nayaka, so merging would not be a spelling decision at all. Both "
            "are SCHEDULED TRIBE, so rule 2 cannot separate them. 1 individual, "
            "and no household -- the string reaches the cleaned file only through "
            "the individual-level counts quoted above.",
        ),
        (
            "VANNIKULA",
            "THIGALA",
            "'Vanniyakula Kshatriya' is the formal name of the community Karnataka "
            "also lists as Tigala, so these two arguably name one OBC group -- and "
            "they are the second-largest and one of the smallest groups in the "
            "village, so the merge would visibly change the distribution (28 + 3 "
            "households). Refused as two names rather than two spellings, which is "
            "rule 1's whole point; recorded here because a reader who knows the "
            "setting will expect it and should see that it was decided, not "
            "missed.",
        ),
        (
            "VOKKALIGA",
            "GOWDA",
            "'Gowda' is a title carried by Vokkaliga households (and by others), "
            "not a spelling of 'Vokkaliga'. Both OBC, so rule 2 is silent. 3 "
            "individuals, 1 household. Left as given for the same reason MUSLIMS "
            "and SHIYA were in village 6: the string records what the respondent "
            "answered, and a title is not evidence of the subcaste underneath it.",
        ),
    ),
}

_WS = re.compile(r"\s+")


def normalise(value: object) -> str | None:
    """Uppercase, strip, collapse internal whitespace. `None` for a non-answer."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return None
    text = _WS.sub(" ", str(value).strip().upper())
    return None if text in MISSING_CODES else text


def alias_map(village: int) -> dict[str, str]:
    """Flat raw -> canonical map for `village`, validated on the way out.

    Raises `KeyError` for a village with no reviewed map, rather than returning
    an empty one: a village that has not been eyeballed must not quietly produce
    a "cleaned" file identical to the raw one.
    """
    if village not in ALIASES:
        raise KeyError(
            f"no reviewed alias map for village {village}; add one to subcaste.ALIASES "
            "after eyeballing its subcaste levels (docs/household_design.md §7.4)"
        )
    groups = ALIASES[village]
    flat: dict[str, str] = {}
    for canonical, variants in groups.items():
        canon = normalise(canonical)
        assert canon is not None
        for variant in variants:
            raw = normalise(variant)
            assert raw is not None
            if raw in flat:
                raise ValueError(f"village {village}: {raw!r} is mapped twice ({flat[raw]!r}, {canon!r})")
            if raw in {normalise(c) for c in groups}:
                raise ValueError(f"village {village}: {raw!r} is both a canonical form and a variant")
            flat[raw] = canon
    return flat


def clean_series(values: pd.Series, village: int) -> pd.Series:
    """`values` with spellings merged. Unmapped strings pass through unchanged."""
    mapping = alias_map(village)
    out = values.map(normalise)
    return out.map(lambda s: mapping.get(s, s) if s is not None else pd.NA).astype("string")


def report(values: pd.Series, village: int) -> str:
    """Human-readable summary of what the map did to `values`."""
    raw = values.map(normalise)
    cleaned = clean_series(values, village)
    mapping = alias_map(village)

    lines = [
        f"village {village}: {raw.notna().sum()} households with a subcaste, "
        f"{raw.dropna().nunique()} raw levels -> {cleaned.dropna().nunique()} cleaned levels",
    ]
    changed = raw.notna() & (raw != cleaned)
    for canonical in sorted(set(cleaned.dropna())):
        members = sorted(set(raw[cleaned == canonical].dropna()))
        n = int((cleaned == canonical).sum())
        if len(members) > 1:
            lines.append(f"  {canonical:<16} <- {', '.join(members)}  ({n} households)")
        elif members and members[0] != canonical:
            # A rename rather than a merge: nothing else in this village carries
            # the canonical spelling, but the row still changed and has to show.
            lines.append(f"  {canonical:<16} <- {members[0]}  ({n} households, renamed only)")
    unmapped = sorted(set(raw.dropna()) - set(mapping) - set(mapping.values()))
    if unmapped:
        lines.append(f"  unmapped, passed through ({len(unmapped)}): {', '.join(unmapped)}")
    lines.append(f"  {int(changed.sum())} household rows rewritten")
    return "\n".join(lines)


def clean_features(
    village: int,
    features_dir: Path | str = Path("output/features"),
    output_dir: Path | str | None = None,
    drop_in: bool = False,
) -> tuple[pd.DataFrame, list[Path]]:
    """Read `hh_features_<village>.csv`, write `CLEANED_hh_features_<village>.csv`.

    Everything but `subcaste` is copied through untouched -- including
    `occupation_head`, whose free-text variants are real answers rather than
    spellings of one answer (`docs/household_design.md` §4.3c). Column order is
    preserved, with `subcaste_raw` inserted directly after `subcaste`.

    `profiler`, `game_master` and `tools` all address the feature table by the
    fixed name `hh_features_<village>.csv`, so the `CLEANED_` file is an
    artefact to read and diff rather than something they will pick up. With
    `drop_in=True` the same frame is written a second time under the plain name,
    which makes `--features-dir <output_dir>` switch the whole pipeline onto the
    cleaned table with no code change. Never write that copy back into the
    directory holding the originals -- the guard below refuses.
    """
    features_dir = Path(features_dir)
    output_dir = Path(output_dir) if output_dir is not None else features_dir
    src = features_dir / f"hh_features_{village}.csv"
    # Nullable dtypes throughout, so the round trip is byte-identical: default
    # inference turns `capita` into a float the moment one household's size is
    # unrecoverable, and the cleaned file would then differ from the raw one in
    # a column this module has no business touching.
    df = pd.read_csv(src, dtype_backend="numpy_nullable")

    raw = df["subcaste"].copy()
    df["subcaste"] = clean_series(raw, village)
    df.insert(df.columns.get_loc("subcaste") + 1, "subcaste_raw", raw)

    output_dir.mkdir(parents=True, exist_ok=True)
    written = [output_dir / f"CLEANED_hh_features_{village}.csv"]
    if drop_in:
        if output_dir.resolve() == features_dir.resolve():
            raise ValueError(
                "--drop-in would overwrite the raw feature table; give -o a different directory"
            )
        written.append(output_dir / f"hh_features_{village}.csv")
    for dest in written:
        df.to_csv(dest, index=False)
    return df, written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--villages",
        type=int,
        nargs="+",
        default=sorted(ALIASES),
        help="villages to clean (default: every village with a reviewed alias map)",
    )
    p.add_argument("--features-dir", type=Path, default=Path("output/features"))
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="directory for CLEANED_hh_features_<village>.csv (default: --features-dir)",
    )
    p.add_argument(
        "--drop-in",
        action="store_true",
        help="also write the cleaned table as hh_features_<village>.csv, so "
        "--features-dir <output-dir> switches the pipeline onto it (needs -o)",
    )
    a = p.parse_args(argv)

    for village in a.villages:
        try:
            df, written = clean_features(village, a.features_dir, a.output_dir, a.drop_in)
        except (KeyError, ValueError, FileNotFoundError) as e:
            print(f"error: village {village}: {e}", file=sys.stderr)
            return 1
        print(report(df["subcaste_raw"], village))
        for dest in written:
            print(f"  -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
