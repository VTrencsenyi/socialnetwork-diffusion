"""Verification script for the output/pilot_v0 adoption-rate pilot (n=20 reps).

Ad hoc check against claims drafted for report/main.tex and prior session notes:
1. size of the design grid and how many designs separate the adopter/non-adopter
   samples per design_tests.csv's Fisher's-exact test, at q < 0.05
2. how many of those are "inverted" (diff < 0: the non-adopter sample was keener)
3. the A1B1 households (6026 self-adopter / 6039 self-non-adopter, and their
   informers 6032 / 6099) and whether the flip is a "wealth mismatch"
4. the 6032 (SHG+bank=yes -> 0.75) / 6099 (SHG+bank=no -> 0.20) numbers quoted in
   main.tex for design A0B1C0D0
5. the marginal (ceteris paribus) effect of the endorsement axis (C), isolated
   from the household-identity contrast that design_tests.csv actually tests

Run from the repo root: `python src/pilot/verify_pilot_v0.py`
"""
from __future__ import annotations

import glob
import itertools
import re

import pandas as pd

PILOT_DIR = "output/pilot_v0"
FEATURES_PATH = "output/features/CLEANED_hh_features_6.csv"

PROFILE_LEVELS = ("", "DEMOGRAPHIC", "NARRATIVE")
INFORMER_LEVELS = ("", "DEMOGRAPHIC", "NARRATIVE")
ENDORSEMENT_LEVELS = ("", "ENDORSEMENT")
INSTRUCTION_LEVELS = ("", "MOA", "DT")
LEVELS = {"": 0, "DEMOGRAPHIC": 1, "NARRATIVE": 2}
INSTR = {"": 0, "MOA": 1, "DT": 2}


def design_label(a, b, c, d) -> str:
    return f"A{LEVELS[a]}B{LEVELS[b]}C{int(bool(c))}D{INSTR[d]}"


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    # ---- 1 & 2: grid size, significance, inversion --------------------------
    section("1-2. Grid size and design_tests.csv significance")
    all_labels = {
        design_label(a, b, c, d)
        for a, b, c, d in itertools.product(
            PROFILE_LEVELS, INFORMER_LEVELS, ENDORSEMENT_LEVELS, INSTRUCTION_LEVELS
        )
    }
    print(f"full factorial grid: {len(all_labels)} designs (3x3x2x3)")

    tests = pd.read_csv(f"{PILOT_DIR}/design_tests.csv")
    tested = set(tests["design"])
    missing = sorted(all_labels - tested)
    print(f"designs with a Fisher's-exact comparison in design_tests.csv: {len(tested)}")
    print(f"designs with no comparison (base case, single 'none' arm): {missing}")

    sig = tests[tests["significant"]]
    print(f"significant at q<0.05: {len(sig)} / {len(tested)}")
    inverted = sig[sig["diff"] < 0]
    print(f"of those, inverted (non-adopter sample keener, diff<0): {len(inverted)}")
    print(sig[["design", "rate_adopter", "rate_non_adopter", "diff", "p", "q"]]
          .sort_values("q").to_string(index=False))

    # ---- 3: A1B1 households and wealth -------------------------------------
    section("3. A1B1 households (6026/6032 adopter pair vs 6039/6099 non-adopter pair)")
    feats = pd.read_csv(FEATURES_PATH).set_index("hhid")
    cols = ["rooms", "beds", "capita", "rooms_per_capita", "beds_per_capita",
            "electricity", "own_latrine", "has_shg", "has_savings", "_adopted"]
    print(feats.loc[[6026, 6032, 6039, 6099], cols].to_string())

    a1b1 = tests[tests["design"].str.startswith("A1B1")]
    print("\nA1B1* rows in design_tests.csv:")
    print(a1b1[["design", "rate_adopter", "rate_non_adopter", "diff", "significant"]].to_string(index=False))

    # ---- 4: A0B1C0D0 numbers vs main.tex -----------------------------------
    section("4. A0B1C0D0 (main.tex quotes 6032->0.75, 6099->0.20)")
    row = tests[tests["design"] == "A0B1C0D0"]
    print(row[["design", "rate_adopter", "rate_non_adopter", "diff", "significant"]].to_string(index=False))
    print("note: A0B0C0D0 has NO comparison row -- it is one of the 3 pure base cases "
          "(sample='none' only); the correct label for this claim is A0B1C0D0.")

    # ---- 5: isolated endorsement (C) marginal effect ------------------------
    section("5. Endorsement (C) marginal effect, isolated from household identity")
    files = glob.glob(f"{PILOT_DIR}/gpt_5_4_nano_*.csv")
    rows = []
    for f in files:
        label = re.search(r"gpt_5_4_nano_(A\dB\dC\dD\d)\.csv", f).group(1)
        df = pd.read_csv(f)
        df["design"] = label
        df["A"], df["B"], df["C"], df["D"] = (int(label[1]), int(label[3]),
                                               int(label[5]), int(label[7]))
        df["adopt"] = (df["decision"].astype(str).str.strip() == "(Y)").astype(int)
        rows.append(df[["design", "sample", "A", "B", "C", "D", "adopt"]])
    all_df = pd.concat(rows, ignore_index=True)

    pooled = all_df.groupby("C")["adopt"].agg(["mean", "count"])
    print("pooled adoption rate by C level (all rows, all designs/samples):")
    print(pooled.to_string())

    pivot = all_df.groupby(["A", "B", "D", "sample", "C"])["adopt"].mean().unstack("C").dropna()
    pivot["diff_C1_minus_C0"] = pivot[1] - pivot[0]
    print(f"\npaired same-household C1-vs-C0 contrasts (n={len(pivot)} pairs):")
    print(f"  mean diff   = {pivot['diff_C1_minus_C0'].mean():+.3f}")
    print(f"  median diff = {pivot['diff_C1_minus_C0'].median():+.3f}")
    print(f"  std diff    = {pivot['diff_C1_minus_C0'].std():.3f}")
    print(f"  range       = [{pivot['diff_C1_minus_C0'].min():+.2f}, {pivot['diff_C1_minus_C0'].max():+.2f}]")


if __name__ == "__main__":
    main()
