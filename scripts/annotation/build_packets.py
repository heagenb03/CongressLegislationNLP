"""Build intern annotation packets: sample bills, pick gold traps, assign
two annotators per bill, and write data/annotation/packet_plan.csv.

The Google Sheet is created from packet_plan.csv by push_to_sheet() (Task 4).

Run from the project root:
    python scripts/annotation/build_packets.py            # writes packet_plan.csv
    python scripts/annotation/build_packets.py --push     # also creates the Google Sheet
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from annotation.annotation_utils import (  # noqa: E402
    bill_number_display,
    chamber_from_type,
    congress_gov_url,
    is_amendment_type,
    keyword_count_and_strong,
    normalize_id_key,
)
from modeling.extract_features import (  # noqa: E402
    STRONG_KEYWORDS,
    con_legis_num_to_path,
    extract_from_json,
)

# --- Configuration (edit before the sprint) ---
SEED = 20260721
INTERNS: list[str] = [f"intern{n}" for n in range(1, 9)]  # replace with real names
N_TEST_119 = 300
N_TRAIN_NEG = 140
N_GOLD_POS = 4
N_GOLD_NEG = 6

PLAN_COLUMNS = [
    "intern", "con_legis_num", "congress", "legislation_type", "chamber",
    "bill_number", "title", "link", "target", "is_gold_trap", "gold_label",
    "display_order",
]


def sample_119_test(df119: pd.DataFrame, n: int, rng: random.Random) -> pd.DataFrame:
    """Random sample of n non-amendment Congress-119 survivors."""
    pool = df119[~df119["legislation_type"].map(is_amendment_type)].copy()
    idx = list(pool.index)
    rng.shuffle(idx)
    return pool.loc[idx[:n]].reset_index(drop=True)


def sample_train_negatives(
    df_all: pd.DataFrame, gold_keys: set[str], n: int, rng: random.Random
) -> pd.DataFrame:
    """Sample n likely-negative survivors from Congresses 101-116.

    Keep rows with exactly one weak keyword (keyword_count == 1 and no strong
    keyword) that are NOT already in the gold set. Non-amendments only.
    """
    df = df_all[(df_all["congress"] >= 101) & (df_all["congress"] <= 116)].copy()
    df = df[~df["legislation_type"].map(is_amendment_type)]

    def _keep(mk: str) -> bool:
        count, strong = keyword_count_and_strong(str(mk), STRONG_KEYWORDS)
        return count == 1 and not strong

    df = df[df["matched_keywords"].fillna("").map(_keep)]
    df = df[~df["legislation_id"].map(normalize_id_key).isin(gold_keys)]

    idx = list(df.index)
    rng.shuffle(idx)
    return df.loc[idx[:n]].reset_index(drop=True)


def select_gold_traps(
    gold_df: pd.DataFrame, n_pos: int, n_neg: int, rng: random.Random
) -> pd.DataFrame:
    """Pick clear positives and negatives from the gold set as hidden traps.

    Prefers negatives with a single weak keyword (the subtle cases). Returns
    columns con_legis_num, gold_label.
    """
    pos = gold_df[gold_df["manual_coding"] == 1]
    neg = gold_df[gold_df["manual_coding"] == 0]

    def _pick(frame: pd.DataFrame, k: int) -> list[str]:
        ids = list(frame["con_legis_num"].astype(str))
        rng.shuffle(ids)
        return ids[:k]

    rows = [(cid, 1) for cid in _pick(pos, n_pos)] + [(cid, 0) for cid in _pick(neg, n_neg)]
    return pd.DataFrame(rows, columns=["con_legis_num", "gold_label"])


def assign_interns(bill_ids: list[str], interns: list[str]) -> list[tuple[str, str, str]]:
    """Assign each bill to two distinct, load-balanced interns.

    Uses neighbor rotation: bill k -> interns[k % m] and interns[(k+1) % m].
    Guarantees a != b and an even load of 2*len(bills)/m per intern.
    """
    m = len(interns)
    if m < 2:
        raise ValueError("Need at least 2 interns for double annotation")
    pairs: list[tuple[str, str, str]] = []
    for k, bid in enumerate(bill_ids):
        a = interns[k % m]
        b = interns[(k + 1) % m]
        pairs.append((bid, a, b))
    return pairs


def _display_fields(row: pd.Series, raw_root: Path) -> tuple[str, str]:
    """Return (title, link) for a survivor row, loading title from raw JSON."""
    ltype = str(row["legislation_type"])
    number = str(row["legislation_number"])
    congress = int(row["congress"])
    # Reconstruct a con_legis_num the path helper understands.
    cid = f"{congress}_{ltype}.{number}"
    json_path = con_legis_num_to_path(cid, raw_root)
    title = ""
    if json_path is not None and json_path.exists():
        official_title, _short, _summary, _subjects = extract_from_json(json_path)
        title = official_title
    link = congress_gov_url(congress, ltype, number)
    return title, link


def build_plan_rows(
    test_df: pd.DataFrame,
    neg_df: pd.DataFrame,
    traps: pd.DataFrame,
    interns: list[str],
    raw_root: Path,
    rng: random.Random,
) -> pd.DataFrame:
    """Assemble the full packet_plan (one row per intern-bill pairing)."""
    records: list[dict] = []

    # Real bills: test + negatives, each assigned to two interns.
    real = []
    for _i, row in test_df.iterrows():
        real.append((row, "test_119"))
    for _i, row in neg_df.iterrows():
        real.append((row, "train_neg"))

    bill_ids = [f"{int(r['congress'])}_{r['legislation_type']}.{r['legislation_number']}"
                for r, _t in real]
    pairs = assign_interns(bill_ids, interns)

    for (row, target), (cid, a, b) in zip(real, pairs):
        title, link = _display_fields(row, raw_root)
        ltype = str(row["legislation_type"])
        number = str(row["legislation_number"])
        congress = int(row["congress"])
        for intern in (a, b):
            records.append({
                "intern": intern,
                "con_legis_num": cid,
                "congress": congress,
                "legislation_type": ltype,
                "chamber": chamber_from_type(ltype),
                "bill_number": bill_number_display(ltype, number),
                "title": title,
                "link": link,
                "target": target,
                "is_gold_trap": False,
                "gold_label": "",
            })

    # Gold traps: every intern sees every trap.
    for _i, trow in traps.iterrows():
        cid = str(trow["con_legis_num"])
        parts = cid.split("_", 1)
        congress = int(parts[0])
        # Derive type/number from the dot-notation id for display.
        toks = parts[1].split(".")
        number = toks[-1]
        ltype = "".join(toks[:-1])
        json_path = con_legis_num_to_path(cid, raw_root)
        title = ""
        if json_path is not None and json_path.exists():
            title = extract_from_json(json_path)[0]
        link = congress_gov_url(congress, ltype, number)
        for intern in interns:
            records.append({
                "intern": intern,
                "con_legis_num": cid,
                "congress": congress,
                "legislation_type": ltype,
                "chamber": chamber_from_type(ltype),
                "bill_number": bill_number_display(ltype, number),
                "title": title,
                "link": link,
                "target": "gold",
                "is_gold_trap": True,
                "gold_label": int(trow["gold_label"]),
            })

    plan = pd.DataFrame.from_records(records)

    # Per-intern display shuffle so traps aren't clustered; keyed by intern.
    plan["display_order"] = 0
    out_frames = []
    for intern, grp in plan.groupby("intern", sort=True):
        g = grp.sample(frac=1.0, random_state=rng.randint(0, 2**31 - 1)).reset_index(drop=True)
        g["display_order"] = range(len(g))
        out_frames.append(g)
    return pd.concat(out_frames, ignore_index=True)[PLAN_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true", help="also create the Google Sheet")
    args = parser.parse_args()

    rng = random.Random(SEED)
    raw_root = ROOT / "raw_data" / "raw_legislation"

    df119 = pd.read_csv(ROOT / "data" / "processed" / "china_filter_results_119.csv")
    df_all = pd.read_csv(ROOT / "data" / "processed" / "china_filter_results.csv")
    gold = pd.read_csv(ROOT / "data" / "raw" / "twl_coded_legislation_101_to_118.csv",
                       on_bad_lines="skip")
    gold = gold[gold["manual_coding"].isin([0, 1])].copy()
    gold["manual_coding"] = gold["manual_coding"].astype(int)
    gold_keys = set(gold["con_legis_num"].astype(str).map(normalize_id_key))

    test_df = sample_119_test(df119, N_TEST_119, rng)
    neg_df = sample_train_negatives(df_all, gold_keys, N_TRAIN_NEG, rng)
    traps = select_gold_traps(gold, N_GOLD_POS, N_GOLD_NEG, rng)

    plan = build_plan_rows(test_df, neg_df, traps, INTERNS, raw_root, rng)

    out_dir = ROOT / "data" / "annotation"
    out_dir.mkdir(parents=True, exist_ok=True)
    plan.to_csv(out_dir / "packet_plan.csv", index=False)
    print(f"Wrote {len(plan)} plan rows for {plan['intern'].nunique()} interns "
          f"({len(test_df)} test, {len(neg_df)} neg, {len(traps)} traps).")

    if args.push:
        from annotation.build_packets_sheet import push_to_sheet
        push_to_sheet(plan)


if __name__ == "__main__":
    main()
