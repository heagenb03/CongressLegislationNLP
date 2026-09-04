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
import re
from pathlib import Path

import pandas as pd

from congress_nlp import paths
from congress_nlp.annotation.utils import (
    bill_number_display,
    chamber_from_type,
    congress_gov_url,
    is_amendment_type,
    keyword_count_and_strong,
)
from congress_nlp.filtering.keywords import STRONG_KEYWORDS
from congress_nlp.ids import normalize_id_key, to_json_path
from congress_nlp.rawdata import read_bill_fields


# --- Configuration (edit before the sprint) ---
SEED = 20260721
INTERNS: list[str] = ["Asher S.", "Elizabeth O.", "Ines A.", "Kate K.", "Lena H.", "Nikita B.", "Noah M.", "Serena K."]
N_TEST_119 = 499
N_TRAIN_NEG = 140
N_GOLD_POS = 4
N_GOLD_NEG = 6

# Every annotated bill must have a CRS summary, so future batches can train
# summary-bearing input configurations. NOTE: this is expensive on a current
# Congress — only 33% of the 119th had a summary when it was annotated, and the
# with-summary subgroup skews less positive (35% vs 59%). Set False for a
# current-Congress sprint where losing two thirds of the pool is unacceptable.
REQUIRE_SUMMARY = True

PLAN_COLUMNS = [
    "intern", "con_legis_num", "congress", "legislation_type", "chamber",
    "bill_number", "title", "link", "target", "is_gold_trap", "gold_label",
    "display_order",
]


def has_summary_on_disk(con_legis_num: str, raw_root: Path) -> bool:
    """True if this bill's data.json carries non-empty CRS summary text."""
    json_path = to_json_path(str(con_legis_num), raw_root)
    if json_path is None or not json_path.exists():
        return False
    _title, _short, summary, _subjects = read_bill_fields(json_path)
    return bool(summary and summary.strip())


def _filter_to_summarized(
    df: pd.DataFrame, raw_root: Path | None, require_summary: bool, label: str
) -> pd.DataFrame:
    """Drop no-summary candidates, reporting how much of the pool was lost."""
    if not require_summary or raw_root is None:
        return df
    keep = df["legislation_id"].map(lambda i: has_summary_on_disk(i, raw_root))
    out = df[keep]
    print(f"  [{label}] summary gate: {len(df)} candidates -> {len(out)} "
          f"({len(df) - len(out)} dropped for no CRS summary)")
    return out


def sample_119_test(
    df119: pd.DataFrame,
    n: int,
    rng: random.Random,
    raw_root: Path | None = None,
    require_summary: bool = False,
) -> pd.DataFrame:
    """Random sample of n non-amendment Congress-119 survivors."""
    pool = df119[~df119["legislation_type"].map(is_amendment_type)].copy()
    pool = _filter_to_summarized(pool, raw_root, require_summary, "119 test")
    idx = list(pool.index)
    rng.shuffle(idx)
    return pool.loc[idx[:n]].reset_index(drop=True)


def sample_train_negatives(
    df_all: pd.DataFrame,
    gold_keys: set[str],
    n: int,
    rng: random.Random,
    raw_root: Path | None = None,
    require_summary: bool = False,
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
    df = _filter_to_summarized(df, raw_root, require_summary, "train neg")

    idx = list(df.index)
    rng.shuffle(idx)
    return df.loc[idx[:n]].reset_index(drop=True)


def select_gold_traps(
    gold_df: pd.DataFrame, n_pos: int, n_neg: int, rng: random.Random
) -> pd.DataFrame:
    """Pick clear positives and negatives from the gold set as hidden traps.

    Performs a deterministic (seeded) random pick of n_pos positives and
    n_neg negatives from the gold set -- no keyword-based filtering. Returns
    columns con_legis_num, gold_label.

    Amendments (s.amdt.* / h.amdt.*) are excluded: the gold CSV contains ~94 of
    them, but interns never label amendments (see is_amendment_type), and the
    display/link machinery only handles bills and resolutions.
    """
    gold_df = gold_df[~gold_df["con_legis_num"].astype(str).str.contains(r"\.amdt\.", case=False, regex=True)]
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
    Guarantees a != b and a load of 2*len(bills)/m per intern that is even
    when len(bills) is large relative to m (the real sprint scale); for very
    small bill counts the spread across interns can exceed 1.
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


def _bare_number(legislation_number: str) -> str:
    """Strip a leading type prefix from a raw directory name like 'hr21' or
    'sconres13', returning just the trailing digits ('21', '13')."""
    match = re.search(r"(\d+)$", str(legislation_number))
    if match is None:
        return str(legislation_number)
    return match.group(1)


def _display_fields(row: pd.Series, raw_root: Path) -> tuple[str, str]:
    """Return (title, link) for a survivor row, loading title from raw JSON."""
    ltype = str(row["legislation_type"])
    congress = int(row["congress"])
    number = _bare_number(row["legislation_number"])
    # legislation_id is already the canonical dotted con_legis_num.
    cid = str(row["legislation_id"])
    json_path = to_json_path(cid, raw_root)
    title = ""
    if json_path is not None and json_path.exists():
        official_title, _short, _summary, _subjects = read_bill_fields(json_path)
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

    bill_ids = [str(r["legislation_id"]) for r, _t in real]
    pairs = assign_interns(bill_ids, interns)

    for (row, target), (cid, a, b) in zip(real, pairs):
        title, link = _display_fields(row, raw_root)
        ltype = str(row["legislation_type"])
        number = _bare_number(row["legislation_number"])
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
        json_path = to_json_path(cid, raw_root)
        title = ""
        if json_path is not None and json_path.exists():
            title = read_bill_fields(json_path)[0]
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
    raw_root = paths.RAW_LEGISLATION

    df119 = pd.read_csv(paths.MANIFEST_DIR / "china_filter_119.csv")
    df_all = pd.read_csv(paths.MANIFEST_DIR / "china_filter_101_118.csv")
    gold = pd.read_csv(paths.GOLD_LABELS,
                       on_bad_lines="skip")
    gold = gold[gold["manual_coding"].isin([0, 1])].copy()
    gold["manual_coding"] = gold["manual_coding"].astype(int)
    gold_keys = set(gold["con_legis_num"].astype(str).map(normalize_id_key))

    test_df = sample_119_test(df119, N_TEST_119, rng,
                              raw_root=raw_root, require_summary=REQUIRE_SUMMARY)
    neg_df = sample_train_negatives(df_all, gold_keys, N_TRAIN_NEG, rng,
                                    raw_root=raw_root, require_summary=REQUIRE_SUMMARY)
    traps = select_gold_traps(gold, N_GOLD_POS, N_GOLD_NEG, rng)

    plan = build_plan_rows(test_df, neg_df, traps, INTERNS, raw_root, rng)

    out_dir = paths.ANNOTATION_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    plan.to_csv(out_dir / "packet_plan.csv", index=False)
    print(f"Wrote {len(plan)} plan rows for {plan['intern'].nunique()} interns "
          f"({len(test_df)} test, {len(neg_df)} neg, {len(traps)} traps).")

    if args.push:
        from congress_nlp.annotation.sheets import push_to_sheet
        push_to_sheet(plan)


if __name__ == "__main__":
    main()
