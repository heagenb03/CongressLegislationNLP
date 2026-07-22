import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from annotation.build_packets import (
    sample_119_test,
    sample_train_negatives,
    select_gold_traps,
    assign_interns,
)


def test_sample_119_excludes_amendments_and_is_deterministic():
    df = pd.DataFrame({
        "legislation_id": [f"119_hr.{i}" for i in range(10)] + ["119_samdt.1"],
        "congress": [119] * 11,
        "legislation_type": ["hr"] * 10 + ["samdt"],
        "legislation_number": [str(i) for i in range(10)] + ["1"],
        "matched_keywords": ["china"] * 11,
    })
    a = sample_119_test(df, 5, random.Random(1))
    b = sample_119_test(df, 5, random.Random(1))
    assert len(a) == 5
    assert list(a["legislation_id"]) == list(b["legislation_id"])  # deterministic
    assert "samdt" not in set(a["legislation_type"])               # no amendments


def test_sample_train_negatives_filters_weak_single_keyword_and_gold():
    df = pd.DataFrame({
        "legislation_id": ["105_hr.1", "105_hr.2", "105_hr.3", "105_hr.4"],
        "congress": [105, 105, 105, 105],
        "legislation_type": ["hr"] * 4,
        "legislation_number": ["1", "2", "3", "4"],
        # hr.1: single weak kw (keep). hr.2: strong kw (drop). hr.3: 2 kws (drop).
        # hr.4: single weak kw but in gold (drop).
        "matched_keywords": ["tariff", "prc", "china|tariff", "tariff"],
    })
    gold_keys = {"105_hr_4"}  # normalize_id_key form
    out = sample_train_negatives(df, gold_keys, 10, random.Random(1))
    assert list(out["legislation_id"]) == ["105_hr.1"]


def test_select_gold_traps_balanced_and_deterministic():
    gold = pd.DataFrame({
        "con_legis_num": [f"110_hr.{i}" for i in range(20)],
        "manual_coding": [1] * 10 + [0] * 10,
        "matched_keywords": ["prc"] * 10 + ["tariff"] * 10,
    })
    traps = select_gold_traps(gold, n_pos=3, n_neg=4, rng=random.Random(2))
    assert (traps["gold_label"] == 1).sum() == 3
    assert (traps["gold_label"] == 0).sum() == 4
    again = select_gold_traps(gold, n_pos=3, n_neg=4, rng=random.Random(2))
    assert list(traps["con_legis_num"]) == list(again["con_legis_num"])


def test_assign_interns_two_distinct_and_balanced():
    interns = [f"i{n}" for n in range(8)]
    bills = [f"b{n}" for n in range(80)]
    pairs = assign_interns(bills, interns)
    assert len(pairs) == 80
    for _bid, a, b in pairs:
        assert a != b
    # Each intern's total real-bill load within +/-1 of the mean (2*80/8 = 20)
    from collections import Counter
    load = Counter()
    for _bid, a, b in pairs:
        load[a] += 1
        load[b] += 1
    assert max(load.values()) - min(load.values()) <= 1
