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
    build_plan_rows,
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


def test_build_plan_rows_uses_legislation_id_and_bare_number_for_display():
    """Regression test for the legislation_number-prefix bug (Task 3 review).

    Real Stage-1 manifests store legislation_id already dotted
    (e.g. '119_h.r.21') and legislation_number as the raw prefixed directory
    name (e.g. 'hr21'), NOT a bare number. build_plan_rows must use
    legislation_id verbatim as con_legis_num, and must strip the prefix from
    legislation_number before using it for bill_number/link display.
    """
    test_df = pd.DataFrame({
        "legislation_id": ["119_h.r.21"],
        "congress": [119],
        "legislation_type": ["hr"],
        "legislation_number": ["hr21"],
        "matched_keywords": ["china"],
    })
    neg_df = pd.DataFrame({
        "legislation_id": ["119_s.con.res.13"],
        "congress": [119],
        "legislation_type": ["sconres"],
        "legislation_number": ["sconres13"],
        "matched_keywords": ["tariff"],
    })
    traps = pd.DataFrame(columns=["con_legis_num", "gold_label"])
    interns = [f"i{n}" for n in range(1, 5)]
    raw_root = Path("nonexistent_raw_root_for_test")

    plan = build_plan_rows(
        test_df, neg_df, traps, interns, raw_root, random.Random(3)
    )

    real = plan[~plan["is_gold_trap"]]
    assert set(real["target"]) == {"test_119", "train_neg"}

    hr_rows = real[real["con_legis_num"] == "119_h.r.21"]
    assert len(hr_rows) == 2
    assert hr_rows["bill_number"].eq("H.R. 21").all()
    assert hr_rows["link"].str.endswith("/21").all()
    assert hr_rows["intern"].nunique() == 2

    sconres_rows = real[real["con_legis_num"] == "119_s.con.res.13"]
    assert len(sconres_rows) == 2
    assert sconres_rows["bill_number"].eq("S.Con.Res. 13").all()
    assert sconres_rows["link"].str.endswith("/13").all()
    assert sconres_rows["intern"].nunique() == 2

    # No malformed 'hr.hr21'-style ids ever leaked into the plan.
    assert not real["con_legis_num"].str.contains(r"\.hr21|\.sconres13").any()
