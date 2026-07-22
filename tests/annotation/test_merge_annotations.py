import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from annotation.merge_annotations import (
    score_gold_traps,
    resolve_labels,
    finalize,
)


def _plan(rows):
    cols = ["intern", "con_legis_num", "congress", "target", "title", "link",
            "is_gold_trap", "gold_label"]
    return pd.DataFrame(rows, columns=cols)


def test_score_gold_traps():
    plan = _plan([
        ["i1", "110_hr.1", 110, "gold", "t", "u", True, 1],
        ["i2", "110_hr.1", 110, "gold", "t", "u", True, 1],
        ["i1", "110_hr.2", 110, "gold", "t", "u", True, 0],
        ["i2", "110_hr.2", 110, "gold", "t", "u", True, 0],
    ])
    resp = pd.DataFrame([
        ["110_hr.1", "i1", "Yes", ""],   # correct (1)
        ["110_hr.1", "i2", "No", ""],    # wrong
        ["110_hr.2", "i1", "No", ""],    # correct (0)
        ["110_hr.2", "i2", "No", ""],    # correct (0)
    ], columns=["con_legis_num", "intern", "label", "notes"])
    rep = score_gold_traps(plan, resp).set_index("intern")
    assert rep.loc["i1", "n_correct"] == 2 and rep.loc["i1", "n_traps"] == 2
    assert rep.loc["i2", "n_correct"] == 1
    assert abs(rep.loc["i2", "accuracy"] - 0.5) < 1e-9


def test_resolve_labels_agree_disagree_unsure():
    plan = _plan([
        ["i1", "119_hr.1", 119, "test_119", "T1", "L1", False, ""],
        ["i2", "119_hr.1", 119, "test_119", "T1", "L1", False, ""],
        ["i1", "105_hr.9", 105, "train_neg", "T2", "L2", False, ""],
        ["i2", "105_hr.9", 105, "train_neg", "T2", "L2", False, ""],
        ["i1", "119_hr.7", 119, "test_119", "T3", "L3", False, ""],
        ["i2", "119_hr.7", 119, "test_119", "T3", "L3", False, ""],
    ])
    resp = pd.DataFrame([
        ["119_hr.1", "i1", "Yes", ""], ["119_hr.1", "i2", "Yes", ""],   # agree -> 1
        ["105_hr.9", "i1", "No", "x"], ["105_hr.9", "i2", "Yes", "y"],  # disagree
        ["119_hr.7", "i1", "Unsure", ""], ["119_hr.7", "i2", "No", ""], # unsure
    ], columns=["con_legis_num", "intern", "label", "notes"])

    auto, adj = resolve_labels(plan, resp)
    assert set(auto["con_legis_num"]) == {"119_hr.1"}
    assert int(auto.iloc[0]["manual_coding"]) == 1
    assert set(adj["con_legis_num"]) == {"105_hr.9", "119_hr.7"}
    # Adjudication rows carry both labels and notes for the disagreement.
    d = adj.set_index("con_legis_num").loc["105_hr.9"]
    assert {d["label_a"], d["label_b"]} == {"No", "Yes"}


def test_finalize_splits_by_target():
    plan = _plan([
        ["i1", "119_hr.1", 119, "test_119", "T1", "L1", False, ""],
        ["i1", "105_hr.9", 105, "train_neg", "T2", "L2", False, ""],
    ])
    auto = pd.DataFrame([
        ["119_hr.1", 119, "test_119", 1],
    ], columns=["con_legis_num", "congress", "target", "manual_coding"])
    adjudicated = pd.DataFrame([
        ["105_hr.9", 105, "train_neg", 0],
    ], columns=["con_legis_num", "congress", "target", "manual_coding"])

    out = finalize(auto, adjudicated, plan)
    assert set(out["test_119"]["split"]) == {"test"}
    assert set(out["train_neg"]["split"]) == {"train"}
    assert int(out["train_neg"].iloc[0]["manual_coding"]) == 0
