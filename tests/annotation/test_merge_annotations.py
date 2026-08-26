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


# --- local CSV loading -------------------------------------------------------

def test_intern_name_from_filename_keeps_trailing_period():
    from annotation.merge_annotations import intern_name_from_filename
    # packet_plan.csv stores names as "Asher S." — the trailing period is part
    # of the name, not the extension, so the join fails if it is stripped.
    assert intern_name_from_filename(
        "Congress Data Annotations Checked - Asher S..csv") == "Asher S."
    assert intern_name_from_filename(
        Path("x/y/Congress Data Annotations Checked - Noah M..csv")) == "Noah M."


def test_load_local_responses_reads_and_renames(tmp_path):
    from annotation.merge_annotations import load_local_responses
    (tmp_path / "Congress Data Annotations Checked - Kate K..csv").write_text(
        "Congress,Title,Label,Notes,con_legis_num\n"
        "119,T1,Yes,note-a,119_hr.1\n"
        "115,T2,No,,115_s.5\n", encoding="utf-8")
    (tmp_path / "Congress Data Annotations Checked - Lena H..csv").write_text(
        "Congress,Title,Label,Notes,con_legis_num\n"
        "119,T1,No,,119_hr.1\n", encoding="utf-8")

    resp = load_local_responses(tmp_path)
    assert list(resp.columns) == ["con_legis_num", "intern", "label", "notes"]
    assert len(resp) == 3
    assert set(resp["intern"]) == {"Kate K.", "Lena H."}
    assert resp.set_index(["con_legis_num", "intern"]).loc[
        ("119_hr.1", "Kate K."), "label"] == "Yes"


def test_load_local_responses_errors_on_missing_column(tmp_path):
    import pytest
    from annotation.merge_annotations import load_local_responses
    (tmp_path / "Congress Data Annotations Checked - Kate K..csv").write_text(
        "Congress,Title,Notes\n119,T1,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Label"):
        load_local_responses(tmp_path)


def test_load_local_responses_errors_when_no_files(tmp_path):
    import pytest
    from annotation.merge_annotations import load_local_responses
    with pytest.raises(FileNotFoundError):
        load_local_responses(tmp_path)


# --- annotator-coverage check ------------------------------------------------

def test_check_two_annotators_flags_bad_coverage():
    from annotation.merge_annotations import check_two_annotators
    plan = _plan([
        ["i1", "119_hr.1", 119, "test_119", "T", "L", False, ""],
        ["i2", "119_hr.1", 119, "test_119", "T", "L", False, ""],
        ["i1", "119_hr.2", 119, "test_119", "T", "L", False, ""],
        ["i1", "119_hr.3", 119, "test_119", "T", "L", False, ""],
        ["i2", "119_hr.3", 119, "test_119", "T", "L", False, ""],
        ["i1", "110_hr.9", 110, "gold", "T", "L", True, 1],
    ])
    resp = pd.DataFrame([
        ["119_hr.1", "i1", "Yes", ""], ["119_hr.1", "i2", "No", ""],   # ok: 2 interns
        ["119_hr.2", "i1", "Yes", ""],                                  # only 1 intern
        ["119_hr.3", "i1", "Yes", ""], ["119_hr.3", "i1", "Yes", ""],   # dup row, 1 intern
        ["110_hr.9", "i1", "Yes", ""],                                  # trap: ignored
    ], columns=["con_legis_num", "intern", "label", "notes"])
    bad = check_two_annotators(plan, resp)
    assert set(bad.index) == {"119_hr.2", "119_hr.3"}


# --- adjudication parsing ----------------------------------------------------

def test_parse_adjudicated_accepts_int_float_and_words():
    from annotation.merge_annotations import parse_adjudicated
    adj = pd.DataFrame({
        "con_legis_num": ["a", "b", "c", "d"],
        "final_label": [1, 0.0, "Yes", "no"],
    })
    out = parse_adjudicated(adj)
    assert list(out["manual_coding"]) == [1, 0, 1, 0]


def test_parse_adjudicated_raises_naming_blank_and_bad_values():
    import pytest
    from annotation.merge_annotations import parse_adjudicated
    adj = pd.DataFrame({
        "con_legis_num": ["a", "b", "c"],
        "final_label": [1, None, "maybe"],
    })
    with pytest.raises(ValueError) as exc:
        parse_adjudicated(adj)
    msg = str(exc.value)
    assert "b" in msg and "c" in msg and "maybe" in msg


# --- finalize row-count guard ------------------------------------------------

def test_finalize_raises_when_rows_missing():
    import pytest
    from annotation.merge_annotations import finalize
    plan = _plan([
        ["i1", "119_hr.1", 119, "test_119", "T1", "L1", False, ""],
        ["i1", "105_hr.9", 105, "train_neg", "T2", "L2", False, ""],
    ])
    auto = pd.DataFrame([["119_hr.1", 119, "test_119", 1]],
                        columns=["con_legis_num", "congress", "target", "manual_coding"])
    empty = pd.DataFrame(columns=["con_legis_num", "congress", "target", "manual_coding"])
    with pytest.raises(ValueError, match="105_hr.9"):
        finalize(auto, empty, plan)


def test_train_neg_reversals_surfaces_presumed_negatives():
    from annotation.merge_annotations import train_neg_reversals
    frames = {
        "train_neg": pd.DataFrame({"con_legis_num": ["105_hr.9", "106_s.1"],
                                   "manual_coding": [1, 0]}),
        "test_119": pd.DataFrame({"con_legis_num": ["119_hr.1"], "manual_coding": [1]}),
    }
    assert train_neg_reversals(frames) == ["105_hr.9"]
