import numpy as np
import pytest

from congress_nlp.classifiers.zero_shot import CANDIDATES, WORDINGS, ZeroShotScorer


def test_both_wordings_are_pinned_and_format_a_candidate():
    assert set(WORDINGS) == {"topical", "primary"}
    assert WORDINGS["topical"].format("China policy") == "This bill is about China policy."
    assert (
        WORDINGS["primary"].format("China policy")
        == "This bill is primarily about China policy."
    )


def test_china_policy_is_the_first_candidate():
    assert CANDIDATES[0] == "China policy"
    assert len(CANDIDATES) == 2


def test_fit_is_a_no_op_and_does_not_load_a_model():
    scorer = ZeroShotScorer()
    scorer.fit(["anything"], np.array([1]))
    assert scorer._pipeline is None


def test_predict_proba_reads_the_china_score_from_each_result(monkeypatch):
    """The pipeline returns labels sorted by score, so index by label, not by position."""
    scorer = ZeroShotScorer()

    def fake_pipeline(texts, candidate_labels, hypothesis_template, multi_label):
        return [
            {"labels": ["China policy", "other policy"], "scores": [0.8, 0.2]},
            {"labels": ["other policy", "China policy"], "scores": [0.7, 0.3]},
        ]

    scorer._pipeline = fake_pipeline
    out = scorer.predict_proba(["a", "b"])
    assert out == pytest.approx(np.array([0.8, 0.3]))
