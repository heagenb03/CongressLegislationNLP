"""CPU smoke tests on a tiny random model. No real encoder is downloaded."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from congress_nlp.classifiers.transformer import SEEDS, TransformerClassifier

TINY = "hf-internal-testing/tiny-random-roberta"


def _data():
    texts = [
        "china trade sanctions bill",
        "people's republic of china tariffs",
        "farm subsidy reauthorization",
        "highway bridge repair funding",
    ]
    return texts, np.array([1, 1, 0, 0])


def test_the_five_seeds_are_the_project_seeds():
    assert SEEDS == (20260721, 20260722, 20260723, 20260724, 20260725)


def test_pos_weight_is_negatives_over_positives():
    clf = TransformerClassifier(model_name=TINY, device="cpu", fp16=False)
    assert clf._pos_weight(np.array([1, 0, 0, 0])) == pytest.approx(3.0)
    assert clf._pos_weight(np.array([1, 1, 0, 0])) == pytest.approx(1.0)
    assert clf._pos_weight(np.array([1, 1, 1])) == pytest.approx(1.0)  # no negatives


def test_fit_then_predict_returns_probabilities_in_the_unit_interval():
    texts, labels = _data()
    clf = TransformerClassifier(
        model_name=TINY, device="cpu", fp16=False, epochs=1, batch_size=2
    )
    clf.fit(texts, labels)
    probs = clf.predict_proba(texts)
    assert probs.shape == (4,)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
    clf.close()


def test_the_same_seed_gives_the_same_predictions_on_cpu():
    texts, labels = _data()
    runs = []
    for _ in range(2):
        clf = TransformerClassifier(
            model_name=TINY, seed=20260721, device="cpu", fp16=False, epochs=1, batch_size=2
        )
        clf.fit(texts, labels)
        runs.append(clf.predict_proba(texts))
        clf.close()
    assert runs[0] == pytest.approx(runs[1])


def test_long_text_is_truncated_to_the_cap_not_rejected():
    texts, labels = _data()
    long_text = "china " * 4000
    clf = TransformerClassifier(
        model_name=TINY, device="cpu", fp16=False, epochs=1, batch_size=2, max_length=128
    )
    clf.fit(texts, labels)
    assert clf.predict_proba([long_text]).shape == (1,)
    clf.close()
