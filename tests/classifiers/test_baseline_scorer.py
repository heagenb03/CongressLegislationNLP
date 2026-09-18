import numpy as np

from congress_nlp.classifiers.baseline import MAX_FEATURES, MAX_ITER, TfidfLogRegScorer


def _texts():
    positive = ["china trade sanctions", "people's republic of china", "china tariffs"]
    negative = ["farm subsidies bill", "highway funding act", "school lunch program"]
    return positive + negative, np.array([1, 1, 1, 0, 0, 0])


def test_scorer_ranks_a_china_text_above_an_unrelated_one():
    texts, labels = _texts()
    scorer = TfidfLogRegScorer()
    scorer.fit(texts, labels)
    probs = scorer.predict_proba(["china export controls", "national park funding"])
    assert probs.shape == (2,)
    assert probs[0] > probs[1]


def test_probabilities_are_in_the_unit_interval():
    texts, labels = _texts()
    scorer = TfidfLogRegScorer()
    scorer.fit(texts, labels)
    probs = scorer.predict_proba(texts)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


def test_refitting_forgets_the_previous_vocabulary():
    """A scorer reused across folds must not carry the old fold's vocabulary."""
    texts, labels = _texts()
    scorer = TfidfLogRegScorer()
    scorer.fit(texts, labels)
    scorer.fit(["tibet human rights", "bridge repair"], np.array([1, 0]))
    assert set(scorer.vectorizer.vocabulary_) == {"tibet", "human", "rights", "bridge", "repair"}


def test_defaults_match_the_frozen_baseline_hyperparameters():
    scorer = TfidfLogRegScorer()
    assert scorer.max_features == MAX_FEATURES == 10000
    assert scorer.max_iter == MAX_ITER == 1000
