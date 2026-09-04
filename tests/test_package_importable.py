def test_congress_nlp_is_importable():
    import congress_nlp

    assert congress_nlp.__name__ == "congress_nlp"


def test_every_subpackage_is_importable():
    import importlib

    for name in (
        "congress_nlp.filtering",
        "congress_nlp.annotation",
        "congress_nlp.features",
        "congress_nlp.classifiers",
        "congress_nlp.evaluation",
    ):
        assert importlib.import_module(name).__name__ == name
