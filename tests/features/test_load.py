import pandas as pd
import pytest

from congress_nlp import paths
from congress_nlp.features.load import load_features
from congress_nlp.splits import assign_split


@pytest.fixture
def features_csv(tmp_path, monkeypatch):
    congresses = [101, 116, 117, 118]
    df = pd.DataFrame(
        {
            "congress": congresses,
            "split": [assign_split(c) for c in congresses],
            "official_title": [f"A bill {c}" for c in congresses],
            "short_title": ["" for _ in congresses],
            "subjects": ["China|Trade" for _ in congresses],
        }
    )
    path = tmp_path / "features.csv"
    df.to_csv(path, index=False)
    monkeypatch.setattr(paths, "FEATURES_CSV", path)
    return path


def test_default_load_hides_parked_rows(features_csv):
    assert set(load_features().congress) == {101, 116, 117}


def test_parked_rows_load_only_when_named(features_csv):
    assert set(load_features(splits=["parked"]).congress) == {118}
