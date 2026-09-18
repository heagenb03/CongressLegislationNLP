"""Zero-shot classification with facebook/bart-large-mnli.

A Scorer with a no-op fit: there is nothing to train, so the same instance is
valid on every fold. The model is loaded lazily on the first predict_proba, so
importing this module costs nothing and does not require torch.

Two hypothesis wordings are pinned because they target different boundaries.
The labeled negatives include omnibus bills carrying one China section: they
ARE about China in the topical sense and are still labeled 0, because the
standard is "primarily about China policy". Measure both on LOCO and pick from
the numbers. Never pick on the 119 split.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

MODEL_NAME = "facebook/bart-large-mnli"
CANDIDATES: tuple[str, str] = ("China policy", "other policy")
WORDINGS: dict[str, str] = {
    "topical": "This bill is about {}.",
    "primary": "This bill is primarily about {}.",
}
DEFAULT_WORDING = "topical"


@dataclass
class ZeroShotScorer:
    """Softmax over the two candidate labels; the score is P(China policy).

    The zero-shot pipeline truncates the sequence side by default, so a long
    summary cannot overflow BART's 1024-token window.
    """

    hypothesis: str = WORDINGS[DEFAULT_WORDING]
    model_name: str = MODEL_NAME
    batch_size: int = 8
    device: int = 0  # CUDA ordinal; pass -1 on a machine with no GPU
    _pipeline: object | None = field(default=None, init=False, repr=False)

    def fit(self, texts: Sequence[str], labels: np.ndarray) -> None:
        """No-op. Zero-shot means no training; the fold's labels are unused."""

    def _ensure_pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=self.device,
                batch_size=self.batch_size,
            )
        return self._pipeline

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        classify = self._ensure_pipeline()
        results = classify(
            list(texts),
            candidate_labels=list(CANDIDATES),
            hypothesis_template=self.hypothesis,
            multi_label=False,
        )
        if isinstance(results, dict):
            results = [results]
        return np.array(
            [r["scores"][r["labels"].index(CANDIDATES[0])] for r in results],
            dtype=float,
        )

    def close(self) -> None:
        self._pipeline = None
