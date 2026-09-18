"""Fine-tuned encoder as a Scorer, with a hand-written training loop.

Why not transformers.Trainer: class imbalance is handled by weighted loss per
the project's design decision, and Trainer with num_labels=1 applies MSELoss and
treats the task as regression. One logit plus
BCEWithLogitsLoss(pos_weight=n_neg/n_pos) is the mechanism, so the loop is here.

Input is capped at 512 tokens with dynamic padding, NOT 256. The long bills are
the omnibus negatives -- a defense authorization carrying one China section
among hundreds. Truncating harder deletes exactly the signal that marks a bill
as omnibus, which is the dominant negative pattern this classifier has to
reject. Under OOM the fallback is batch 8 with grad_accum 2, never a shorter
cap.

fp16 makes same-seed run-to-run variation real. The 5-seed spread absorbs it;
report mean +/- sd over SEEDS, never a single run and never the best seed.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

SEEDS: tuple[int, ...] = (20260721, 20260722, 20260723, 20260724, 20260725)
DEFAULT_MODEL = "roberta-base"
MAX_LENGTH = 512


@dataclass
class TransformerClassifier:
    model_name: str = DEFAULT_MODEL
    seed: int = SEEDS[0]
    epochs: int = 4
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    batch_size: int = 16
    grad_accum: int = 1
    max_length: int = MAX_LENGTH
    eval_batch_size: int = 32
    device: str | None = None
    fp16: bool = True
    _tokenizer: object | None = field(default=None, init=False, repr=False)
    _model: object | None = field(default=None, init=False, repr=False)

    def _resolve_device(self) -> str:
        import torch

        if self.device is not None:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _pos_weight(labels: np.ndarray) -> float:
        """n_negatives / n_positives, the weight on the positive class.

        Falls back to 1.0 when either class is absent, which cannot happen in a
        real fold but does happen in tests.
        """
        labels = np.asarray(labels)
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        if n_pos == 0 or n_neg == 0:
            return 1.0
        return n_neg / n_pos

    def _seed_everything(self) -> None:
        import torch

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def _encode(self, texts: Sequence[str]) -> list[dict]:
        encoded = self._tokenizer(
            list(texts), truncation=True, max_length=self.max_length
        )
        return [
            {"input_ids": ids, "attention_mask": mask}
            for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"])
        ]

    def _collate(self, batch: list[dict]):
        """Pad to the longest sequence in this batch, not to max_length."""
        import torch

        features = [
            {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}
            for b in batch
        ]
        out = self._tokenizer.pad(features, padding=True, return_tensors="pt")
        if "label" in batch[0]:
            out["labels"] = torch.tensor(
                [b["label"] for b in batch], dtype=torch.float32
            )
        return out

    def fit(self, texts: Sequence[str], labels: np.ndarray) -> None:
        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )

        self._seed_everything()
        device = self._resolve_device()
        use_fp16 = self.fp16 and device == "cuda"

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # ignore_mismatched_sizes: a checkpoint that already carries a classifier
        # head of another size (the tiny test fixture ships a 2-label head) gets a
        # fresh 1-logit head instead of an error. A no-op for roberta-base and
        # distilbert-base-uncased, which ship no classifier weights.
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=1, ignore_mismatched_sizes=True
        ).to(device)

        labels = np.asarray(labels, dtype=float)
        dataset = [
            {**item, "label": float(label)}
            for item, label in zip(self._encode(texts), labels)
        ]
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate,
            generator=torch.Generator().manual_seed(self.seed),
        )

        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self._pos_weight(labels)], device=device)
        )
        optimizer = AdamW(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        steps_per_epoch = max(1, len(loader) // self.grad_accum)
        total_steps = steps_per_epoch * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.warmup_ratio * total_steps),
            num_training_steps=total_steps,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

        self._model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            pending = False
            for step, batch in enumerate(loader, start=1):
                batch = {k: v.to(device) for k, v in batch.items()}
                targets = batch.pop("labels")
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                    logits = self._model(**batch).logits.squeeze(-1)
                    loss = loss_fn(logits, targets) / self.grad_accum
                scaler.scale(loss).backward()
                pending = True
                if step % self.grad_accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    pending = False
            if pending:
                # The epoch's last batches did not fill an accumulation group.
                # Stepping on the tail keeps every training row in the update;
                # dropping it would silently shrink an already small fold.
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        import torch
        from torch.utils.data import DataLoader

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("fit() must be called before predict_proba().")

        device = self._resolve_device()
        use_fp16 = self.fp16 and device == "cuda"
        loader = DataLoader(
            self._encode(texts),
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=self._collate,
        )

        self._model.eval()
        probs: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                    logits = self._model(**batch).logits.squeeze(-1)
                probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        return np.concatenate(probs).astype(float)

    def close(self) -> None:
        """Free the model so the next fold starts with clean VRAM."""
        import gc

        import torch

        self._model = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
