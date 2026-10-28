"""Frozen random draw of keyword-filter survivors, plus the blank labeling sheet.

The draw is the evaluation set: PER_ROUND * N_ROUNDS non-amendment survivors per
congress, drawn once with a fixed seed. It never conditions on summary presence.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from congress_nlp import paths
from congress_nlp.annotation.utils import congress_gov_url
from congress_nlp.features.load import load_features
from congress_nlp.ids import is_amendment, normalize_id_key, to_json_path
from congress_nlp.rawdata import read_bill_fields

logger = logging.getLogger(__name__)

DRAW_SEED = 20261027
PER_ROUND = 30
N_ROUNDS = 2
DRAW_LO = 101
DRAW_HI = 117

_DRAW_COLS = ["con_legis_num", "congress", "round", "already_labeled", "seed"]
_SHEET_COLS = [
    "con_legis_num", "congress", "round", "congress_gov_url", "official_title",
    "short_title", "subjects", "summary_text", "manual_coding", "notes",
]


def draw_sample(
    manifest: pd.DataFrame,
    labeled_keys: set[str],
    *,
    per_round: int = PER_ROUND,
    n_rounds: int = N_ROUNDS,
    seed: int = DRAW_SEED,
    lo: int = DRAW_LO,
    hi: int = DRAW_HI,
) -> pd.DataFrame:
    """Draw per_round * n_rounds bills per congress in lo..hi; no top-ups."""
    m = manifest.drop_duplicates("legislation_id")
    m = m[m["congress"].between(lo, hi)]
    m = m[~m["legislation_id"].map(is_amendment)]
    rng = np.random.default_rng(seed)
    total = per_round * n_rounds
    rows: list[dict] = []
    for congress in sorted(m["congress"].unique()):
        ids = sorted(m.loc[m["congress"] == congress, "legislation_id"])
        picked = rng.choice(len(ids), size=min(total, len(ids)), replace=False)
        for i, idx in enumerate(picked):
            cid = ids[int(idx)]
            rows.append(
                {
                    "con_legis_num": cid,
                    "congress": int(congress),
                    "round": i // per_round + 1,
                    "already_labeled": normalize_id_key(cid) in labeled_keys,
                    "seed": seed,
                }
            )
    return pd.DataFrame(rows, columns=_DRAW_COLS)


def labeling_sheet(
    draw: pd.DataFrame,
    manifest: pd.DataFrame,
    raw_root: Path,
    seed: int = DRAW_SEED,
) -> pd.DataFrame:
    """Blank sheet of unlabeled drawn bills: round 1 first, shuffled within round."""
    todo = draw[~draw["already_labeled"]]
    meta = manifest.drop_duplicates("legislation_id").set_index("legislation_id")
    rows: list[dict] = []
    for r in todo.itertuples():
        cid = r.con_legis_num
        title = short = summary = subjects = ""
        path = to_json_path(cid, raw_root)
        if path is not None and path.exists():
            title, short, summary, subjects = read_bill_fields(path)
        else:
            logger.warning("raw data.json missing for drawn bill %s", cid)
        url = ""
        if cid in meta.index:
            m = meta.loc[cid]
            url = congress_gov_url(
                int(m["congress"]), str(m["legislation_type"]), str(m["legislation_number"])
            )
        rows.append(
            {
                "con_legis_num": cid,
                "congress": r.congress,
                "round": r.round,
                "congress_gov_url": url,
                "official_title": title,
                "short_title": short,
                "subjects": subjects,
                "summary_text": summary,
                "manual_coding": pd.NA,
                "notes": pd.NA,
            }
        )
    sheet = pd.DataFrame(rows, columns=_SHEET_COLS)
    sheet = sheet.sample(frac=1, random_state=seed).reset_index(drop=True)
    return sheet.sort_values("round", kind="stable").reset_index(drop=True)


def main() -> None:
    for target in (paths.RANDOM_DRAW_CSV, paths.RANDOM_DRAW_LABELS):
        if target.exists():
            raise FileExistsError(
                f"{target} already exists; the draw is frozen and data/raw is never overwritten"
            )
    manifest = pd.concat(
        [pd.read_csv(p) for p in paths.manifest_paths()], ignore_index=True
    )
    feats = pd.concat(
        [load_features(splits=None), load_features(splits=["parked"])], ignore_index=True
    )
    labeled_keys = {normalize_id_key(i) for i in feats["con_legis_num"]}
    draw = draw_sample(manifest, labeled_keys)
    sheet = labeling_sheet(draw, manifest, paths.RAW_LEGISLATION)
    draw.to_csv(paths.RANDOM_DRAW_CSV, index=False)
    sheet.to_csv(paths.RANDOM_DRAW_LABELS, index=False)
    print("congress  drawn  already_labeled  to_label")
    for congress, g in draw.groupby("congress"):
        lab = int(g["already_labeled"].sum())
        print(f"{congress:>8}  {len(g):>5}  {lab:>15}  {len(g) - lab:>8}")
