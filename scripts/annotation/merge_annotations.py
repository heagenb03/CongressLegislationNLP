"""Merge intern labels into resolved training/test data.

Two phases:
  1. build : pull the Sheet, score gold traps, auto-accept agreements, and
             write an adjudication queue for disagreements + Unsure.
  2. finalize : after Heagen fills final_label in the adjudication queue,
                combine auto + adjudicated -> data/raw/intern_coded_*.csv.

Run from the project root:
    python scripts/annotation/merge_annotations.py build
    python scripts/annotation/merge_annotations.py finalize

The Sheet pull (pull_responses) is network I/O run manually by Heagen.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from annotation.annotation_utils import normalize_id_key  # noqa: E402
from modeling.extract_features import con_legis_num_to_path, extract_from_json  # noqa: E402

LABEL_TO_INT: dict[str, object] = {"Yes": 1, "No": 0, "Unsure": None}

ANN = ROOT / "data" / "annotation"
RAW = ROOT / "data" / "raw"

TARGET_SPLIT = {"test_119": "test", "train_neg": "train"}
TARGET_FILE = {
    "test_119": RAW / "intern_coded_119.csv",
    "train_neg": RAW / "intern_coded_negatives_101_116.csv",
}


def score_gold_traps(plan: pd.DataFrame, resp: pd.DataFrame) -> pd.DataFrame:
    """Per-intern accuracy on the hidden gold-trap bills."""
    traps = plan[plan["is_gold_trap"]][["intern", "con_legis_num", "gold_label"]].copy()
    traps["gold_label"] = traps["gold_label"].astype(int)
    merged = traps.merge(resp, on=["intern", "con_legis_num"], how="left")
    merged["pred"] = merged["label"].map(LABEL_TO_INT)
    merged["correct"] = merged["pred"] == merged["gold_label"]
    rows = []
    for intern, grp in merged.groupby("intern"):
        n = len(grp)
        c = int(grp["correct"].sum())
        rows.append({"intern": intern, "n_traps": n, "n_correct": c,
                     "accuracy": (c / n) if n else 0.0})
    return pd.DataFrame(rows)


def resolve_labels(plan: pd.DataFrame, resp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split real (non-trap) bills into auto-accepted and adjudication-needed."""
    real = plan[~plan["is_gold_trap"]].drop_duplicates("con_legis_num").copy()
    resp_real = resp.merge(
        real[["con_legis_num"]], on="con_legis_num", how="inner"
    )

    auto_rows, adj_rows = [], []
    for cid, meta in real.set_index("con_legis_num").iterrows():
        r = resp_real[resp_real["con_legis_num"] == cid]
        labels = list(r["label"])
        notes = list(r["notes"].fillna("")) if "notes" in r else ["", ""]
        while len(labels) < 2:
            labels.append("")
            notes.append("")
        a, b = labels[0], labels[1]
        na, nb = notes[0], notes[1]
        ai, bi = LABEL_TO_INT.get(a), LABEL_TO_INT.get(b)

        if ai is not None and ai == bi:
            auto_rows.append({
                "con_legis_num": cid, "congress": int(meta["congress"]),
                "target": meta["target"], "manual_coding": int(ai),
            })
        else:
            adj_rows.append({
                "con_legis_num": cid, "congress": int(meta["congress"]),
                "target": meta["target"], "title": meta.get("title", ""),
                "link": meta.get("link", ""), "label_a": a, "label_b": b,
                "notes_a": na, "notes_b": nb, "final_label": "",
            })

    auto_df = pd.DataFrame(auto_rows, columns=[
        "con_legis_num", "congress", "target", "manual_coding"])
    adj_df = pd.DataFrame(adj_rows, columns=[
        "con_legis_num", "congress", "target", "title", "link",
        "label_a", "label_b", "notes_a", "notes_b", "final_label"])
    return auto_df, adj_df


def finalize(auto_df: pd.DataFrame, adjudicated_df: pd.DataFrame,
             plan: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Combine auto + adjudicated into per-target raw-file frames."""
    combined = pd.concat([
        auto_df[["con_legis_num", "congress", "target", "manual_coding"]],
        adjudicated_df[["con_legis_num", "congress", "target", "manual_coding"]],
    ], ignore_index=True)
    combined["manual_coding"] = combined["manual_coding"].astype(int)

    titles = plan.drop_duplicates("con_legis_num").set_index("con_legis_num")["title"]
    combined["title"] = combined["con_legis_num"].map(titles).fillna("")
    combined["split"] = combined["target"].map(TARGET_SPLIT)
    combined["source"] = "intern_2026"

    out: dict[str, pd.DataFrame] = {}
    for target, grp in combined.groupby("target"):
        out[target] = grp[[
            "con_legis_num", "congress", "title", "manual_coding", "split", "source"
        ]].reset_index(drop=True)
    return out


def pull_responses() -> pd.DataFrame:
    """[MANUAL I/O] Pull all intern tabs into long form (con_legis_num, intern, label, notes)."""
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_file(
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"], scopes=scopes)
    sh = gspread.authorize(creds).open_by_key(os.environ["ANNOTATION_SHEET_ID"])

    frames = []
    for ws in sh.worksheets():
        records = ws.get_all_records()  # header row -> dicts
        if not records:
            continue
        df = pd.DataFrame(records)
        df["intern"] = ws.title
        df = df.rename(columns={"Label": "label", "Notes": "notes"})
        frames.append(df[["con_legis_num", "intern", "label", "notes"]])
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["build", "finalize"])
    args = parser.parse_args()

    plan = pd.read_csv(ANN / "packet_plan.csv")
    plan["is_gold_trap"] = plan["is_gold_trap"].astype(bool)

    if args.phase == "build":
        resp = pull_responses()
        resp.to_csv(ANN / "responses_raw.csv", index=False)  # audit trail

        score_gold_traps(plan, resp).to_csv(ANN / "reliability_report.csv", index=False)
        auto_df, adj_df = resolve_labels(plan, resp)
        auto_df.to_csv(ANN / "resolved_auto.csv", index=False)
        adj_df.to_csv(ANN / "adjudication_queue.csv", index=False)
        print(f"Auto-accepted {len(auto_df)}; {len(adj_df)} need adjudication. "
              f"Fill final_label in adjudication_queue.csv, then run finalize.")
        return

    # finalize
    auto_df = pd.read_csv(ANN / "resolved_auto.csv")
    adj = pd.read_csv(ANN / "adjudication_queue.csv")
    adj = adj[adj["final_label"].isin([0, 1, "0", "1"])].copy()
    adj["manual_coding"] = adj["final_label"].astype(int)

    out = finalize(auto_df, adj, plan)
    for target, frame in out.items():
        path = TARGET_FILE[target]
        frame.to_csv(path, index=False)
        print(f"Wrote {len(frame)} rows -> {path}")


if __name__ == "__main__":
    main()
