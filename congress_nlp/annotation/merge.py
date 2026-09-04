"""Merge intern labels into resolved training/test data.

Two phases:
  1. build : load the intern labels, score gold traps, auto-accept agreements,
             and write an adjudication queue for disagreements + Unsure.
  2. finalize : after Heagen fills final_label in the adjudication queue,
                combine auto + adjudicated -> data/raw/intern_coded_*.csv.

Run from the project root:
    python scripts/annotation/merge_annotations.py build
    python scripts/annotation/merge_annotations.py finalize

Labels are read from the downloaded intern CSVs in data/raw/ by default.
Pass --from-sheet to pull them live from the Google Sheet instead (network I/O).
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import pandas as pd

from congress_nlp.annotation.utils import normalize_id_key
from congress_nlp.features.extract import con_legis_num_to_path, extract_from_json

ROOT = Path(__file__).resolve().parents[2]

LABEL_TO_INT: dict[str, object] = {"Yes": 1, "No": 0, "Unsure": None}

ANN = ROOT / "data" / "annotation"
RAW = ROOT / "data" / "raw" / "Summer2026InternsData"

INTERN_CSV_GLOB = "Congress Data Annotations Checked - *.csv"

# Accepted spellings when Heagen fills final_label by hand.
_FINAL_LABEL_MAP = {"0": 0, "1": 1, "no": 0, "yes": 1, "n": 0, "y": 1}

TARGET_SPLIT = {"test_119": "test", "train_neg": "train"}
TARGET_FILE = {
    "test_119": RAW / "intern_coded_119.csv",
    "train_neg": RAW / "intern_coded_negatives_101_116.csv",
}


def intern_name_from_filename(path: "str | Path") -> str:
    """Extract the intern name from a downloaded sheet-tab CSV filename.

    The name carries a trailing period ("Asher S."), which packet_plan.csv also
    stores, so only the ".csv" extension may be stripped.
    """
    name = Path(path).name
    if name.lower().endswith(".csv"):
        name = name[:-4]
    return name.split(" - ", 1)[1].strip() if " - " in name else name.strip()


def load_local_responses(raw_dir: Path) -> pd.DataFrame:
    """Load the downloaded intern CSVs into long form.

    Returns one row per (bill, intern) with columns
    con_legis_num / intern / label / notes.
    """
    paths = sorted(Path(raw_dir).glob(INTERN_CSV_GLOB))
    if not paths:
        raise FileNotFoundError(
            f"No intern CSVs matching {INTERN_CSV_GLOB!r} in {raw_dir}"
        )

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        missing = [c for c in ("con_legis_num", "Label") if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name} is missing column(s): {missing}")
        if "Notes" not in df.columns:
            df["Notes"] = ""
        df = df.rename(columns={"Label": "label", "Notes": "notes"})
        df["intern"] = intern_name_from_filename(path)
        frames.append(df[["con_legis_num", "intern", "label", "notes"]])

    resp = pd.concat(frames, ignore_index=True)
    resp["label"] = resp["label"].fillna("").astype(str).str.strip()
    return resp


def check_two_annotators(plan: pd.DataFrame, resp: pd.DataFrame) -> pd.Series:
    """Return real (non-trap) bills NOT covered by exactly 2 distinct interns.

    Counts distinct interns, not rows, so a duplicated row from one annotator
    is caught rather than passing as two-way coverage.
    """
    real = plan[~plan["is_gold_trap"]][["con_legis_num"]].drop_duplicates()
    covered = resp.merge(real, on="con_legis_num", how="inner")
    counts = covered.groupby("con_legis_num")["intern"].nunique()
    counts = counts.reindex(real["con_legis_num"], fill_value=0)
    return counts[counts != 2]


def _normalize_cell(value: object) -> str:
    """Stringify a hand-filled cell, collapsing 1.0 -> '1' and NaN -> ''."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (int, float)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def parse_adjudicated(adj: pd.DataFrame) -> pd.DataFrame:
    """Parse final_label into manual_coding, raising on blank/unreadable cells.

    The previous silent .isin([0, 1]) filter dropped unfilled rows without a
    word, which then shrank the training set invisibly downstream.
    """
    out = adj.copy()
    raw = out["final_label"].map(_normalize_cell)
    parsed = raw.str.lower().map(_FINAL_LABEL_MAP)

    bad = parsed.isna()
    if bad.any():
        ids = list(out.loc[bad, "con_legis_num"])
        blanks = int((raw[bad] == "").sum())
        vals = sorted({v for v in raw[bad] if v})
        raise ValueError(
            f"{int(bad.sum())} adjudication row(s) have no usable final_label "
            f"({blanks} blank; unrecognized values: {vals or 'none'}). "
            f"Fill 0/1 (or No/Yes) for: {ids}"
        )

    out["manual_coding"] = parsed.astype(int)
    return out


def train_neg_reversals(frames: dict[str, pd.DataFrame]) -> list[str]:
    """Presumed-negative bills that came back labeled 1 -> gold-set misses."""
    frame = frames.get("train_neg")
    if frame is None or frame.empty:
        return []
    return list(frame.loc[frame["manual_coding"] == 1, "con_legis_num"])


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

    expected = set(plan.loc[~plan["is_gold_trap"], "con_legis_num"])
    got = set(combined["con_legis_num"])
    if got != expected:
        missing, extra = sorted(expected - got), sorted(got - expected)
        raise ValueError(
            f"finalize expected {len(expected)} labeled bills, got {len(got)}. "
            f"Missing ({len(missing)}): {missing[:20]}"
            + (" ..." if len(missing) > 20 else "")
            + (f" | Unexpected ({len(extra)}): {extra[:20]}" if extra else "")
        )

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


def write_adjudication_queue(frame: pd.DataFrame, path: Path) -> None:
    """Write the adjudication queue with every field quoted.

    The queue is filled in by hand in a text editor, where the `link` column is
    followed by `,Yes,No,,,`. Editors treat "," as a legal URL path character
    and only trim *trailing* punctuation, so an unquoted link is detected as
    ".../senate-bill/732,Yes,No" and opens a 404. A double quote terminates
    link detection, so quoting the fields keeps the URL clickable. Neither the
    csv module nor pandas supports quoting a single column, hence QUOTE_ALL.
    """
    frame.to_csv(path, index=False, quoting=csv.QUOTE_ALL)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["build", "finalize"])
    parser.add_argument(
        "--from-sheet", action="store_true",
        help="Pull labels live from the Google Sheet instead of the local CSVs.",
    )
    args = parser.parse_args()

    plan = pd.read_csv(ANN / "packet_plan.csv")
    plan["is_gold_trap"] = plan["is_gold_trap"].astype(bool)

    if args.phase == "build":
        resp = pull_responses() if args.from_sheet else load_local_responses(RAW)
        resp.to_csv(ANN / "responses_raw.csv", index=False)  # audit trail

        bad_coverage = check_two_annotators(plan, resp)
        if not bad_coverage.empty:
            raise ValueError(
                f"{len(bad_coverage)} bill(s) not covered by exactly 2 distinct "
                f"interns: {bad_coverage.to_dict()}"
            )

        score_gold_traps(plan, resp).to_csv(ANN / "reliability_report.csv", index=False)
        auto_df, adj_df = resolve_labels(plan, resp)
        auto_df.to_csv(ANN / "resolved_auto.csv", index=False)
        write_adjudication_queue(adj_df, ANN / "adjudication_queue.csv")
        print(f"Auto-accepted {len(auto_df)}; {len(adj_df)} need adjudication. "
              f"Fill final_label in adjudication_queue.csv, then run finalize.")
        return

    # finalize
    auto_df = pd.read_csv(ANN / "resolved_auto.csv")
    adj = parse_adjudicated(pd.read_csv(ANN / "adjudication_queue.csv"))

    out = finalize(auto_df, adj, plan)
    for target, frame in out.items():
        path = TARGET_FILE[target]
        frame.to_csv(path, index=False)
        print(f"Wrote {len(frame)} rows -> {path}")

    reversals = train_neg_reversals(out)
    if reversals:
        print(
            f"\nNOTE: {len(reversals)} presumed-negative bill(s) from 101-116 were "
            f"labeled China-related. These are gold-set misses, not just training "
            f"rows: {reversals}"
        )


if __name__ == "__main__":
    main()
