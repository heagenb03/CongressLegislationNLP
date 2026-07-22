"""Create the locked, dropdown-only Google Sheet from packet_plan.csv.

Requires env vars GOOGLE_APPLICATION_CREDENTIALS (service-account JSON) and
ANNOTATION_SHEET_ID (target Sheet). See docs/annotation/google_sheet_setup.md.

This module performs network I/O and is run manually by Heagen.
"""
from __future__ import annotations

import os

import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# Visible header shown to interns. con_legis_num is written to a trailing
# hidden/locked column so migration stays keyed by ID.
HEADER = ["Congress", "Chamber", "Bill #", "Title", "Link", "Label", "Notes", "con_legis_num"]
LABEL_CHOICES = ["Yes", "No", "Unsure"]


def _client() -> gspread.Client:
    path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    creds = Credentials.from_service_account_file(path, scopes=SCOPES)
    return gspread.authorize(creds)


def _tab_rows(plan: pd.DataFrame, intern: str) -> list[list]:
    grp = plan[plan["intern"] == intern].sort_values("display_order")
    rows = [HEADER]
    for _i, r in grp.iterrows():
        rows.append([
            r["congress"], r["chamber"], r["bill_number"], r["title"],
            r["link"], "", "", r["con_legis_num"],
        ])
    return rows


def push_to_sheet(plan: pd.DataFrame) -> None:
    client = _client()
    sh = client.open_by_key(os.environ["ANNOTATION_SHEET_ID"])

    for intern in sorted(plan["intern"].unique()):
        rows = _tab_rows(plan, intern)
        n = len(rows)
        try:
            ws = sh.worksheet(intern)
            sh.del_worksheet(ws)
        except gspread.WorksheetNotFound:
            pass
        ws = sh.add_worksheet(title=intern, rows=n + 5, cols=len(HEADER))
        ws.update("A1", rows, value_input_option="RAW")

        # Data-validation dropdown on the Label column (rows 2..n).
        ws.add_validation(
            f"F2:F{n}",
            gspread.worksheet.ValidationConditionType.one_of_list,
            LABEL_CHOICES,
            strict=True,
            showCustomUi=True,
        )
        # Hide the con_legis_num column (H).
        ws.hide_columns(7, 8)
        # Selectively protect the header, bill fields (A-E), and hidden id (H);
        # Label (F) and Notes (G) are left unprotected and editable.
        ws.add_protected_range("A1:H1", description="header")
        ws.add_protected_range(f"A2:E{n}", description="bill fields")
        ws.add_protected_range(f"H2:H{n}", description="id")
        # Freeze header row.
        ws.freeze(rows=1)

    print(f"Pushed {plan['intern'].nunique()} intern tabs to the Sheet.")
