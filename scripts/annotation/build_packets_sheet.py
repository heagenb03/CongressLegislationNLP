"""Create the locked, dropdown-only Google Sheet from packet_plan.csv.

Auth (two supported paths, see docs/annotation/google_sheet_setup.md):
  * Default: Application Default Credentials from a gcloud user login
    (`gcloud auth application-default login --scopes=...spreadsheets,...drive`).
    The Sheet is owned by your own Google account and no key file is needed.
  * Fallback: set GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON key.

ANNOTATION_SHEET_ID selects the target Sheet. If it is unset, a new Sheet is
created (in the authenticated user's Drive) and its ID/URL are printed.

This module performs network I/O and is run manually by Heagen.
"""
from __future__ import annotations

import os

import google.auth
import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

NEW_SHEET_TITLE = "China Legislation Annotation Sprint"

# Visible header shown to interns. con_legis_num is written to a trailing
# hidden/locked column so migration stays keyed by ID.
HEADER = ["Congress", "Chamber", "Bill #", "Title", "Link", "Label", "Notes", "con_legis_num"]
LABEL_CHOICES = ["Yes", "No", "Unsure"]


def _client() -> gspread.Client:
    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if path:
        # Fallback: explicit service-account key.
        creds = Credentials.from_service_account_file(path, scopes=SCOPES)
    else:
        # Default: gcloud Application Default Credentials (user login).
        # The token must already carry the Sheets + Drive scopes, granted at
        # `gcloud auth application-default login --scopes=...`.
        creds, _ = google.auth.default(scopes=SCOPES)
    return gspread.authorize(creds)


def _open_or_create(client: gspread.Client) -> gspread.Spreadsheet:
    sheet_id = os.environ.get("ANNOTATION_SHEET_ID")
    if sheet_id:
        return client.open_by_key(sheet_id)
    sh = client.create(NEW_SHEET_TITLE)
    print(f"Created new Sheet '{NEW_SHEET_TITLE}'")
    print(f"  ANNOTATION_SHEET_ID = {sh.id}")
    print(f"  URL = https://docs.google.com/spreadsheets/d/{sh.id}/edit")
    return sh


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
    sh = _open_or_create(client)
    existing_before = {ws.title for ws in sh.worksheets()}

    for intern in sorted(plan["intern"].unique()):
        rows = _tab_rows(plan, intern)
        n = len(rows)
        try:
            ws = sh.worksheet(intern)
            sh.del_worksheet(ws)
        except gspread.WorksheetNotFound:
            pass
        ws = sh.add_worksheet(title=intern, rows=n + 5, cols=len(HEADER))
        # gspread 6.x Worksheet.update is values-first: update(values, range_name).
        ws.update(rows, "A1", value_input_option="RAW")

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

    # Drop any pre-existing default tabs (e.g. "Sheet1" on a just-created Sheet)
    # so interns only see their own tabs. Never delete an intern tab.
    intern_tabs = set(plan["intern"].unique())
    if len(sh.worksheets()) > len(intern_tabs):
        for title in existing_before:
            if title not in intern_tabs:
                try:
                    sh.del_worksheet(sh.worksheet(title))
                except gspread.WorksheetNotFound:
                    pass

    print(f"Pushed {plan['intern'].nunique()} intern tabs to the Sheet.")
    print(f"  URL = https://docs.google.com/spreadsheets/d/{sh.id}/edit")
