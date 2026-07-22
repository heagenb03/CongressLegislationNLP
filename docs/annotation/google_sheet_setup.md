# Google Sheet setup (one-time)

1. Go to https://console.cloud.google.com/ → create a project (or reuse one).
2. Enable **Google Sheets API** and **Google Drive API** for the project.
3. Create a **Service Account** → add a **JSON key** → download it.
4. Save the JSON somewhere private (NOT in the repo). Set an env var:
   - PowerShell: `$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\path\to\key.json"`
5. Create an empty Google Sheet in your Drive. Share it (Editor) with the
   service-account email (found inside the JSON as `client_email`).
6. Copy the Sheet's ID from its URL and set:
   - PowerShell: `$env:ANNOTATION_SHEET_ID = "<sheet id>"`
7. Share the Sheet (Editor) with each intern's Google account.

The key file is never committed — keep it outside the repo, on disk somewhere
private, and reference it only via `GOOGLE_APPLICATION_CREDENTIALS`.

## Before you run: gspread 6.1.2 API checklist

`requirements.txt` pins `gspread==6.1.2`, but `scripts/annotation/build_packets_sheet.py`
was transcribed from a plan written against older gspread call forms. Verify each
of these against the installed 6.1.2 API **before** running `build_packets.py --push`.

1. **[HIGH] `ws.update("A1", rows, value_input_option="RAW")` argument order** —
   gspread 6.0+ reordered `Worksheet.update` to values-first
   (`update(values, range_name=None, ...)`). As written, this call binds
   `values="A1"` and `range_name=rows`, which would silently write garbage to the
   sheet. Check the installed signature (`python -c "import gspread, inspect;
   print(inspect.signature(gspread.Worksheet.update))"`) and, if it's
   values-first, correct the call to `ws.update(rows, "A1", value_input_option="RAW")`.

2. **`ValidationConditionType` path/casing** — the code references
   `gspread.worksheet.ValidationConditionType.one_of_list`. In some 6.x releases
   this enum lives at `gspread.utils.ValidationConditionType.ONE_OF_LIST`
   (upper snake case). Confirm the module path, member name/casing, and the
   `add_validation(range, condition_type, values, strict=..., showCustomUi=...)`
   parameter names and order against 6.1.2 before running.

3. **`ws.hide_columns(7, 8)` semantics** — confirm this 0-based half-open
   interval hides only index 7 (column H, the hidden `con_legis_num`) and
   leaves column G (Notes) visible.

4. **[HIGH — functional acceptance, not just a signature check] Protected
   ranges actually lock interns out.** Interns have Editor access to the Sheet,
   and `add_protected_range(...)` is called with no `editors=` /
   `requesting_user_can_edit=` argument. Verify (a) `description=` is a valid
   kwarg in 6.1.2, and (b) — critically, by testing in the browser as an
   intern-level Editor after pushing — that columns A-E and the hidden
   `con_legis_num` truly cannot be edited, not merely that editing them shows a
   dismissible warning. This is the sprint's core acceptance criterion
   ("interns only pick Yes/No/Unsure; every other field locked"). If a bare
   protected range only warns instead of blocking, add explicit editor
   restrictions before handing the Sheet to interns.
