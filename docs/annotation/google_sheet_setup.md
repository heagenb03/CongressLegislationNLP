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

The key file is never committed. `.gitignore` already excludes `*.json` under
credential paths — keep the key outside the repo to be safe.
