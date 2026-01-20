
# Væksthus Brainstorm — Streamlit App

This repository hosts a browser‑based brainstorm and prioritization tool with Dot Voting, Impact–Effort, and Pairwise battles. It runs on **Streamlit Cloud** with no local installs.

## Files
- `app.py` — the app
- `.streamlit/config.toml` — streamlit config
- `data/` — created at runtime for session storage
- `exports/` — CSV/Excel exports are written here
- `requirements.txt` — Python deps for Streamlit Cloud

## Quick Deploy (Streamlit Cloud)
1. Push these files to your GitHub repo (root).
2. Go to Streamlit Community Cloud and click **New app**.
3. Select this repo, choose the default branch, **main file:** `app.py`.
4. Click **Deploy**. After build, your app opens at a public URL.

## Facilitator Flow
1. In the sidebar choose **Facilitator** → **Create new session** to get a 6‑character code.
2. Share the code with participants; they select **Participant**, enter the code and their display name.
3. Use **Stage** to move from *Ideate* → *Prioritize* → *Results*.
4. Hit **Export** in *Results* to download CSV/Excel.

## Notes
- Streamlit Cloud filesystem is ephemeral; exports should be downloaded after the workshop.
- For enterprise persistence, port this to **Power Apps + Dataverse** or wire Azure Blob Storage.
