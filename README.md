
# Væksthus Brainstorm (Streamlit Prototype)

A lightweight web app for structured ideation and **gamified prioritization** (Dot Voting, Impact–Effort, Pairwise Battles). Stores sessions locally (JSON) and exports results to CSV/Excel.

---
## Quick start

1. **Create and activate a virtual environment** (optional but recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app**
   ```bash
   streamlit run app.py
   ```
4. Open the local URL (usually http://localhost:8501).

---
## How it works
- **Facilitator** creates a session (title, goal, modes) and sets the **current stage**.
- Participants join with the **Session Code** and a display name.
- **Stages**: Ideate → (optional) Cluster → Prioritize (choose game mode) → Results.
- **Storage**: `data/sessions/<session_id>.json` (safe for multiple browser clients on the same machine).
- **Exports**: CSV and Excel files in the `exports/` folder.

> This prototype is file‑based by design so it can run without any external services. For production, move storage to a database (e.g., Azure SQL/Dataverse) and add auth.

---
## Game modes
- **Dot Voting**: each participant gets a limited number of dots to allocate.
- **Impact–Effort (2×2)**: rate Impact and Effort (1–5). Auto ranking = mean(Impact) – mean(Effort).
- **Pairwise Battles**: rapid A vs. B choices; global ranking uses an Elo‑style rating.

---
## Files
- `app.py`: Streamlit app
- `requirements.txt`
- `.streamlit/config.toml`: theme
- `data/` & `exports/`: local storage

---
## Notes
- Multiple participants can connect from different browsers to the same machine. Streamlit serves a single process; this app uses JSON files for cross‑client state.
- If two writes collide, last write wins (good enough for workshops). For production, use row‑level storage.
