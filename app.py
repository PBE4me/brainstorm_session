
import os
import json
import random
import string
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

APP_NAME = "Væksthus Brainstorm"
DATA_DIR = Path("data/sessions")
EXPORT_DIR = Path("exports")
DATA_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------- Utilities -----------------------------

def _session_path(code: str) -> Path:
    return DATA_DIR / f"{code.upper()}.json"


def gen_code(n: int = 6) -> str:
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=n))


def load_session(code: str) -> dict:
    p = _session_path(code)
    if p.exists():
        with p.open('r', encoding='utf-8') as f:
            return json.load(f)
    # default structure
    return {
        "meta": {
            "code": code.upper(),
            "title": APP_NAME,
            "goal": "",
            "created_at": datetime.utcnow().isoformat(),
            "language": "en",
            "modes": {"dot": True, "impact_effort": True, "pairwise": False},
            "stage": "ideate",  # ideate | prioritize | results
            "dots_per_person": 5,
            "criteria_weights": {"dots": 0.4, "elo": 0.3, "impact_effort": 0.3},
        },
        "ideas": {},              # idea_id -> {title, desc, author, created_at}
        "participants": {},       # user_id -> {name, joined_at}
        "votes": [],              # list of {user_id, idea_id, points}
        "ratings": [],            # list of {user_id, idea_id, impact, effort}
        "comparisons": [],        # list of {user_id, a, b, chosen}
    }


def save_session(state: dict):
    p = _session_path(state["meta"]["code"])
    tmp = p.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    tmp.replace(p)


def ensure_user_id() -> str:
    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = ''.join(random.choices(string.ascii_lowercase+string.digits, k=8))
    return st.session_state['user_id']

# ----------------------------- Scoring -----------------------------

def normalize(series):
    s = pd.Series(series, dtype=float)
    if len(s) == 0:
        return s
    mn, mx = s.min(), s.max()
    if mx - mn == 0:
        return pd.Series([0.5]*len(s), index=s.index)
    return (s - mn) / (mx - mn)


def compute_elo(ideas, comparisons, k=32):
    # Initialize ratings
    ids = list(ideas.keys())
    elo = {i: 1000.0 for i in ids}
    # Shuffle to avoid bias
    comps = list(comparisons)
    random.shuffle(comps)
    for c in comps:
        a, b, chosen = c['a'], c['b'], c['chosen']
        if a not in elo or b not in elo:
            continue
        ra, rb = elo[a], elo[b]
        ea = 1.0 / (1.0 + 10 ** ((rb - ra)/400))
        eb = 1.0 / (1.0 + 10 ** ((ra - rb)/400))
        sa = 1.0 if chosen == a else 0.0
        sb = 1.0 if chosen == b else 0.0
        elo[a] = ra + k*(sa - ea)
        elo[b] = rb + k*(sb - eb)
    return pd.Series(elo)


def aggregate_results(state):
    ideas = state['ideas']
    if not ideas:
        return pd.DataFrame(columns=['idea_id','title','desc','author','dots','elo','impact_effort','score'])

    df_ideas = pd.DataFrame([
        {"idea_id": iid, **idata} for iid, idata in ideas.items()
    ])

    # Dots
    if state['votes']:
        df_votes = pd.DataFrame(state['votes'])
        dots = df_votes.groupby('idea_id')['points'].sum()
    else:
        dots = pd.Series({iid: 0 for iid in ideas.keys()})
    dots_n = normalize(dots)

    # Impact/Effort
    if state['ratings']:
        df_r = pd.DataFrame(state['ratings'])
        agg = df_r.groupby('idea_id')[['impact','effort']].mean()
        impact_effort_score = agg['impact'] - 0.6*agg['effort']
    else:
        impact_effort_score = pd.Series({iid: 0 for iid in ideas.keys()}, dtype=float)
    ie_n = normalize(impact_effort_score)

    # Pairwise Elo
    elo_raw = compute_elo(ideas, state['comparisons']) if state['comparisons'] else pd.Series({iid: 1000 for iid in ideas.keys()}, dtype=float)
    elo_n = normalize(elo_raw)

    w = state['meta']['criteria_weights']
    final_score = w['dots']*dots_n + w['elo']*elo_n + w['impact_effort']*ie_n

    out = df_ideas.merge(dots_n.rename('dots'), left_on('idea_id'), right_index=True, how='left')
    
    out = df_ideas.merge(dots_n.rename('dots'), left_on='idea_id', right_index=True, how='left')                   .merge(elo_n.rename('elo'), left_on='idea_id', right_index=True, how='left')                   .merge(ie_n.rename('impact_effort'), left_on='idea_id', right_index=True, how='left')
    out['score'] = final_score.values
    out = out.sort_values('score', ascending=False).reset_index(drop=True)
    return out

# ----------------------------- UI Blocks -----------------------------

def sidebar_role_and_code():
    st.sidebar.header("Role & Session")
    role = st.sidebar.radio("I am a", ["Facilitator", "Participant"], horizontal=True)
    code = st.sidebar.text_input("Session code", value=st.session_state.get('session_code', ''))
    col1, col2 = st.sidebar.columns([1,1])
    if col1.button("Create new session"):
        code_new = gen_code()
        st.session_state['session_code'] = code_new
        state = load_session(code_new)
        save_session(state)
        st.success(f"Created session {code_new}")
        st.experimental_rerun()
    if code:
        st.session_state['session_code'] = code.upper()
    return role, st.session_state.get('session_code')


def facilitator_panel(state):
    st.subheader("Facilitator")
    meta = state['meta']
    meta['title'] = st.text_input("Session title", meta.get('title', APP_NAME))
    meta['goal'] = st.text_area("Goal / prompt", meta.get('goal', ""))
    meta['language'] = st.selectbox("Language", ["en", "da"], index=0 if meta.get('language','en')=='en' else 1)

    st.markdown("**Game modes**")
    m = meta['modes']
    c1,c2,c3 = st.columns(3)
    m['dot'] = c1.checkbox("Dot voting", value=m.get('dot', True))
    m['impact_effort'] = c2.checkbox("Impact–Effort", value=m.get('impact_effort', True))
    m['pairwise'] = c3.checkbox("Pairwise battles", value=m.get('pairwise', False))

    meta['dots_per_person'] = st.number_input("Dots per participant", min_value=1, max_value=20, value=meta.get('dots_per_person',5))

    st.markdown("**Stage**")
    meta['stage'] = st.radio("", ["ideate","prioritize","results"], horizontal=True, index=["ideate","prioritize","results"].index(meta.get('stage','ideate')))

    st.markdown("**Weights (final scoring)**")
    w = meta['criteria_weights']
    w['dots'] = st.slider("Dots weight", 0.0, 1.0, float(w.get('dots',0.4)), 0.05)
    w['elo'] = st.slider("Pairwise (Elo) weight", 0.0, 1.0, float(w.get('elo',0.3)), 0.05)
    w['impact_effort'] = st.slider("Impact–Effort weight", 0.0, 1.0, float(w.get('impact_effort',0.3)), 0.05)
    total = w['dots'] + w['elo'] + w['impact_effort']
    if abs(total-1.0) > 1e-6:
        st.warning(f"Weights sum to {total:.2f}. They should sum to 1. We'll normalize on save.")

    if st.button("Save session settings"):
        # Normalize weights
        s = sum(w.values())
        if s > 0:
            for k in list(w.keys()):
                w[k] = float(w[k]) / s
        save_session(state)
        st.success("Saved")

    st.divider()

    st.markdown("**Ideas (manage)**")
    if state['ideas']:
        df = pd.DataFrame([{**v, 'idea_id':k} for k,v in state['ideas'].items()])[["idea_id","title","desc","author","created_at"]]
        st.dataframe(df, use_container_width=True)
    colA, colB = st.columns([2,1])
    with colA:
        bulk = st.text_area("Bulk add ideas (one per line)")
    with colB:
        author = st.text_input("Author (optional)")
        if st.button("Add ideas"):
            lines = [l.strip() for l in bulk.splitlines() if l.strip()]
            for t in lines:
                iid = ''.join(random.choices(string.ascii_uppercase+string.digits, k=6))
                state['ideas'][iid] = {
                    'title': t[:120],
                    'desc': '',
                    'author': author or 'facilitator',
                    'created_at': datetime.utcnow().isoformat(),
                }
            save_session(state)
            st.success(f"Added {len(lines)} ideas")
            st.experimental_rerun()


def participant_panel(state):
    st.subheader("Participant")
    uid = ensure_user_id()
    name = st.text_input("Your display name", value=st.session_state.get('display_name',''))
    if name:
        st.session_state['display_name'] = name
        if uid not in state['participants']:
            state['participants'][uid] = {"name": name, "joined_at": datetime.utcnow().isoformat()}
            save_session(state)
    st.info(f"Stage: {state['meta']['stage'].upper()} | Modes: " + ", ".join([k for k,v in state['meta']['modes'].items() if v]))

    if state['meta']['stage'] == 'ideate':
        st.markdown("### Add ideas")
        col1, col2 = st.columns([2,1])
        with col1:
            title = st.text_input("Idea title")
            desc = st.text_area("Short description (optional)")
        with col2:
            if st.button("Submit idea") and title.strip():
                iid = ''.join(random.choices(string.ascii_uppercase+string.digits, k=6))
                state['ideas'][iid] = {
                    'title': title.strip()[:120],
                    'desc': desc.strip(),
                    'author': name or 'anon',
                    'created_at': datetime.utcnow().isoformat(),
                }
                save_session(state)
                st.success("Idea added")
                st.experimental_rerun()
        st.divider()
        if state['ideas']:
            st.markdown("#### Current ideas")
            st.table(pd.DataFrame([{**v, 'idea_id':k} for k,v in state['ideas'].items()])[['idea_id','title','author']])

    elif state['meta']['stage'] == 'prioritize':
        st.markdown("### Prioritization")
        modes = state['meta']['modes']
        ideas = state['ideas']
        if not ideas:
            st.warning("No ideas yet.")
            return
        ids = list(ideas.keys())

        if modes.get('dot', True):
            st.markdown("#### Dot voting")
            remaining = state['meta'].get('dots_per_person',5) - sum(v['points'] for v in state['votes'] if v['user_id']==uid)
            st.caption(f"You have {max(0, remaining)} dots left")
            options = {f"{ideas[i]['title']} ({i})": i for i in ids}
            choice = st.selectbox("Pick an idea to award dots", list(options.keys()))
            pts = st.slider("Dots", 0, max(remaining,0), min(2, max(0, remaining)))
            if st.button("Give dots"):
                if pts>0 and remaining>0:
                    state['votes'].append({"user_id": uid, "idea_id": options[choice], "points": int(pts)})
                    save_session(state)
                    st.experimental_rerun()

        if modes.get('impact_effort', True):
            st.markdown("#### Impact–Effort ratings")
            target = st.selectbox("Which idea to rate?", ids, format_func=lambda x: ideas[x]['title'])
            impact = st.slider("Impact (1=low, 5=high)", 1, 5, 4)
            effort = st.slider("Effort (1=low, 5=high)", 1, 5, 2)
            if st.button("Submit rating"):
                state['ratings'].append({"user_id": uid, "idea_id": target, "impact": int(impact), "effort": int(effort)})
                save_session(state)
                st.success("Rating saved")

        if modes.get('pairwise', False):
            st.markdown("#### Pairwise battles")
            if len(ids) >= 2:
                a, b = random.sample(ids, 2)
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(ideas[a]['title']):
                        state['comparisons'].append({"user_id": uid, "a": a, "b": b, "chosen": a})
                        save_session(state)
                        st.experimental_rerun()
                with col2:
                    if st.button(ideas[b]['title']):
                        state['comparisons'].append({"user_id": uid, "a": a, "b": b, "chosen": b})
                        save_session(state)
                        st.experimental_rerun()
            else:
                st.info("Need at least two ideas for pairwise battles.")

    else:  # results
        st.markdown("### Results")
        res = aggregate_results(state)
        if not res.empty:
            st.dataframe(res[['title','dots','elo','impact_effort','score']].rename(columns={
                'title':'Idea','dots':'Dots','elo':'Pairwise','impact_effort':'Impact–Effort','score':'Score'
            }), use_container_width=True)
            # Export buttons
            if st.button("Export CSV & Excel"):
                ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                base = f"session_{state['meta']['code']}_{ts}"
                # CSVs
                pd.DataFrame([{**v,'idea_id':k} for k,v in state['ideas'].items()]).to_csv(EXPORT_DIR / f"{base}_ideas.csv", index=False)
                pd.DataFrame(state['votes']).to_csv(EXPORT_DIR / f"{base}_votes.csv", index=False)
                pd.DataFrame(state['ratings']).to_csv(EXPORT_DIR / f"{base}_ratings.csv", index=False)
                pd.DataFrame(state['comparisons']).to_csv(EXPORT_DIR / f"{base}_comparisons.csv", index=False)
                res.to_csv(EXPORT_DIR / f"{base}_results.csv", index=False)
                # Excel workbook
                xlsx_path = EXPORT_DIR / f"{base}_results.xlsx"
                with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
                    pd.DataFrame([{**v,'idea_id':k} for k,v in state['ideas'].items()]).to_excel(writer, index=False, sheet_name='ideas')
                    pd.DataFrame(state['votes']).to_excel(writer, index=False, sheet_name='votes')
                    pd.DataFrame(state['ratings']).to_excel(writer, index=False, sheet_name='ratings')
                    pd.DataFrame(state['comparisons']).to_excel(writer, index=False, sheet_name='comparisons')
                    res.to_excel(writer, index=False, sheet_name='results')
                st.success(f"Exports written to '{EXPORT_DIR}' folder.")
        else:
            st.info("No results yet. Collect some input first.")

# ----------------------------- App -----------------------------

st.set_page_config(page_title=APP_NAME, layout='wide')
st.title(APP_NAME)
st.caption("Structured ideation, gamified prioritization, instant exports.")

role, code = sidebar_role_and_code()

if not code:
    st.info("Create a session or enter an existing code in the sidebar.")
    st.stop()

state = load_session(code)

if role == "Facilitator":
    facilitator_panel(state)
else:
    participant_panel(state)
