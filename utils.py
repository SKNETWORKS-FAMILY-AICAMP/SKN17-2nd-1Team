import re
import numpy as np
import pandas as pd
import streamlit as st
from collections import Counter
from joblib import load
from urllib.parse import quote_plus
from catboost import CatBoostClassifier
from sklearn.preprocessing import MinMaxScaler

# --- 상수 정의 ---
FILE_PATH = './data/'
MODEL_PATH = './model/'

# --- 데이터 및 모델 로드 (캐싱 적용) ---
@st.cache_data
def load_data():
    df = pd.read_csv(f'{FILE_PATH}survey_results_cleaned.csv')
    df_fin = pd.read_csv(f'{FILE_PATH}survey_results_cleaned_scaler.csv')
    
    _df_cb = df.loc[df_fin.index].copy()
    for _c in ['LearnCode','DevType','LanguageHaveWorkedWith','SOHow']:
        if _c not in _df_cb.columns: _df_cb[_c] = np.nan
        _df_cb[_c] = _df_cb[_c].fillna('')

    _df_cb_use = pd.concat([
        df_fin.drop('is_churned', axis=1),
        _df_cb[['LearnCode','DevType','LanguageHaveWorkedWith','SOHow']]
    ], axis=1)
    
    peers_df = _df_cb_use.join(df_fin['is_churned'])
    peers_full_keep = peers_df[peers_df['is_churned'] == 0].copy()
    X_cols = df_fin.drop('is_churned', axis=1).columns.tolist()
    return peers_full_keep, X_cols

@st.cache_resource
def load_models():
    try:
        model = load(f"{MODEL_PATH}catboost_model.joblib")
        scaler = load(f"{MODEL_PATH}scaler.joblib")
    except FileNotFoundError:
        st.error(f"Model files not found in '{MODEL_PATH}' folder. Please run 'train_model.py' first.")
        st.stop()
    return model, scaler

# --- 입력 데이터 처리 함수 ---
def transform_inputs_for_prediction(form_data, all_feature_columns):
    def _bin_by_edges(x, edges, labels):
        idx = np.digitize([x], edges, right=True)[0] - 1
        return int(labels[max(0, min(idx, len(labels)-1))])

    ai_integration_data = form_data.get('ai_integration', {})
    forecast_score = sum(1 if v == 'More integrated' else 2 for v in ai_integration_data.values())

    input_dict = {
        'Lang_Diversity': _bin_by_edges(len(form_data.get('languages', [])), [0, 2, 4, 6, 8, 10, 15, 50], list(range(7))),
        'AI_Tool_Count': _bin_by_edges(len(form_data.get('ai_tools', [])), [0, 1, 3, 5, 8, 13], list(range(5))),
        'WorkExp': _bin_by_edges(form_data.get('workexp', 0), [-1, 2, 5, 10, 20, float('inf')], list(range(1, 6))),
        'YearsCode': _bin_by_edges(form_data.get('years_total', 0), [-1, 2, 5, 10, 20, float('inf')], list(range(1, 6))),
        'DevRole_Count': _bin_by_edges(len(form_data.get('dev_selected', [])), [-1, 1, 2, 3, 5, float('inf')], list(range(5))),
        'SOComm_encoded': {'No, not at all':0,'No, not really':1,'Not sure':2,'Neutral':3,'Yes, somewhat':4,'Yes, definitely':5}.get(form_data.get('so_comm'), 2),
        'AIThreat_num': {'Yes':1,'No':0,"I'm not sure":0.5}.get(form_data.get('ai_threat'), 0.5),
        'Age_encoded': form_data.get('age_encoded', 2),
        'NEWSOSites_count': len(form_data.get('newsites_sel', [])),
        'SOHow_count': len(form_data.get('sohow_selected_en', [])),
        'AIForecastScore': forecast_score,
        'Challenges_count': 0,
    }
    input_row = pd.DataFrame([input_dict])
    return input_row.reindex(columns=all_feature_columns, fill_value=0)

# --- 추천 로직 관련 모든 헬퍼 함수 (원본 기능 전체 복구) ---
_def_sep = re.compile(r"\s*;\s*")
def _split_sc(s: str):
    if not isinstance(s, str) or not s.strip(): return []
    return [p.strip() for p in _def_sep.split(s) if p and p.strip()]

def _dedup(seq):
    out, seen = [], set()
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _langs_to_tags(lang_list):
    LANG_TAG_MAP = {'Python': ['python'], 'JavaScript': ['javascript'], 'TypeScript': ['typescript'], 'HTML/CSS': ['html', 'css'], 'SQL': ['sql']}
    tags = [tag for lang in lang_list for tag in LANG_TAG_MAP.get(lang, [re.sub(r"[^a-z0-9+#.-]", "-", str(lang).strip().lower())])]
    return _dedup(tags)

def SOHOW_TOPIC_MAP_EN(sohow_en_list, top_k=2):
    SOHOW_TOPIC_MAP = {
        'Quickly finding code solutions': 'Error resolution / Bug reproduction',
        'Finding reliable guidance': 'Best practices / Design comparison',
        'Learning new-to-me technology': 'Understanding new concepts/samples',
        'Learning new-to-everyone technology': 'Latest tech trends/examples',
        'Showcase expertise': 'Sharing practical solutions',
        'Engage with community': 'Feedback for code/answer improvement'
    }
    counts = Counter(SOHOW_TOPIC_MAP.get(s) for s in sohow_en_list if SOHOW_TOPIC_MAP.get(s))
    return [k for k, _ in counts.most_common(top_k)] or ['Error resolution / Bug reproduction']

def build_service_plan(risk_score, newsites_count, sohow_en, time_min):
    path = 'Onboarding' if (newsites_count <= 1 or len(sohow_en) == 0) else 'Contributor'
    if time_min < 30: level = 'light'
    elif time_min < 90: level = 'mid'
    else: level = 'high'
    goals = {
        'light': {'read': 3, 'comment': 1, 'ask_prep': 1, 'ask_post': 0, 'follow': 1},
        'mid':   {'read': 6, 'comment': 2, 'ask_prep': 1, 'ask_post': 1 if path == 'Contributor' else 0, 'follow': 1},
        'high':  {'read':12, 'comment': 3, 'ask_prep': 1, 'ask_post': 2 if path == 'Contributor' else 0, 'follow': 2}
    }[level]
    tasks = [
        {'Task': 'Follow Tags', 'Goal': f"{goals['follow']} tag(s)", 'How-to': "Follow tags related to your interests"},
        {'Task': 'Read & Bookmark', 'Goal': f"{goals['read']} Q&As", 'How-to': "Bookmark high-quality answers"},
        {'Task': 'Leave Comments', 'Goal': f"{goals['comment']} comment(s)", 'How-to': "Ask for clarification or provide feedback"},
    ]
    if goals['ask_post'] > 0:
        tasks.append({'Task': 'Post a Question', 'Goal': f"{goals['ask_post']} question(s)", 'How-to': "Use a template, include a Minimal, Reproducible Example (MRE)"})
    else:
        tasks.append({'Task': 'Draft a Question', 'Goal': f"{goals['ask_prep']} draft(s)", 'How-to': "Prepare a question using the template (don't post yet)"})
    return path, level.upper(), tasks

def build_question_template(tags, topics):
    primary = tags[0] if tags else 'programming'
    topic = topics[0] if topics else 'Error resolution'
    tag_line = ', '.join(tags[:5]) if tags else primary
    return (
        f"[Title] [{primary}] {topic} — One-line summary of the issue\n\n"
        f"[Body Checklist]\n"
        f"1) Environment: OS / Language/Framework versions\n"
        f"2) What I've tried: Summary of attempts\n"
        f"3) Actual vs. Expected: What is different?\n"
        f"4) Minimal, Reproducible Example (MRE):\n```\n# Paste the smallest code that reproduces the problem here\n```\n\n"
        f"5) Error Logs (if any): Paste error messages/stack traces\n\n"
        f"[Suggested Tags] {tag_line}\n"
    )

def build_saved_search_queries(tags, topics):
    t = (tags or ['programming'])[:2]
    topic = topics[0] if topics else 'Error resolution'
    return [
        f'is:question created:7d score:0 answers:0 tag:{t[0]}',
        f'is:question created:7d tag:{" ".join(t)} "{topic}"'.strip(),
        f'is:question hasaccepted:yes tag:{t[0]} duplicate:no'
    ]

def so_tag_url(tag: str) -> str:
    return f"https://stackoverflow.com/questions/tagged/{quote_plus((tag or '').strip('# '))}"

def so_ask_url(tags: list[str]) -> str:
    safe_tags = ';'.join([t.strip('# ') for t in (tags or []) if t])
    return f"https://stackoverflow.com/questions/ask?tags={quote_plus(safe_tags)}" if safe_tags else "https://stackoverflow.com/questions/ask"

def parse_goal_int(s: str) -> int:
    m = re.search(r"(\d+)", str(s))
    return int(m.group(1)) if m else 1

POINTS_PER_UNIT = {'Follow Tags': 10, 'Read & Bookmark': 2, 'Leave Comments': 6, 'Draft a Question': 6, 'Post a Question': 20}