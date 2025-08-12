import streamlit as st
import numpy as np
import pandas as pd
import utils
import config

# streamlit 실행 파일
# --- 1. 데이터/모델 로드 및 기본 설정 ---
peers_full_keep, X_cols = utils.load_data()
model, scaler = utils.load_models()
OPTIMAL_THRESHOLD = 0.5

# --- 2. UI 구성 (CSS 및 제목) ---
st.set_page_config(
    page_title="2025 Developer Survey",
    page_icon="🚀",
    layout="wide",
)
st.title("🚀 2025 Stack Overflow Developer Survey")
st.markdown("")
st.markdown("Hello World!")
st.markdown("Thank you for taking the 2025 Stack Overflow Developer Survey, the longest running survey of software developers (and anyone else who codes!) on Earth.")
st.markdown("---")
st.markdown("""
<style>
    /* Google Fonts 임포트 */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    /* ... (이하 CSS 코드는 이전과 동일) ... */
    html, body, [class*="st-"] { font-family: 'Poppins', sans-serif; }
    h1 { color: #FFFFFF; text-shadow: 2px 2px 8px rgba(255,255,255,0.3); }
    h4 { color: #C1C8D1; font-weight: 400; margin-top: 1.5em; margin-bottom: 0.5em; }
    div.st-emotion-cache-1r6slb0 > div {
        background: linear-gradient(145deg, #1e2025, #23272e);
        box-shadow: 5px 5px 15px #1a1c20, -5px -5px 15px #2c3038;
        border-radius: 15px; border: 1px solid #2c3038; padding: 2rem; margin-bottom: 1.5rem;
    }
    div[data-testid="stForm"] div.stButton > button {
        border: none; border-radius: 25px; font-weight: 600; color: #FFFFFF;
        width: 100%; padding: 12px 0; background: linear-gradient(45deg, #6a11cb 0%, #2575fc 100%);
        transition: all 0.3s ease-in-out;
    }
    div[data-testid="stForm"] div.stButton > button:hover {
        transform: scale(1.02); box-shadow: 0 0 25px #2575fc;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. 설문조사 폼 ---
if 'results_visible' not in st.session_state:
    st.session_state.results_visible = False

with st.form('survey_form'):
    form_data = {}
    with st.container(border=False):
        st.header('1) Basic Information')
        age_map = {'Under 18 years old':0, '18-24 years':1, '25-34 years':2, '35-44 years':3, '45-54 years':4, '55-64 years':5, '65 years or older':6}
        form_data['age_label'] = st.selectbox('Your age group', list(age_map.keys()), index=None, placeholder="Select your age group...")

        form_data['years_total'] = st.slider('Total years of coding experience (including education)', 0, 50, 0, 1)
        form_data['workexp'] = st.slider('Years of professional coding experience', 0, 50, 0, 1)

    with st.container(border=False):
        st.header('2) Your Role & Tech Stack')
        form_data['dev_selected'] = st.multiselect('Your current developer role(s)', config.DEVTYPE_OPTIONS, default=[])
        form_data['languages'] = st.multiselect('Languages you have worked with recently', config.LANGUAGE_OPTIONS, default=[])
        form_data['learn_sel'] = st.multiselect('How you learn to code', config.LEARN_CODE_OPTIONS, default=[])

    with st.container(border=False):
        st.header('3) Community & AI Perception')
        ko_en_map = config.SOHOW_OPTIONS_KO_EN_MAP
        selected_ko = st.multiselect('How do you use Stack Overflow?', list(ko_en_map.keys()), default=[])
        form_data['sohow_selected_en'] = [ko_en_map[k] for k in selected_ko]
        
        c5, c6 = st.columns(2, gap="large")
        with c5:
            form_data['newsites_sel'] = st.multiselect('Which SO-related sites have you visited?', config.NEWSITES_OPTIONS, default=[])
            form_data['so_comm'] = st.selectbox('Do you feel part of the SO community?', ['No, not at all','No, not really','Not sure','Neutral','Yes, somewhat','Yes, definitely'], index=None, placeholder="Select an option...")
        with c6:
            form_data['ai_tools'] = st.multiselect('Which AI tools do you use?', config.AI_TOOLS_OPTIONS, default=[])
            form_data['ai_threat'] = st.selectbox('Is AI a threat to your job?', ['Yes','No',"I'm not sure"], index=None, placeholder="Select an option...")
        
        st.markdown("#### How integrated will AI tools be in your workflow 1 year from now?")
        with st.popover("Click to set integration level for each item", use_container_width=True):
            integration_options = ['More integrated', 'Much more integrated']
            form_data['ai_integration'] = {}
            for part in config.AI_WORKFLOW_PARTS:
                form_data['ai_integration'][part] = st.radio(
                    f"» **{part}**", integration_options, horizontal=True, key=f"ai_integ_{part}", index=0
                )

    st.header("Part 4: Your Goal (Not used for prediction)")
    form_data['time_budget'] = st.slider('How much time (minutes) can you invest in the community this week?', 15, 180, 60, step=15)
    
    submitted = st.form_submit_button('Analyze & Get My Action Plan')
    if submitted:
        if not form_data['age_label']:
            st.warning("Please select your age group before submitting.")
            st.session_state.results_visible = False
        else:
            form_data['age_encoded'] = age_map[form_data['age_label']]
            st.session_state.results_visible = True
            st.session_state.form_data = form_data

# --- 4. 결과 처리 및 표시 ---
if st.session_state.get('results_visible', False):
    form_data = st.session_state.form_data
    
    model_features = utils.transform_inputs_for_prediction(form_data, X_cols)
    scaled_features = scaler.transform(model_features)
    risk_score = model.predict_proba(scaled_features)[0, 1]
    label = 'High Risk' if risk_score >= OPTIMAL_THRESHOLD else 'Low Risk'

    st.divider()
    st.header('📊 Your Analysis Result & Personalized Plan')
    c1, c2 = st.columns(2)
    c1.metric('Churn Risk Score', f"{risk_score*100:.1f}%")
    c2.metric('Prediction', label)

    if risk_score >= OPTIMAL_THRESHOLD:
        with st.container(border=True):
            tags = utils._langs_to_tags(form_data['languages'])
            topics = utils.SOHOW_TOPIC_MAP_EN(form_data['sohow_selected_en'])
            path, level, tasks = utils.build_service_plan(risk_score, len(form_data['newsites_sel']), form_data['sohow_selected_en'], form_data['time_budget'])
            
            st.markdown(f"**Your Personalized Path**: {path} | **Intensity**: {level} | **Weekly Time**: {form_data['time_budget']} mins")
            st.markdown('**Your Weekly Action Checklist**')

            task_df = pd.DataFrame(tasks)
            task_df['goal_n'] = task_df['Goal'].apply(utils.parse_goal_int)
            task_df = task_df[task_df['goal_n'] > 0].reset_index(drop=True)

            total_progress, total_points = [], 0
            for i, row in task_df.iterrows():
                task_id = f"task_{i}"
                c1, c2, c3, c4 = st.columns([1.3, 1.6, 0.9, 0.8])
                with c1: st.markdown(f"**{row['Task']}**")
                with c2: st.caption(row['How-to'])
                with c3:
                    done_count = 0
                    cols_units = st.columns(min(row['goal_n'], 5))
                    for j in range(row['goal_n']):
                        with cols_units[j % len(cols_units)]:
                            if st.checkbox(f"_{j+1}", key=f"{task_id}_{j}", label_visibility="hidden"):
                                done_count += 1
                    st.write(f"**{done_count} / {row['goal_n']}**")
                with c4:
                    if row['Task'] == 'Follow Tags': st.link_button("Go", utils.so_tag_url(tags[0] if tags else ''))
                    elif row['Task'] in ['Post a Question', 'Draft a Question']: st.link_button("Go", utils.so_ask_url(tags))

                task_progress = min(done_count / row['goal_n'], 1.0) if row['goal_n'] > 0 else 1.0
                total_progress.append(task_progress)
                total_points += utils.POINTS_PER_UNIT.get(row['Task'], 0) * done_count
            
            overall_progress = sum(total_progress) / len(total_progress) if total_progress else 1.0
            st.progress(overall_progress, text=f"Overall Progress: {overall_progress:.0%} | Points Earned: {total_points}")
            if overall_progress >= 1.0:
                st.balloons()
                st.success('Challenge Complete! 🏅 +50 Bonus Points')
        
        st.markdown("")
        tab1, tab2 = st.tabs(["💡 Personalized Question Template", "🔍 Recommended Search Queries"])

        with tab1:
            st.markdown("Copy and use this template for high-quality questions.")
            st.code(utils.build_question_template(tags, topics), language='markdown')
        
        with tab2:
            st.markdown("Save these queries on Stack Overflow to find questions you can answer.")
            queries = utils.build_saved_search_queries(tags, topics)
            st.code("\n".join(queries))
    else:
        st.success("✨ You're an engaged community member! Keep up the great work.")