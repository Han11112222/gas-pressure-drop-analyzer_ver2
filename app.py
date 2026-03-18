import streamlit as st
import pandas as pd
import io
import os
import tempfile
import json
import time
import google.generativeai as genai

st.set_page_config(page_title="공동주택 관경 적합성 검토기", layout="wide")

STANDARD_CAL = 10145 
STANDARD_PRESSURE = 0.3000 

pipe_data = {
    '400P': {'inner_d': 32.92, 'ball': 2.36, 'el90': 9.53, 'el45': 4.76, 'tee': 28.50, 'tee14': 9.53, 'red12': 4.05},
    '355P': {'inner_d': 29.04, 'ball': 2.13, 'el90': 8.51, 'el45': 4.25, 'tee': 24.68, 'tee14': 7.66, 'red12': 3.49},
    '280P': {'inner_d': 22.92, 'ball': 1.65, 'el90': 7.28, 'el45': 3.64, 'tee': 18.77, 'tee14': 6.53,  'red12': 2.63},
    '225P': {'inner_d': 18.50, 'ball': 1.31, 'el90': 5.82,  'el45': 2.91, 'tee': 12.74, 'tee14': 5.34,  'red12': 2.18},
    '160P': {'inner_d': 13.18, 'ball': 0.93, 'el90': 4.07,  'el45': 2.04, 'tee': 8.49, 'tee14': 3.65,  'red12': 1.53},
    '90P':  {'inner_d': 7.36,  'ball': 0.49, 'el90': 2.24,  'el45': 1.12, 'tee': 3.79,  'tee14': 1.32,  'red12': 0.84},
    '65S':  {'inner_d': 6.90,  'ball': 0.43, 'el90': 2.00,  'el45': 1.00, 'tee': 3.20,  'tee14': 1.30,  'red12': 0.70},
    '50S':  {'inner_d': 5.32,  'ball': 0.35, 'el90': 1.70,  'el45': 0.85, 'tee': 2.60,  'tee14': 1.00,  'red12': 0.60},
    '40S':  {'inner_d': 4.21,  'ball': 0.30, 'el90': 1.40,  'el45': 0.70, 'tee': 2.10,  'tee14': 0.70,  'red12': 0.45}
}

default_unit_costs = {
    '400P': 350000, '355P': 280000, '280P': 200000, '225P': 150000,
    '160P': 100000, '90P': 60000, '65S': 50000, '50S': 40000, '40S': 35000
}

def get_sim_rate(n):
    if n <= 0: return 0.0
    elif n <= 2: return 1.0
    elif n <= 5: return 0.80
    elif n <= 10: return 0.65
    elif n <= 15: return 0.58
    elif n <= 30: return 0.44
    elif n <= 45: return 0.39
    elif n <= 60: return 0.36
    elif n <= 75: return 0.35
    elif n <= 90: return 0.34
    elif n <= 105: return 0.33
    elif n <= 120: return 0.33
    elif n <= 150: return 0.31
    elif n <= 200: return 0.30
    elif n <= 300: return 0.29
    else: return 0.28

if 'reset_data' not in st.session_state:
    st.session_state['reset_data'] = False
if 'ai_df' not in st.session_state:
    st.session_state['ai_df'] = pd.DataFrame()

input_columns = ['구간', '세대수(세대)', '선정관경', '직관길이(m)', '볼밸브(개)', '90도엘보(개)', '45도엘보(개)', '동경티(개)', '1/4축소티(개)', '1/2레듀샤(개)']
empty_df = pd.DataFrame(columns=input_columns)
df = empty_df.copy()

# ==========================================
# 좌측 사이드바
# ==========================================
with st.sidebar:
    st.title("메뉴 이동")
    menu = st.radio("작업 모드를 선택하세요:", ["📊 1. 관경 산출 (엑셀/수기)", "🤖 2. 관경 산출 고도화 (도면 AI)"])
    st.markdown("---")

    if menu == "📊 1. 관경 산출 (엑셀/수기)":
        st.header("⚙️ 엑셀 데이터 불러오기")
        uploaded_file = st.file_uploader("관경산출식 엑셀/CSV 업로드", type=['xlsx', 'xls', 'csv'])
        
        if not st.session_state['reset_data']:
            if uploaded_file:
                try:
                    xls = pd.ExcelFile(uploaded_file)
                    sheet_names = [s for s in xls.sheet_names if '관경산출식' in s]
                    selected_sheet = st.selectbox("불러올 시트 선택", sheet_names if sheet_names else xls.sheet_names)
                    df_excel = pd.read_excel(uploaded_file, sheet_name=selected_sheet, skiprows=7)

                    df = df_excel.iloc[:, [1, 9, 16, 11]].copy()
                    df.columns = ['구간', '세대수(세대)', '선정관경', '직관길이(m)']
                    df = df.dropna(subset=['구간'])
                    df = df[~df['구간'].astype(str).str.contains('계|합')]
                    
                    for fitting in ['볼밸브(개)', '90도엘보(개)', '45도엘보(개)', '동경티(개)', '1/4축소티(개)', '1/2레듀샤(개)']:
                        df[fitting] = 0
                        
                    df = df[input_columns]
                    for col in ['세대수(세대)', '직관길이(m)']:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                except Exception:
                    df = pd.DataFrame([["A-B", 1740, "400P", 64.0, 0, 1, 0, 0, 0, 0]], columns=input_columns)
            else:
                df = pd.DataFrame([["A-B", 1740, "400P", 64.0, 0, 1, 0, 0, 0, 0]], columns=input_columns)

    elif menu == "🤖 2. 관경 산출 고도화 (도면 AI)":
        st.header("⚙️ AI 도면 분석기 (Gemini 1.5 Flash)")
        api_key = st.text_input("🔑 발급받은 Gemini API Key 입력", type="password")
        uploaded_pdf = st.file_uploader("도면 업로드 (PDF 한정)", type=['pdf'])
        
        if st.button("🤖 도면 분석 시작 (실제 AI 호출)"):
            if not api_key:
                st.error("API Key를 먼저 입력해 주세요!")
            elif not uploaded_pdf:
                st.error("도면 PDF 파일을 업로드해 주세요!")
            else:
                st.session_state['reset_data'] = False
                genai.configure(api_key=api_key)
                
                with st.spinner("구글 AI가 도면(PDF)을 분석 중입니다. 파일 크기에 따라 잠시 소요될 수 있습니다..."):
                    try:
                        # 1. 임시 파일로 저장
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                            tmp.write(uploaded_pdf.getvalue())
                            tmp_path = tmp.name
                            
                        # 2. 대용량 통로(File API)로 구글 서버에 업로드
                        sample_file = genai.upload_file(path=tmp_path)
                        
                        # 3. PDF 처리가 끝날 때까지 대기
                        while sample_file.state.name == "PROCESSING":
                            time.sleep(2)
                            sample_file = genai.get_file(sample_file.name)
                        
                        if sample_file.state.name == "FAILED":
                            st.error("구글 서버에서 PDF 파일을 처리하는 데 실패했습니다.")
                        else:
                            # 4. 🔥안정적인 최신 범용 모델인 gemini-1.5-flash 로 교체 적용!
                            model = genai.GenerativeModel('gemini-1.5-flash')
                            prompt = """
                            이 도면은 아파트 가스배관 관경산출 도면입니다. 도면의 배관 경로, 관경 텍스트(예: PE 400mm, RED 280x225 등), 부속류를 분석하여 각 구간별 물량을 추출하세요.
                            결과는 반드시 아래 JSON 배열(List of Dicts) 형태로만 반환하세요. 마크다운(` ```json `)이나 다른 설명은 절대 추가하지 마세요.
                            필요한 Key: '구간'(예: A-B), '세대수(세대)'(기본 1740), '선정관경'(예: 400P, 280P 등 P나 S를 붙임), '직관길이(m)'(숫자), '볼밸브(개)', '90도엘보(개)', '45도엘보(개)', '동경티(개)', '1/4축소티(개)', '1/2레듀샤(개)'.
                            값을 모르면 0으로 채우세요.
                            """
                            response = model.generate_content([sample_file, prompt])
                            
                            # 5. 응답결과 파싱 (만약 마크다운이 섞여와도 벗겨낼 수 있도록 방어코드 추가)
                            raw_text = response.text.strip()
                            if raw_text.startswith("```json"):
                                raw_text = raw_text[7:]
                            if raw_text.endswith("```"):
                                raw_text = raw_text[:-3]
                            raw_text = raw_text.strip()

                            ai_data = json.loads(raw_text)
                            st.session_state['ai_df'] = pd.DataFrame(ai_data)
                            st.toast("✅ AI 도면 분석 성공! 데이터가 에디터에 연동되었습니다.")
                            
                        # 6. 보안 파기: 서버에 올린 임시 파일 즉시 삭제
                        genai.delete_file(sample_file.name)
                        os.remove(tmp_path)
                        
                    except Exception as e:
                        st.error(f"AI 분석 중 오류가 발생했습니다. (에러내용: {e})")
                        
        if not st.session_state['ai_df'].empty and not st.session_state['reset_data']:
            df = st.session_state['ai_df'][input_columns]

# ==========================================
# 공통 UI: 메인 화면
# ==========================================
if menu == "📊 1. 관경 산출 (엑셀/수기)":
    st.title("🏢 공동주택 도시가스 관경 사전 검토기")
else:
    st.title("🚀 AI 도면 자동 인식 및 관경 산출 (Gemini API)")

st.markdown("---")

st.markdown("### 1️⃣ 세대당 가스소비량 설정")
col1, col2, col3 = st.columns(3)
boiler_kcal = col1.number_input("보일러 발열량 (kcal/hr)", value=22100, step=100)
range_kcal = col2.number_input("가스레인지 발열량 (kcal/hr)", value=7000, step=100)
household_flow = (boiler_kcal + range_kcal) / STANDARD_CAL 
col3.info(f"💡 산출된 세대당 유량: **{household_flow:.4f} ㎥/hr**")

st.markdown("---")

st.markdown("### 2️⃣ 구간별 도면 물량 데이터 에디터")
col_btn, _ = st.columns([1, 4])
with col_btn:
    if st.button("🗑️ 표 전체 지우기 (초기화)"):
        st.session_state['reset_data'] = True
        st.session_state['ai_df'] = pd.DataFrame()
        st.rerun()

df = df.fillna(0) 

edited_df = st.data_editor(
    df,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=False,
    column_config={
        "선정관경": st.column_config.SelectboxColumn("선정관경", options=list(pipe_data.keys())),
        "직관길이(m)": st.column_config.NumberColumn("직관길이(m)", format="%.2f"),
        "세대수(세대)": st.column_config.NumberColumn("세대수(세대)", format="%d"),
    }
)

edited_df['구간'] = edited_df['구간'].astype(str).str.strip() 
edited_df = edited_df[~edited_df['구간'].isin(['', '0', 'nan', 'None'])] 
edited_df = edited_df.fillna(0) 

total_actual_drop = 0
result_data = []

if not edited_df.empty:
    for idx, row in edited_df.iterrows():
        pipe_type = str(row['선정관경']).strip()
        if pipe_type not in pipe_data: pipe_type = '400P' 
            
        p_info = pipe_data.get(pipe_type)
        eq_length = (float(row['볼밸브(개)']) * p_info['ball']) + \
                    (float(row['90도엘보(개)']) * p_info['el90']) + \
                    (float(row['45도엘보(개)']) * p_info['el45']) + \
                    (float(row['동경티(개)']) * p_info['tee']) + \
                    (float(row['1/4축소티(개)']) * p_info['tee14']) + \
                    (float(row['1/2레듀샤(개)']) * p_info['red12'])
        
        edited_df.at[idx, '관상당합계'] = eq_length
        edited_df.at[idx, '관길이(m)'] = float(row['직관길이(m)']) + eq_length

    grand_total_length = edited_df['관길이(m)'].sum()

    for idx, row in edited_df.iterrows():
        pipe_type = str(row['선정관경']).strip()
        if pipe_type not in pipe_data: pipe_type = '400P'
            
        p_info = pipe_data.get(pipe_type)
        inner_d = p_info['inner_d']
        
        세대수 = int(row['세대수(세대)'])
        sim_rate = get_sim_rate(세대수)
        q_calc = 세대수 * sim_rate * household_flow
        관길이 = row['관길이(m)']
        
        p_drop = 0.01222 * (관길이 * (q_calc ** 2)) / (inner_d ** 5) if inner_d > 0 else 0
        total_actual_drop += p_drop
        allowable_drop = STANDARD_PRESSURE * (관길이 / grand_total_length) if grand_total_length > 0 else 0
        
        result_data.append({
            "구간": row['구간'],
            "선정관경": pipe_type,
            "세대수(세대)": 세대수,
            "동시사용률": sim_rate,
            "직관길이(m)": round(row['직관길이(m)'], 2),
            "관상당합계": round(row['관상당합계'], 2),
            "관길이(m)": round(관길이, 2),
            "유량(㎥/hr)": round(q_calc, 2),
            "실_압력손실(kPa)": round(p_drop, 4),
            "구간_허용압력(kPa)": round(allowable_drop, 4)
        })

result_df = pd.DataFrame(result_data)

st.markdown("---")

# ==========================================
# 3. 최종 결과 표
# ==========================================
status_msg = ""
diagnosis_msg = ""

if total_actual_drop == 0:
    status_msg = "데이터 미입력"
elif total_actual_drop <= STANDARD_PRESSURE:
    status_msg = f"✅ 적합 (총 압력손실 {total_actual_drop:.4f} kPa - 기준치 이내)"
else:
    status_msg = f"🚨 부적합 (총 압력손실 {total_actual_drop:.4f} kPa - 기준치 초과)"
    if not result_df.empty:
        worst_idx = result_df['실_압력손실(kPa)'].idxmax()
        worst_section = result_df.loc[worst_idx, '구간']
        worst_drop = result_df.loc[worst_idx, '실_압력손실(kPa)']
        diagnosis_msg = f"⚠️ [{worst_section}] 구간에서 압력손실({worst_drop:.4f} kPa)이 가장 큽니다."

def convert_df_to_excel(df, status, diagnosis):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='관경산출결과')
        worksheet = writer.sheets['관경산출결과']
        last_row = len(df) + 2
        worksheet.cell(row=last_row + 1, column=1, value="■ 최종 판정 결과")
        worksheet.cell(row=last_row + 2, column=1, value=status)
        if diagnosis:
            worksheet.cell(row=last_row + 3, column=1, value="■ 진단 코멘트")
            worksheet.cell(row=last_row + 4, column=1, value=diagnosis)
    return output.getvalue()

col_title, col_download = st.columns([8, 2])
with col_title: st.markdown("### 3️⃣ 최종 관경산출표")
with col_download:
    if not result_df.empty:
        st.download_button("📥 엑셀 다운로드", data=convert_df_to_excel(result_df, status_msg, diagnosis_msg), file_name="최종관경산출.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

st.dataframe(
    result_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "세대수(세대)": st.column_config.NumberColumn("세대수(세대)", format="%,d"),
        "유량(㎥/hr)": st.column_config.NumberColumn("유량(㎥/hr)", format="%,.2f"),
        "동시사용률": st.column_config.NumberColumn("동시사용률", format="%.2f"),
        "직관길이(m)": st.column_config.NumberColumn("직관길이(m)", format="%,.2f"),
        "관상당합계": st.column_config.NumberColumn("관상당합계", format="%,.2f"),
        "관길이(m)": st.column_config.NumberColumn("관길이(m)", format="%,.2f"),
        "실_압력손실(kPa)": st.column_config.NumberColumn("실_압력손실(kPa)", format="%.4f"),
        "구간_허용압력(kPa)": st.column_config.NumberColumn("구간 허용압력(kPa)", format="%.4f"),
    }
)

st.markdown("#### 🎯 최종 판정 결과")
col1, col2, col3 = st.columns([1, 1, 2])
with col1: st.metric("실압력 손실 (kPa)", f"{total_actual_drop:.4f}")
with col2: st.metric("허용압력 손실 (kPa)", f"{STANDARD_PRESSURE:.4f}")
with col3:
    if total_actual_drop == 0: st.info("데이터를 입력해 주세요.")
    elif total_actual_drop <= STANDARD_PRESSURE:
        st.markdown("""<div style="background-color:#d4edda; padding:20px; border-radius:10px; text-align:center;"><h2 style="color:#155724; margin:0;">✅ 적 합 (공사 불필요)</h2></div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style="background-color:#f8d7da; padding:20px; border-radius:10px; text-align:center;"><h2 style="color:#721c24; margin:0;">🚨 부 적 합 (관경 확대 요망)</h2></div>""", unsafe_allow_html=True)
        if diagnosis_msg: st.markdown(f"""<div style="margin-top:15px; padding:15px; background-color:#fff3cd; border-left:5px solid #ffc107; color:#856404;"><strong>{diagnosis_msg}</strong></div>""", unsafe_allow_html=True)

# 공사 예상 비용 추산기 (부적합 시 노출)
if total_actual_drop > STANDARD_PRESSURE and not result_df.empty:
    st.markdown("---")
    st.markdown("### 💰 [부록] 총 배관 공사 예상 비용 추산기")
    summary_df = result_df.groupby('선정관경')['관길이(m)'].sum().reset_index()
    cost_df = pd.DataFrame([{"선정관경": r['선정관경'], "총 관길이(m)": round(r['관길이(m)'], 2), "예상단가(원/m)": default_unit_costs.get(r['선정관경'], 100000)} for _, r in summary_df.iterrows()])
    c1, c2 = st.columns([2, 1])
    with c1: edited_cost = st.data_editor(cost_df, hide_index=True, use_container_width=True)
    with c2:
        total_cost = (edited_cost['총 관길이(m)'] * edited_cost['예상단가(원/m)']).sum()
        st.metric("💡 총 배관 공사 예상 비용", f"{int(total_cost):,} 원")
