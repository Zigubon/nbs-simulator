import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import interp1d

# -----------------------------------------------------------
# 1. 환경 설정 및 스타일
# -----------------------------------------------------------
st.set_page_config(page_title="ZIGUBON | Forest Carbon Simulator", page_icon="🌲", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stCard { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    div[data-testid="stMetricValue"] { font-size: 26px; color: #145A32; font-weight: 700; }
    div[data-testid="stMetricLabel"] { font-size: 14px; color: #666; }
    div[data-testid="stExpander"] { border: 1px solid #e0e0e0; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# 2. 데이터 로드
# -----------------------------------------------------------
@st.cache_data
def load_data():
    try:
        # index.html에서 가상 파일 시스템으로 넘겨준 파일 읽기
        forest = pd.read_csv("forest_data_2026.csv")
        price = pd.read_csv("carbon_price_scenarios.csv")
        benefit = pd.read_csv("co_benefits.csv")
        return forest, price, benefit
    except Exception as e:
        return None, None, None

df_forest, df_price, df_benefit = load_data()

if df_forest is None:
    st.error("❌ 데이터를 불러올 수 없습니다. index.html 및 CSV 파일 상태를 확인해주세요.")
    st.stop()

# -----------------------------------------------------------
# 3. 사이드바 UI (CBI 지표 추가)
# -----------------------------------------------------------
with st.sidebar:
    st.title("🌲 시뮬레이션 설정")
    
    # [섹션 1] 기본 개요
    st.subheader("1️⃣ 사업 개요")
    area = st.number_input("사업 면적 (ha)", min_value=1.0, value=50.0, step=1.0)
    project_period = st.slider("사업 기간 (년)", 20, 100, 30)
    
    st.markdown("---")
    
    # [섹션 2] 수종 및 비율
    st.subheader("2️⃣ 수종 포트폴리오 (CBI 지표 4)")
    species_list = df_forest['name'].unique()
    
    # 기본값
    default_sp = [species_list[0], species_list[1]] if len(species_list) > 1 else [species_list[0]]
    selected_species = st.multiselect("식재 수종 선택", species_list, default=default_sp)
    
    if not selected_species:
        st.warning("⚠️ 최소 1개 이상의 수종을 선택해주세요.")
        st.stop()
    
    # 수종별 비율 슬라이더
    species_ratios = {}
    if len(selected_species) > 1:
        st.caption("👇 수종별 점유 비율(%)")
        total_ratio = 0
        for sp in selected_species:
            default_val = int(100 / len(selected_species))
            ratio = st.slider(f"{sp} 비율", 0, 100, default_val, key=f"ratio_{sp}")
            species_ratios[sp] = ratio / 100.0
            total_ratio += ratio
        
        if total_ratio != 100:
            st.error(f"⚠️ 비율 합계가 {total_ratio}%입니다. 100%에 맞춰주세요.")
    else:
        species_ratios[selected_species[0]] = 1.0

    st.markdown("---")

    # [섹션 3] 생태적 연결성 (CBI 지표 2 반영) - 신규 기능
    st.subheader("3️⃣ 생태 네트워크 (CBI 지표 2)")
    connectivity_help = """
    **싱가포르 지수(CBI) 지표 2: 연결성 조치**
    대상지가 주변 산림이나 생태축과 얼마나 잘 연결되어 있는지를 평가합니다.
    - 높음: 백두대간 등 핵심 생태축과 직접 연결
    - 낮음: 도심 속 고립된 숲
    """
    connectivity_score = st.select_slider(
        "주변 생태계 연결성 수준",
        options=["고립 (낮음)", "일부 연결 (보통)", "핵심 축 연결 (높음)"],
        value="일부 연결 (보통)",
        help=connectivity_help
    )
    # 점수 매핑 (1~5점)
    conn_map = {"고립 (낮음)": 1.0, "일부 연결 (보통)": 3.0, "핵심 축 연결 (높음)": 5.0}
    conn_value = conn_map[connectivity_score]

    # [섹션 4] 식재 밀도
    st.markdown("---")
    st.subheader("4️⃣ 식재 밀도 (Density)")
    density_factor = st.slider("식재 밀도 지수 (%)", 50, 150, 100) / 100.0
    estimated_trees = int(area * 3000 * density_factor)
    
    # [섹션 5] 경제성
    st.markdown("---")
    st.subheader("5️⃣ 경제성 시나리오")
    price_scenario = st.selectbox("탄소배출권 가격", ["Base (기준)", "High (낙관)", "Low (보수)"])
    price_col_map = {"Base (기준)": "price_base", "High (낙관)": "price_high", "Low (보수)": "price_low"}
    price_col = price_col_map[price_scenario]

# -----------------------------------------------------------
# 4. CBI 기반 분석 로직 (자생종 판단)
# -----------------------------------------------------------
# 한국 산림 기준 자생종(Native) vs 도입종(Exotic/Plantation) 구분 로직
# (실제로는 DB에 있어야 하지만, 편의상 이름으로 매핑)
def check_native(name):
    # 자생종 키워드
    native_keywords = ["소나무", "상수리", "신갈", "졸참", "굴참", "잣나무"] 
    # 도입종 키워드 (낙엽송-일본잎갈나무, 편백-일본원산, 리기다-북미원산, 백합-북미원산)
    if any(k in name for k in native_keywords):
        return True
    return False

# -----------------------------------------------------------
# 5. 통합 계산 엔진
# -----------------------------------------------------------
years = list(range(2026, 2026 + project_period + 1))

total_biomass_carbon = np.zeros(project_period + 1)
total_soil_carbon = np.zeros(project_period + 1)

# CBI 점수 계산 변수
total_native_ratio = 0
weighted_water_score = 0
weighted_fire_score = 0

for sp in selected_species:
    sp_row = df_forest[df_forest['name'] == sp].iloc[0]
    ratio = species_ratios[sp]
    
    # 1. 탄소 계산
    x_points = list(range(0, 51, 5))
    y_points = [sp_row[f'co2_yr_{y}'] for y in x_points]
    f_interp = interp1d(x_points, y_points, kind='linear', fill_value="extrapolate")
    standard_uptake = f_interp(range(project_period + 1))
    
    real_area = area * ratio
    adjusted_uptake = standard_uptake * real_area * density_factor
    soil_uptake = adjusted_uptake * 0.35 # 토양탄소
    
    total_biomass_carbon += adjusted_uptake
    total_soil_carbon += soil_uptake
    
    # 2. CBI 지표 계산 (가중 평균)
    # (1) 자생종 비율 (Indicator 4)
    is_native = check_native(sp)
    if is_native:
        total_native_ratio += ratio * 100 # 자생종이면 해당 비율만큼 점수 추가
        
    # (2) 수자원 및 재해방지 (Indicator 10)
    # co_benefits 데이터 매핑 (이름으로 찾기)
    try:
        # id 매핑 로직이 복잡하므로 순서 기반 가정 or 이름 매핑 시도
        # 여기선 간단히 id가 1,2,3... 순서대로라고 가정하고 인덱싱 (위험하지만 현재 데이터셋 기준)
        # 더 안전한 방법: co_benefits.csv에 name 컬럼이 없으므로 id 매핑 필요.
        # *사용자 데이터 특성상 id 1=상수리, 2=신갈... 순서 일치 가정*
        ben_row = df_benefit.iloc[sp_row['id']-1] # id는 1부터 시작하므로 -1
        weighted_water_score += ben_row['water_index'] * ratio
        weighted_fire_score += ben_row['fire_resistance'] * ratio
    except:
        weighted_water_score += 3.0 * ratio # 기본값

total_project_carbon = total_biomass_carbon + total_soil_carbon
baseline_carbon = total_project_carbon * 0.7
net_credit = total_project_carbon - baseline_carbon

# -----------------------------------------------------------
# 6. 결과 대시보드
# -----------------------------------------------------------
final_carbon = total_project_carbon[-1]

target_year = 2026 + project_period
if target_year > df_price['year'].max():
    unit_price = df_price.iloc[-1][price_col]
else:
    unit_price = df_price[df_price['year'] == target_year][price_col].values[0]
final_value = final_carbon * unit_price
cars_offset = (final_carbon / project_period) / 2.43

# [CBI 종합 점수 산출]
# 1. 자생종 점수 (0~5점): 자생종 비율이 높을수록 5점에 수렴
cbi_native_score = (total_native_ratio / 100.0) * 5.0

# 2. 연결성 점수 (입력값 그대로 사용)
cbi_conn_score = conn_value

# 3. 수자원 점수 (가중평균)
cbi_water_score = weighted_water_score

# 4. 혼효림 보너스 (종 다양성)
cbi_diversity_score = min(5.0, 2.0 + (len(selected_species) * 0.6))

# 종합 평균
final_esg_score = (cbi_native_score + cbi_conn_score + cbi_water_score + cbi_diversity_score) / 4.0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("총 탄소 순흡수량", f"{final_carbon:,.0f} tCO₂", f"연평균 {final_carbon/project_period:,.0f}톤")
with col2:
    st.metric("예상 경제적 가치", f"₩{final_value/100000000:.1f} 억", f"{price_scenario} 시나리오")
with col3:
    st.metric("승용차 배출 상쇄", f"{cars_offset:,.0f} 대/년", "연 2.43tCO₂ 기준")
with col4:
    st.metric("CBI 기반 생물다양성", f"{final_esg_score:.1f} / 5.0", "싱가포르 지수 적용")

st.markdown("---")

# -----------------------------------------------------------
# 7. 시각화 (CBI 레이더 차트 적용)
# -----------------------------------------------------------
c_main, c_sub = st.columns([2, 1])

with c_main:
    st.subheader("📊 탄소 저장고 및 추가성 분석")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=total_biomass_carbon, mode='lines', name='🌲 입목 바이오매스', stackgroup='one', line=dict(width=0, color='#27ae60')))
    fig.add_trace(go.Scatter(x=years, y=total_soil_carbon, mode='lines', name='🟤 토양/기타 저장고', stackgroup='one', line=dict(width=0, color='#8d6e63')))
    fig.add_trace(go.Scatter(x=years, y=baseline_carbon, mode='lines', name='📉 베이스라인 (무관리)', line=dict(color='#7f8c8d', width=2, dash='dash')))
    fig.update_layout(xaxis_title="연도", yaxis_title="누적 tCO₂", height=400, hovermode="x unified", legend=dict(orientation="h", y=1.02, x=1))
    st.plotly_chart(fig, use_container_width=True)

with c_sub:
    st.subheader("🕸️ CBI 생태 가치 평가")
    
    categories = ['자생종 비율 (Ind.4)', '수자원 조절 (Ind.10)', '생태 연결성 (Ind.2)', '종 다양성', '경제적 가치']
    # 경제성 점수 (상대평가)
    econ_score = min(5.0, final_value / 1000000000 * 2) 
    
    r_values = [cbi_native_score, cbi_water_score, cbi_conn_score, cbi_diversity_score, econ_score]
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=r_values, theta=categories, fill='toself', name='Project Score',
        line=dict(color='#145A32')
    ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), showlegend=False, height=350, margin=dict(l=40, r=40, t=20, b=20))
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # CBI 해석 캡션
    st.info(f"""
    **💡 CBI(싱가포르 지수) 분석 결과**
    - **자생종 비율:** {total_native_ratio:.0f}% (소나무, 참나무류 등 고유 수종 비중)
    - **연결성:** '{connectivity_score}' 수준으로 평가됨
    """)

# -----------------------------------------------------------
# 8. 데이터 다운로드
# -----------------------------------------------------------
with st.expander("📥 상세 리포트 다운로드"):
    df_res = pd.DataFrame({
        "Year": years, "Total_Carbon": total_project_carbon, "Biomass": total_biomass_carbon, 
        "Soil": total_soil_carbon, "Baseline": baseline_carbon, "Net_Credit": net_credit
    })
    st.dataframe(df_res, use_container_width=True)
    st.download_button("CSV 다운로드", df_res.to_csv(index=False).encode('utf-8-sig'), "cbi_simulation_report.csv")
