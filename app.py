import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import interp1d

# ==============================================================================
# 1. 시스템 설정 및 디자인
# ==============================================================================
st.set_page_config(
    page_title="ZIGUBON | Forest Carbon & ESG Simulator",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; font-family: 'Pretendard', sans-serif; }
    div[data-testid="stMetricValue"] { font-size: 26px; color: #145A32; font-weight: 800; }
    div[data-testid="stMetricLabel"] { font-size: 14px; color: #555; font-weight: 600; }
    div[data-testid="stCard"] { background-color: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.04); padding: 1rem; }
    .stExpander { border: 1px solid #e0e0e0; border-radius: 8px; background: white; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. 데이터 로드
# ==============================================================================
@st.cache_data
def load_data():
    try:
        forest = pd.read_csv("forest_data_2026.csv")
        price = pd.read_csv("carbon_price_scenarios.csv")
        benefit = pd.read_csv("co_benefits.csv")
        return forest, price, benefit
    except Exception as e:
        return None, None, None

df_forest, df_price, df_benefit = load_data()

if df_forest is None:
    st.error("🚨 데이터 파일을 찾을 수 없습니다. (forest_data_2026.csv 등 확인 필요)")
    st.stop()

# ==============================================================================
# 3. 사이드바 (입력 제어 패널)
# ==============================================================================
with st.sidebar:
    st.title("🌲 시뮬레이션 설정")
    st.markdown("---")
    
    # [섹션 1] 기본 사업 개요
    st.subheader("1️⃣ 사업 개요")
    area = st.number_input("사업 면적 (ha)", min_value=1.0, value=50.0, step=1.0)
    project_period = st.slider("사업 기간 (년)", 5, 50, 30)
    
    st.markdown("---")
    
    # [섹션 2] 수종 및 포트폴리오
    st.subheader("2️⃣ 수종 포트폴리오")
    species_list = df_forest['name'].unique()
    default_sp = [species_list[0], species_list[1]] if len(species_list) > 1 else [species_list[0]]
    selected_species = st.multiselect("식재 수종 선택", species_list, default=default_sp)
    
    if not selected_species:
        st.warning("⚠️ 최소 1개 이상의 수종을 선택해주세요.")
        st.stop()
    
    # 수종별 점유 비율 설정
    species_ratios = {}
    if len(selected_species) > 1:
        st.info("👇 수종별 점유 비율(%)을 설정하세요")
        total_ratio = 0
        for sp in selected_species:
            default_val = int(100 / len(selected_species))
            ratio = st.slider(f"{sp} 비율", 0, 100, default_val, key=f"ratio_{sp}")
            species_ratios[sp] = ratio / 100.0
            total_ratio += ratio
        
        if total_ratio != 100:
            st.error(f"⚠️ 비율 합계: {total_ratio}% (100%에 맞춰주세요)")
    else:
        species_ratios[selected_species[0]] = 1.0

    st.markdown("---")

    # [섹션 3] 생태 연결성 & 밀도
    st.subheader("3️⃣ 생태 및 기술 요소")
    
    # CBI 지표 2번
    connectivity_score = st.select_slider(
        "생태 연결성 (Connectivity)",
        options=["고립 (낮음)", "일부 연결 (보통)", "핵심 축 연결 (높음)"],
        value="일부 연결 (보통)"
    )
    conn_map = {"고립 (낮음)": 1.0, "일부 연결 (보통)": 3.0, "핵심 축 연결 (높음)": 5.0}
    conn_value = conn_map[connectivity_score]
    
    # 식재 밀도
    density_factor = st.slider("식재 밀도 지수 (%)", 50, 150, 100, help="표준(3,000본/ha) 대비 식재 밀도") / 100.0
    estimated_trees = int(area * 3000 * density_factor)
    st.caption(f"🌱 추정 식재 본수: {estimated_trees:,} 본")

    st.markdown("---")
    
    # [섹션 4] 재무 및 리스크
    st.subheader("4️⃣ 재무 및 리스크")
    
    buffer_ratio = st.slider("리스크 버퍼 (Buffer %)", 0, 30, 15, help="영구 손실 대비 유보율") / 100.0
    
    c1, c2 = st.columns(2)
    with c1:
        initial_cost_per_ha = st.number_input("초기 조성비 (만원/ha)", value=1500, step=100)
    with c2:
        annual_cost_per_ha = st.number_input("연 관리비 (만원/ha)", value=50, step=10)
    
    discount_rate = 0.045 # 할인율 4.5%

    st.markdown("---")

    # [섹션 5] 탄소 가격
    st.subheader("5️⃣ 시장 전망")
    price_scenario = st.selectbox("탄소배출권 가격 전망", ["Base (기준)", "High (낙관)", "Low (보수)"])
    price_col_map = {"Base (기준)": "price_base", "High (낙관)": "price_high", "Low (보수)": "price_low"}
    price_col = price_col_map[price_scenario]


# ==============================================================================
# 4. 시뮬레이션 계산 엔진
# ==============================================================================

def check_native(name):
    # 자생종 키워드
    native_keywords = ["소나무", "상수리", "신갈", "졸참", "굴참", "잣나무", "느티나무"] 
    return any(k in name for k in native_keywords)

years = list(range(2026, 2026 + project_period + 1))

total_biomass_carbon = np.zeros(project_period + 1)
total_soil_carbon = np.zeros(project_period + 1)

total_native_ratio = 0
weighted_water_score = 0

# --- Core Loop ---
for sp in selected_species:
    sp_row = df_forest[df_forest['name'] == sp].iloc[0]
    ratio = species_ratios[sp]
    
    # Interpolation
    x_points = list(range(0, 51, 5))
    y_points = [sp_row[f'co2_yr_{y}'] for y in x_points]
    f_interp = interp1d(x_points, y_points, kind='linear', fill_value="extrapolate")
    standard_uptake_per_ha = f_interp(range(project_period + 1))
    
    # Scaling
    real_area = area * ratio
    adjusted_uptake = standard_uptake_per_ha * real_area * density_factor
    
    # Soil Carbon (Tier 1: 35% of Biomass)
    soil_uptake = adjusted_uptake * 0.35
    
    total_biomass_carbon += adjusted_uptake
    total_soil_carbon += soil_uptake
    
    # CBI Weights
    if check_native(sp):
        total_native_ratio += ratio * 100
    
    try:
        ben_row = df_benefit.iloc[sp_row['id']-1]
        weighted_water_score += ben_row['water_index'] * ratio
    except:
        weighted_water_score += 3.0 * ratio

# --- Net Credit ---
total_project_carbon = total_biomass_carbon + total_soil_carbon
baseline_carbon = total_project_carbon * 0.7 
gross_credit = total_project_carbon - baseline_carbon 
buffer_amount = gross_credit * buffer_ratio
net_issuable_credit = gross_credit - buffer_amount

# --- Financials (ROI & NPV) ---
total_initial_cost = initial_cost_per_ha * area * 10000 
annual_cost_year = annual_cost_per_ha * area * 10000
total_cost_nominal = total_initial_cost + (annual_cost_year * project_period)

revenue_stream = []
net_cash_flow = []
net_cash_flow.append(-total_initial_cost) # Year 0

for i, yr in enumerate(years):
    if i == 0: continue
    
    annual_credit = net_issuable_credit[i] - net_issuable_credit[i-1]
    
    if yr > df_price['year'].max():
        curr_price = df_price.iloc[-1][price_col]
    else:
        curr_price = df_price[df_price['year'] == yr][price_col].values[0]
        
    rev = annual_credit * curr_price
    revenue_stream.append(rev)
    
    net_flow = rev - annual_cost_year
    net_cash_flow.append(net_flow)

total_revenue_nominal = sum(revenue_stream)
net_profit_nominal = total_revenue_nominal - total_cost_nominal

# ROI
roi = (net_profit_nominal / total_cost_nominal) * 100 if total_cost_nominal > 0 else 0

# NPV
npv = -total_initial_cost
for t, flow in enumerate(net_cash_flow[1:], start=1):
    npv += flow / ((1 + discount_rate) ** t)

# --- CBI Score ---
cbi_native_score = (total_native_ratio / 100.0) * 5.0
cbi_water_score = weighted_water_score
cbi_conn_score = conn_value
cbi_diversity_score = min(5.0, 2.0 + (len(selected_species) * 0.6))

if roi <= 0: cbi_econ_score = 1.0
elif roi >= 200: cbi_econ_score = 5.0
else: cbi_econ_score = 1.0 + (roi / 50.0)

final_cbi_score = (cbi_native_score + cbi_water_score + cbi_conn_score + cbi_diversity_score + cbi_econ_score) / 5.0


# ==============================================================================
# 5. 메인 대시보드 UI
# ==============================================================================
forest_type = "혼효림 (Mixed Forest)" if len(selected_species) > 1 else "단순림 (Monoculture)"
st.title(f"🌲 {forest_type} 사업성 분석 시뮬레이터")
st.markdown(f"**{area}ha** 면적 / **{project_period}년** 사업 / **{', '.join(selected_species)}** 식재 시나리오 ")

# KPI Cards
col1, col2, col3, col4 = st.columns(4)
final_credit = net_issuable_credit[-1]

with col1:
    st.metric("순 발행 크레딧", f"{final_credit:,.0f} tCO₂", f"버퍼 {int(buffer_ratio*100)}% 차감")
with col2:
    st.metric("예상 순수익", f"₩{net_profit_nominal/100000000:.1f} 억", f"ROI {roi:.1f}%")
with col3:
    st.metric("순현재가치 (NPV)", f"₩{npv/100000000:.1f} 억", f"할인율 {discount_rate*100}% ")
with col4:
    st.metric("CBI 종합 등급", f"{final_cbi_score:.1f} / 5.0", f"싱가포르 지수 기반 ")

st.markdown("---")

# Charts
c_main, c_sub = st.columns([2, 1])

with c_main:
    st.subheader("📊 탄소 저장 및 추가성 분석")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=total_biomass_carbon, mode='lines', name='🌲 입목 바이오매스', stackgroup='one', line=dict(width=0, color='#27ae60')))
    fig.add_trace(go.Scatter(x=years, y=total_soil_carbon, mode='lines', name='🟤 토양/기타 저장고', stackgroup='one', line=dict(width=0, color='#8d6e63')))
    fig.add_trace(go.Scatter(x=years, y=baseline_carbon, mode='lines', name='📉 베이스라인 (무관리)', line=dict(color='#34495e', width=2, dash='dash')))
    
    fig.update_layout(
        xaxis_title="연도", yaxis_title="누적 흡수량 (tCO₂)",
        height=400, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        margin=dict(t=30)
    )
    st.plotly_chart(fig, use_container_width=True)

with c_sub:
    st.subheader("🕸️ CBI 가치 평가")
    categories = ['자생종(Native)', '수자원(Water)', '연결성(Conn.)', '다양성(Div.)', '수익성(ROI)']
    r_values = [cbi_native_score, cbi_water_score, cbi_conn_score, cbi_diversity_score, cbi_econ_score]
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=r_values, theta=categories, fill='toself', name='Score',
        line=dict(color='#145A32')
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=False, height=350,
        margin=dict(l=30, r=30, t=20, b=20)
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    
    with st.expander("💡 CBI 점수 상세"):
        st.write(f"- **자생종:** {total_native_ratio:.0f}%")
        st.write(f"- **수익성:** ROI {roi:.1f}%")
        st.write(f"- **연결성:** {connectivity_score}")

# Data Download
with st.expander("📥 상세 데이터 테이블 다운로드"):
    df_res = pd.DataFrame({
        "Year": years,
        "Total_Carbon": total_project_carbon,
        "Baseline": baseline_carbon,
        "Net_Credit": net_issuable_credit,
        "Cumulative_Cash_Flow": np.cumsum(net_cash_flow[1:])
    })
    st.dataframe(df_res, use_container_width=True)
    st.download_button("CSV 다운로드", df_res.to_csv(index=False).encode('utf-8-sig'), "simulation_report.csv")
