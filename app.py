import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import interp1d

# ==============================================================================
# 1. 시스템 설정
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
    div[data-testid="stMetricValue"] { font-size: 24px; color: #145A32; font-weight: 800; }
    div[data-testid="stMetricLabel"] { font-size: 14px; color: #555; font-weight: 600; }
    div[data-testid="stCard"] { background-color: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.04); padding: 1rem; }
    .stExpander { border: 1px solid #e0e0e0; border-radius: 8px; background: white; }
    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #fff; border-radius: 4px 4px 0 0; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #e8f5e9; color: #145A32; font-weight: bold; }
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
    st.error("🚨 데이터 파일을 찾을 수 없습니다.")
    st.stop()

# ==============================================================================
# 3. 사이드바 (입력 제어)
# ==============================================================================
with st.sidebar:
    st.title("🌲 시뮬레이션 설정")
    st.markdown("---")
    
    # 1. 사업 개요
    st.subheader("1️⃣ 사업 개요")
    area = st.number_input("사업 면적 (ha)", min_value=1.0, value=50.0, step=1.0)
    project_period = st.slider("사업 기간 (년)", 5, 50, 30)
    
    st.markdown("---")
    
    # 2. 수종 및 비율
    st.subheader("2️⃣ 수종 포트폴리오")
    species_list = df_forest['name'].unique()
    default_sp = [species_list[0], species_list[1]] if len(species_list) > 1 else [species_list[0]]
    selected_species = st.multiselect("식재 수종 선택", species_list, default=default_sp)
    
    if not selected_species:
        st.stop()
    
    species_ratios = {}
    if len(selected_species) > 1:
        st.info("👇 수종별 점유 비율(%)")
        total_ratio = 0
        for sp in selected_species:
            default_val = int(100 / len(selected_species))
            ratio = st.slider(f"{sp} 비율", 0, 100, default_val, key=f"ratio_{sp}")
            species_ratios[sp] = ratio / 100.0
            total_ratio += ratio
    else:
        species_ratios[selected_species[0]] = 1.0

    st.markdown("---")

    # 3. 기술 요소
    st.subheader("3️⃣ 생태 및 기술")
    connectivity_score = st.select_slider("생태 연결성", ["낮음", "보통", "높음"], value="보통")
    conn_map = {"낮음": 1.0, "보통": 3.0, "높음": 5.0}
    conn_value = conn_map[connectivity_score]
    
    density_factor = st.slider("식재 밀도 지수 (%)", 50, 150, 100) / 100.0
    estimated_trees = int(area * 3000 * density_factor)
    st.caption(f"🌱 추정 식재 본수: {estimated_trees:,} 본")

    st.markdown("---")
    
    # 4. 재무 및 리스크
    st.subheader("4️⃣ 재무 설계")
    subsidy_rate = st.slider("🏛️ 정부 보조금 지원율 (%)", 0, 100, 90) / 100.0
    other_revenue_per_ha = st.number_input("💰 부가 수익 (만원/ha/년)", value=20, step=10)
    
    c1, c2 = st.columns(2)
    with c1:
        initial_cost_per_ha = st.number_input("초기 조성비 (만원/ha)", value=1500, step=100)
    with c2:
        annual_cost_per_ha = st.number_input("연 관리비 (만원/ha)", value=50, step=10)
    
    discount_rate = 0.045
    buffer_ratio = 0.15

    st.markdown("---")

    # 5. 탄소 가격
    st.subheader("5️⃣ 시장 전망")
    price_scenario = st.selectbox("가격 전망", ["Base (기준)", "High (낙관)", "Low (보수)"])
    price_col_map = {"Base (기준)": "price_base", "High (낙관)": "price_high", "Low (보수)": "price_low"}
    price_col = price_col_map[price_scenario]


# ==============================================================================
# 4. 계산 엔진
# ==============================================================================
def check_native(name):
    native_keywords = ["소나무", "상수리", "신갈", "졸참", "굴참", "잣나무", "느티나무"] 
    return any(k in name for k in native_keywords)

years = list(range(2026, 2026 + project_period + 1))

total_biomass_carbon = np.zeros(project_period + 1)
total_soil_carbon = np.zeros(project_period + 1)
species_data = {} # [복구] 수종별 시계열 데이터 저장용

total_native_ratio = 0
weighted_water_score = 0

# --- Physical Engine ---
for sp in selected_species:
    sp_row = df_forest[df_forest['name'] == sp].iloc[0]
    ratio = species_ratios[sp]
    
    # Interpolation
    x_points = list(range(0, 51, 5))
    y_points = [sp_row[f'co2_yr_{y}'] for y in x_points]
    f_interp = interp1d(x_points, y_points, kind='linear', fill_value="extrapolate")
    standard_uptake = f_interp(range(project_period + 1))
    
    # Scaling
    real_area = area * ratio
    adjusted_uptake = standard_uptake * real_area * density_factor
    soil_uptake = adjusted_uptake * 0.35
    
    # 합산 및 개별 저장
    total_biomass_carbon += adjusted_uptake
    total_soil_carbon += soil_uptake
    
    # [복구] 수종별 총 흡수량 (입목+토양) 저장
    species_data[sp] = adjusted_uptake + soil_uptake
    
    if check_native(sp): total_native_ratio += ratio * 100
    try:
        ben_row = df_benefit.iloc[sp_row['id']-1]
        weighted_water_score += ben_row['water_index'] * ratio
    except:
        weighted_water_score += 3.0 * ratio

# Net Credit
total_project_carbon = total_biomass_carbon + total_soil_carbon
baseline_carbon = total_project_carbon * 0.7 
gross_credit = total_project_carbon - baseline_carbon 
buffer_amount = gross_credit * buffer_ratio
net_issuable_credit = gross_credit - buffer_amount

# --- Financial Engine ---
real_initial_cost = (initial_cost_per_ha * area * 10000) * (1 - subsidy_rate)
annual_cost_total = annual_cost_per_ha * area * 10000
total_cost_real = real_initial_cost + (annual_cost_total * project_period)

other_revenue_total = other_revenue_per_ha * area * 10000 
revenue_stream = []
net_cash_flow = []
net_cash_flow.append(-real_initial_cost) 
cumulative_profit = [-real_initial_cost] 

for i, yr in enumerate(years):
    if i == 0: continue
    
    annual_credit = net_issuable_credit[i] - net_issuable_credit[i-1]
    if yr > df_price['year'].max(): curr_price = df_price.iloc[-1][price_col]
    else: curr_price = df_price[df_price['year'] == yr][price_col].values[0]
    
    total_annual_rev = (annual_credit * curr_price) + other_revenue_total
    revenue_stream.append(total_annual_rev)
    
    net_flow = total_annual_rev - annual_cost_total
    net_cash_flow.append(net_flow)
    cumulative_profit.append(cumulative_profit[-1] + net_flow)

total_revenue_real = sum(revenue_stream)
net_profit_real = total_revenue_real - total_cost_real
roi = (net_profit_real / total_cost_real) * 100 if total_cost_real > 0 else 0

npv = -real_initial_cost
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
# 5. 대시보드 UI (Tabs 도입)
# ==============================================================================
forest_type = "혼효림" if len(selected_species) > 1 else "단순림"
st.title(f"🌲 {forest_type} 통합 분석 시뮬레이터")
st.markdown(f"**{area}ha** / **{project_period}년** / **{len(selected_species)}개 수종** 복합 분석")

# Top Metrics
col1, col2, col3, col4 = st.columns(4)
final_credit = net_issuable_credit[-1]

with col1:
    st.metric("총 순 흡수량 (Net)", f"{final_credit:,.0f} tCO₂", "버퍼 차감 후")
with col2:
    color = "normal" if net_profit_real >= 0 else "inverse"
    st.metric("최종 순수익", f"₩{net_profit_real/100000000:.1f} 억", f"ROI {roi:.1f}%", delta_color=color)
with col3:
    st.metric("순현재가치 (NPV)", f"₩{npv/100000000:.1f} 억", f"할인율 4.5%")
with col4:
    st.metric("CBI 등급", f"{final_cbi_score:.1f} / 5.0", "ESG 종합")

st.markdown("---")

# ------------------------------------------------------------------------------
# 탭 구성: 사용자 요청대로 탄소 그래프와 재무 그래프를 모두 살림
# ------------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 탄소·생태 분석", "💰 재무·수익성 분석", "🕸️ ESG·CBI 평가"])

# [Tab 1] 탄소 & 물리적 생장 (복구됨)
with tab1:
    c_t1, c_t2 = st.columns(2)
    
    with c_t1:
        st.subheader("🌲 총 탄소 흡수 및 베이스라인")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=years, y=total_biomass_carbon, mode='lines', name='입목 바이오매스', stackgroup='one', line=dict(width=0, color='#27ae60')))
        fig1.add_trace(go.Scatter(x=years, y=total_soil_carbon, mode='lines', name='토양/기타 저장고', stackgroup='one', line=dict(width=0, color='#8d6e63')))
        fig1.add_trace(go.Scatter(x=years, y=baseline_carbon, mode='lines', name='베이스라인 (무관리)', line=dict(color='#34495e', width=2, dash='dash')))
        fig1.update_layout(xaxis_title="연도", yaxis_title="tCO₂", height=350, hovermode="x unified", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig1, use_container_width=True)
        
    with c_t2:
        st.subheader("🌿 수종별 흡수량 기여도 (복구됨)")
        fig2 = go.Figure()
        # 수종별 누적 그래프 그리기
        for sp, data in species_data.items():
            fig2.add_trace(go.Scatter(x=years, y=data, mode='lines', name=sp, stackgroup='one'))
        fig2.update_layout(xaxis_title="연도", yaxis_title="tCO₂ (수종별)", height=350, hovermode="x unified", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig2, use_container_width=True)

# [Tab 2] 재무 & 수익성 (J-Curve)
with tab2:
    st.subheader("💰 누적 현금 흐름 (J-Curve Analysis)")
    fig3 = go.Figure()
    fig3.add_hline(y=0, line_dash="dot", line_color="gray")
    fig3.add_trace(go.Scatter(
        x=list(range(0, project_period + 1)), 
        y=cumulative_profit,
        mode='lines', 
        name='누적 순수익',
        fill='tozeroy',
        line=dict(color='#2ecc71' if net_profit_real > 0 else '#e74c3c', width=3)
    ))
    fig3.update_layout(xaxis_title="사업 연차", yaxis_title="누적 금액 (원)", height=400, hovermode="x unified")
    st.plotly_chart(fig3, use_container_width=True)
    
    # 상세 재무표
    with st.expander("💸 상세 비용 및 수익 내역 보기"):
        st.write(f"- **총 비용:** ₩{total_cost_real:,.0f} (초기: ₩{real_initial_cost:,.0f} / 관리: ₩{annual_cost_total:,.0f})")
        st.write(f"- **총 매출:** ₩{total_revenue_real:,.0f} (탄소 판매 + 기타 수익)")
        st.write(f"- **보조금 효과:** 초기 비용의 {int(subsidy_rate*100)}% 절감")

# [Tab 3] ESG & CBI
with tab3:
    c_e1, c_e2 = st.columns([1, 1])
    with c_e1:
        st.subheader("🕸️ CBI 가치 평가 (Radar)")
        categories = ['자생종', '수자원', '연결성', '다양성', '수익성(ROI)']
        r_values = [cbi_native_score, cbi_water_score, cbi_conn_score, cbi_diversity_score, cbi_econ_score]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=r_values, theta=categories, fill='toself', name='Score', line=dict(color='#145A32')))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), showlegend=False, height=350)
        st.plotly_chart(fig_radar, use_container_width=True)
        
    with c_e2:
        st.subheader("📋 CBI 상세 점수표")
        st.info(f"""
        - **자생종 비율:** {total_native_ratio:.0f}% ({cbi_native_score:.1f}점)
        - **경제성(ROI):** {roi:.1f}% ({cbi_econ_score:.1f}점)
        - **생태 연결성:** {connectivity_score} ({cbi_conn_score:.1f}점)
        - **종 다양성:** {len(selected_species)}종 혼합 ({cbi_diversity_score:.1f}점)
        """)

# Data Download (Footer Area)
with st.expander("📥 전체 시뮬레이션 데이터 다운로드"):
    df_res = pd.DataFrame({
        "Year": list(range(0, project_period + 1)),
        "Total_Carbon": [0] + list(total_project_carbon[1:]),
        "Net_Credit": [0] + list(net_issuable_credit[1:]),
        "Cumulative_Profit": cumulative_profit
    })
    st.dataframe(df_res, use_container_width=True)
    st.download_button("CSV 다운로드", df_res.to_csv(index=False).encode('utf-8-sig'), "full_simulation_report.csv")
