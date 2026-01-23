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
# 3. 사이드바 UI (입력 제어 통합)
# -----------------------------------------------------------
with st.sidebar:
    st.title("🌲 시뮬레이션 설정")
    
    # [섹션 1] 기본 개요
    st.subheader("1️⃣ 사업 개요")
    area = st.number_input("사업 면적 (ha)", min_value=1.0, value=50.0, step=1.0)
    # [수정] 사업 기간 5~50년으로 제한
    project_period = st.slider("사업 기간 (년)", 5, 50, 30)
    
    st.markdown("---")
    
    # [섹션 2] 수종 및 비율
    st.subheader("2️⃣ 수종 및 구성비")
    species_list = df_forest['name'].unique()
    default_sp = [species_list[0], species_list[1]] if len(species_list) > 1 else [species_list[0]]
    selected_species = st.multiselect("식재 수종 선택", species_list, default=default_sp)
    
    if not selected_species:
        st.warning("⚠️ 최소 1개 이상의 수종을 선택해주세요.")
        st.stop()
    
    species_ratios = {}
    if len(selected_species) > 1:
        st.caption("👇 수종별 점유 비율(%) 설정")
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
    
    # [섹션 3] 식재 밀도
    st.subheader("3️⃣ 식재 밀도 (Density)")
    density_factor = st.slider("식재 밀도 지수 (%)", 50, 150, 100) / 100.0
    estimated_trees = int(area * 3000 * density_factor)
    st.caption(f"🌲 총 추정 식재 본수: **{estimated_trees:,} 그루**")

    st.markdown("---")
    
    # [섹션 4] 비용 및 리스크 관리 (신규 기능)
    st.subheader("4️⃣ 비용 및 리스크 (Financial)")
    
    # 버퍼 비율 (리스크 관리용 차감)
    buffer_ratio = st.slider("버퍼 비율 (Buffer Ratio, %)", 0, 30, 10, help="산불, 병해충 등 영구적 손실 대비를 위해 적립(차감)하는 크레딧 비율입니다.") / 100.0
    
    # 비용 입력
    col_cost1, col_cost2 = st.columns(2)
    with col_cost1:
        initial_cost_per_ha = st.number_input("초기 조성비 (만원/ha)", value=1500, step=100, help="묘목비, 식재비 등")
    with col_cost2:
        annual_cost_per_ha = st.number_input("연 관리비 (만원/ha)", value=50, step=10, help="풀베기, 모니터링 비용 등")

    # [섹션 5] 탄소 가격
    st.markdown("---")
    st.subheader("5️⃣ 탄소 가격 시나리오")
    price_scenario = st.selectbox("가격 전망", ["Base (기준)", "High (낙관)", "Low (보수)"])
    price_col_map = {"Base (기준)": "price_base", "High (낙관)": "price_high", "Low (보수)": "price_low"}
    price_col = price_col_map[price_scenario]

# -----------------------------------------------------------
# 4. 타이틀 및 로직
# -----------------------------------------------------------
forest_type = "혼효림 (Mixed Forest)" if len(selected_species) > 1 else "단순림 (Monoculture)"
st.title(f"🌲 {forest_type} 사업성 분석 시뮬레이터")

# -----------------------------------------------------------
# 5. 통합 계산 엔진
# -----------------------------------------------------------
years = list(range(2026, 2026 + project_period + 1))

total_biomass_carbon = np.zeros(project_period + 1)
total_soil_carbon = np.zeros(project_period + 1)

# CBI 계산 변수
total_native_ratio = 0
weighted_water_score = 0

# 자생종 구분 함수
def check_native(name):
    native_keywords = ["소나무", "상수리", "신갈", "졸참", "굴참", "잣나무"] 
    return any(k in name for k in native_keywords)

for sp in selected_species:
    sp_row = df_forest[df_forest['name'] == sp].iloc[0]
    ratio = species_ratios[sp]
    
    # 1. 탄소량 계산
    x_points = list(range(0, 51, 5))
    y_points = [sp_row[f'co2_yr_{y}'] for y in x_points]
    f_interp = interp1d(x_points, y_points, kind='linear', fill_value="extrapolate")
    standard_uptake = f_interp(range(project_period + 1))
    
    real_area = area * ratio
    adjusted_uptake = standard_uptake * real_area * density_factor
    soil_uptake = adjusted_uptake * 0.35 
    
    total_biomass_carbon += adjusted_uptake
    total_soil_carbon += soil_uptake
    
    # 2. CBI 가중치 계산
    if check_native(sp):
        total_native_ratio += ratio * 100
        
    try:
        ben_row = df_benefit.iloc[sp_row['id']-1]
        weighted_water_score += ben_row['water_index'] * ratio
    except:
        weighted_water_score += 3.0 * ratio

# --- 탄소 크레딧 계산 (버퍼 반영) ---
total_project_carbon = total_biomass_carbon + total_soil_carbon
baseline_carbon = total_project_carbon * 0.7 
gross_credit = total_project_carbon - baseline_carbon # 총 감축량
buffer_amount = gross_credit * buffer_ratio           # 버퍼(차감)
net_credit = gross_credit - buffer_amount             # 발급 가능 크레딧 (Issuable)

# -----------------------------------------------------------
# 6. 재무(Financial) 분석 엔진 (신규)
# -----------------------------------------------------------
# 1) 비용 (Cost)
total_initial_cost = initial_cost_per_ha * area * 10000 # 만원 -> 원 환산
total_annual_cost = annual_cost_per_ha * area * project_period * 10000
total_cost = total_initial_cost + total_annual_cost

# 2) 수익 (Revenue)
# 마지막 해의 누적 크레딧 * 마지막 해 가격 (단순화된 모델)
target_year = 2026 + project_period
if target_year > df_price['year'].max():
    unit_price = df_price.iloc[-1][price_col]
else:
    unit_price = df_price[df_price['year'] == target_year][price_col].values[0]

total_revenue = net_credit[-1] * unit_price

# 3) 순수익 및 ROI
net_profit = total_revenue - total_cost
roi = (net_profit / total_cost) * 100 if total_cost > 0 else 0

# -----------------------------------------------------------
# 7. CBI 점수 및 KPI
# -----------------------------------------------------------
cbi_native_score = (total_native_ratio / 100.0) * 5.0
cbi_water_score = weighted_water_score
cbi_conn_score = 3.0 # 기본값 (입력받지 않음)
cbi_diversity_score = min(5.0, 2.0 + (len(selected_species) * 0.6))

# [경제성 점수 로직 개선] ROI 기반 평가
# ROI가 0% 이하면 1점, 200% 이상이면 5점 (선형 보간)
if roi <= 0:
    cbi_econ_score = 1.0
elif roi >= 200:
    cbi_econ_score = 5.0
else:
    cbi_econ_score = 1.0 + (roi / 50.0) # 50% 당 1점씩 증가

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("발급 가능 크레딧", f"{net_credit[-1]:,.0f} Credit", f"버퍼 {buffer_ratio*100}% 차감 후")
with col2:
    st.metric("예상 순수익 (Net Profit)", f"₩{net_profit/100000000:.1f} 억", f"ROI {roi:.1f}%")
with col3:
    st.metric("총 사업 비용", f"₩{total_cost/100000000:.1f} 억", f"초기투자 + {project_period}년 관리비")
with col4:
    st.metric("CBI 경제성 등급", f"{cbi_econ_score:.1f} / 5.0", f"ROI 기반 평가")

st.markdown("---")

# -----------------------------------------------------------
# 8. 시각화 (수정됨: 범례 이동 및 경제성 차트 추가)
# -----------------------------------------------------------
c_main, c_sub = st.columns([2, 1])

with c_main:
    st.subheader("📊 탄소 저장 및 추가성 (Additionality)")
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=years, y=total_biomass_carbon, mode='lines', name='🌲 입목 바이오매스', stackgroup='one', line=dict(width=0, color='#27ae60')))
    fig.add_trace(go.Scatter(x=years, y=total_soil_carbon, mode='lines', name='🟤 토양/기타 저장고', stackgroup='one', line=dict(width=0, color='#8d6e63')))
    fig.add_trace(go.Scatter(x=years, y=baseline_carbon, mode='lines', name='📉 베이스라인 (무관리)', line=dict(color='#34495e', width=2, dash='dash')))
    
    # [수정] 범례를 그래프 위로 이동하여 가림 현상 방지
    fig.update_layout(
        xaxis_title="연도", yaxis_title="누적 tCO₂", 
        height=400, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5),
        margin=dict(t=50) # 범례 공간 확보
    )
    st.plotly_chart(fig, use_container_width=True)

with c_sub:
    st.subheader("🕸️ CBI 가치 평가")
    
    categories = ['자생종 비율', '수자원 조절', '생태 연결성', '종 다양성', '경제적 가치(ROI)']
    r_values = [cbi_native_score, cbi_water_score, cbi_conn_score, cbi_diversity_score, cbi_econ_score]
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=r_values, theta=categories, fill='toself', name='Project Score',
        line=dict(color='#145A32')
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=False, height=350,
        margin=dict(l=40, r=40, t=30, b=20)
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # 경제성 평가 방법론 설명
    with st.expander("💡 경제적 가치 평가 방법론"):
        st.caption(f"""
        **ROI(투자대비수익률) 기반 점수 산정**
        - 현재 ROI: **{roi:.1f}%**
        - 총 수익: {total_revenue/100000000:.1f}억 (크레딧 판매)
        - 총 비용: {total_cost/100000000:.1f}억 (조성+관리)
        - 점수 로직: ROI 0%이하(1점) ~ 200%이상(5점)
        """)

# -----------------------------------------------------------
# 9. 데이터 다운로드
# -----------------------------------------------------------
with st.expander("📥 상세 리포트 데이터 다운로드"):
    df_res = pd.DataFrame({
        "Year": years,
        "Total_Project_Carbon": total_project_carbon,
        "Baseline": baseline_carbon,
        "Gross_Credit": gross_credit,
        "Buffer_Deduction": buffer_amount,
        "Net_Issuable_Credit": net_credit
    })
    st.dataframe(df_res, use_container_width=True)
    st.download_button("CSV 다운로드", df_res.to_csv(index=False).encode('utf-8-sig'), "simulation_financial_report.csv")
