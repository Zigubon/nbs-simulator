import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.interpolate import interp1d

# -----------------------------------------------------------
# 1. 환경 설정 및 스타일
# -----------------------------------------------------------
st.set_page_config(page_title="ZIGUBON | NbS Investment Simulator", page_icon="🌲", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f4f6f9; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #2c3e50; }
    .big-font { font-size:18px !important; font-weight: bold; color: #27ae60; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# 2. 데이터 로드 및 로직 함수
# -----------------------------------------------------------
@st.cache_data
def load_data():
    forest = pd.read_csv('forest_data_2026.csv')
    price = pd.read_csv('carbon_price_scenarios.csv')
    benefit = pd.read_csv('co_benefits.csv')
    return forest, price, benefit

def interpolate_growth(forest_df, species_id, years=30):
    species_data = forest_df[forest_df['id'] == species_id].iloc[0]
    x_points = [0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
    y_points = [0] + [species_data[f'co2_yr_{y}'] for y in x_points[1:]]
    f = interp1d(x_points, y_points, kind='linear', fill_value="extrapolate")
    return f(np.arange(1, years + 1))

try:
    df_forest, df_price, df_benefit = load_data()
except:
    st.error("데이터 파일을 찾을 수 없습니다. (forest_data_2026.csv 등)")
    st.stop()

# -----------------------------------------------------------
# 3. 사이드바: 조건 변수 입력 (Simulation Control)
# -----------------------------------------------------------
with st.sidebar:
    st.header("🎛️ 시뮬레이션 조건 설정")
    
    # A. 기본 설정
    st.subheader("1. 사업 개요")
    species_name = st.selectbox("수종 선택", df_forest['name'], index=6) # 상수리나무 기본
    species_id = df_forest[df_forest['name'] == species_name]['id'].values[0]
    area_ha = st.number_input("부지 면적 (ha)", value=10.0, step=0.1)
    sim_years = st.slider("사업 기간 (년)", 10, 40, 30)

    # B. 물리적 리스크 (Survival)
    st.subheader("2. 리스크 변수 (Risk)")
    survival_rate = st.slider("평균 생존율 (%)", 50, 100, 90, help="태풍, 병해충 등으로 인한 예상 생존율") / 100
    
    # C. 재무 설정 (Investment)
    st.subheader("3. 투자 및 재무")
    initial_cost = st.number_input("초기 조성비 (백만원)", value=100) * 1000000
    maintenance_cost = st.number_input("연간 관리비 (백만원)", value=5) * 1000000
    discount_rate = st.slider("할인율 (Discount Rate, %)", 0.0, 10.0, 3.0, 0.1, help="미래 가치를 현재 가치로 환산할 때 적용 (사회적 할인율 등)") / 100

    # D. 가격 민감도 (Sensitivity)
    st.subheader("4. 시장 전망 (Market)")
    price_adj = st.slider("탄소가격 추가 상승률 (CAGR, %)", -5.0, 10.0, 0.0, 0.1, help="기본 시나리오 대비 추가 상승/하락률 적용") / 100
    
    st.markdown("---")
    st.caption("Developed by Zigubon Lab")

# -----------------------------------------------------------
# 4. 시뮬레이션 엔진 계산
# -----------------------------------------------------------
# A. 탄소 흡수량 (물리적 변수 적용)
raw_growth = interpolate_growth(df_forest, species_id, sim_years)
adjusted_growth = raw_growth * area_ha * survival_rate # 생존율 반영

df_sim = pd.DataFrame({
    'year': range(2026, 2026 + sim_years),
    'age': range(1, sim_years + 1),
    'absorption_t': adjusted_growth
})
df_sim['cum_absorption'] = df_sim['absorption_t'].cumsum()

# B. 재무 분석 (경제적 변수 적용)
# 가격 데이터 매핑 (부족하면 마지막 값으로 채움)
price_base = df_price['price_base'].values
if len(price_base) < sim_years:
    price_base = np.pad(price_base, (0, sim_years - len(price_base)), 'edge')
else:
    price_base = price_base[:sim_years]

# 사용자 지정 CAGR 적용
user_price_curve = price_base * ((1 + price_adj) ** np.arange(sim_years))

df_sim['revenue'] = df_sim['absorption_t'] * user_price_curve
df_sim['cost'] = maintenance_cost
df_sim.loc[0, 'cost'] += initial_cost # 첫해 초기비용 추가

df_sim['net_cashflow'] = df_sim['revenue'] - df_sim['cost']
df_sim['cum_cashflow'] = df_sim['net_cashflow'].cumsum()

# NPV 계산
df_sim['discount_factor'] = 1 / ((1 + discount_rate) ** np.arange(sim_years))
df_sim['pv'] = df_sim['net_cashflow'] * df_sim['discount_factor']
npv = df_sim['pv'].sum()
roi = (df_sim['net_cashflow'].sum() / (initial_cost + maintenance_cost * sim_years)) * 100

# -----------------------------------------------------------
# 5. 대시보드 출력
# -----------------------------------------------------------
st.title(f"📊 {species_name} NbS 투자 시뮬레이터")
st.markdown(f"**조건:** {area_ha}ha 식재 | 생존율 {survival_rate*100:.0f}% | 할인율 {discount_rate*100:.1f}%")

# KPI Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("총 탄소 흡수량", f"{df_sim['cum_absorption'].iloc[-1]:,.0f} tCO₂", 
              delta=f"생존율 리스크 -{(1-survival_rate)*100:.0f}% 반영", delta_color="inverse")
with col2:
    st.metric("총 매출액 (Revenue)", f"{df_sim['revenue'].sum()/100000000:.2f} 억원", 
              delta=f"가격조정 {price_adj*100:+.1f}%")
with col3:
    st.metric("순현재가치 (NPV)", f"{npv/100000000:.2f} 억원", 
              help="미래의 현금흐름을 현재 가치로 환산한 값. 0보다 크면 투자 가치 있음.")
with col4:
    color = "normal" if roi > 0 else "inverse"
    st.metric("투자 수익률 (ROI)", f"{roi:.1f} %", delta="BEP(손익분기) 달성 여부 확인", delta_color=color)

# Tabs
tab1, tab2 = st.tabs(["📈 재무/수익성 분석", "🌿 탄소/ESG 분석"])

with tab1:
    col_g1, col_g2 = st.columns([2, 1])
    with col_g1:
        # Cashflow Chart
        fig_cf = go.Figure()
        fig_cf.add_trace(go.Bar(x=df_sim['year'], y=df_sim['revenue'], name='매출 (Revenue)', marker_color='#27ae60'))
        fig_cf.add_trace(go.Bar(x=df_sim['year'], y=-df_sim['cost'], name='비용 (Cost)', marker_color='#e74c3c'))
        fig_cf.add_trace(go.Scatter(x=df_sim['year'], y=df_sim['cum_cashflow'], name='누적 현금흐름', mode='lines', line=dict(color='#2c3e50', width=3)))
        fig_cf.update_layout(title="연도별 현금흐름 (Cash Flow)", barmode='relative', yaxis_title="금액 (원)", hovermode="x unified")
        st.plotly_chart(fig_cf, use_container_width=True)
    
    with col_g2:
        st.subheader("💡 투자 포인트")
        if npv > 0:
            st.success(f"**투자 적격 (Positive NPV)**\n\n이 프로젝트는 현재 가치 기준으로 **약 {npv/1000000:,.0f}백만원**의 초과 이익을 창출합니다.")
        else:
            st.error(f"**투자 주의 (Negative NPV)**\n\n현재 조건에서는 비용이 수익보다 큽니다. 초기 비용을 줄이거나 탄소 가격 상승을 기다려야 합니다.")
        
        st.dataframe(df_sim[['year', 'revenue', 'cost', 'net_cashflow']].style.format("{:,.0f}"), height=300)

with tab2:
    # ESG Data Logic
    b_info = df_benefit[df_benefit['id'] == species_id].iloc[0]
    
    st.subheader("ESG Impact & Co-benefits")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        - **생물다양성 지수:** ⭐ {b_info['biodiversity_index']} / 5.0
        - **수원 함양 기능:** 💧 {b_info['water_index']} / 5.0
        - **내화성(산불저항):** 🔥 {b_info['fire_resistance']} / 3.0
        """)
        st.info(f"ℹ️ **생태적 근거:** {b_info['logic_note']}")
        
        # --- [추가된 부분] 승용차 상쇄 효과 시각화 ---
    with c2:
        st.markdown("### 🚗 생활 속 체감 효과")
        
        # 로직: 국립산림과학원 기준 승용차 1대 연간 배출량 = 약 2.4톤
        # 시뮬레이션된 숲의 '연평균' 흡수량을 기준으로 계산
        avg_absorption = df_sim['absorption_t'].mean()
        cars_offset = avg_absorption / 2.4
        
        st.metric(
            label="연간 승용차 배출 상쇄 효과",
            value=f"{cars_offset:,.0f} 대",
            delta="승용차 1대 = 2.4 tCO₂/년 기준",
            help="출처: 국립산림과학원 「주요 산림수종의 표준탄소흡수량」 (승용차 연평균 주행거리 15,000km 기준)"
        )
    
    with c3:
        # Sensitivity Analysis (간단 버전)
        st.caption("📉 생존율 변화에 따른 총 흡수량 민감도")
        sens_rates = [0.5, 0.7, 0.9, 1.0]
        sens_vals = [raw_growth.sum() * area_ha * r for r in sens_rates]
        fig_sens = px.bar(x=[f"{r*100}%" for r in sens_rates], y=sens_vals, labels={'x':'생존율', 'y':'총 흡수량(t)'}, title="Scenario Analysis")

        st.plotly_chart(fig_sens, use_container_width=True, height=250)
