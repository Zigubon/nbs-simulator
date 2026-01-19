import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
    try:
        # CSV 파일 로드 (파일명이 정확해야 합니다)
        forest = pd.read_csv('forest_data_2026.csv')
        price = pd.read_csv('carbon_price_scenarios.csv')
        benefit = pd.read_csv('co_benefits.csv')
        return forest, price, benefit
    except FileNotFoundError as e:
        st.error(f"❌ 데이터 파일을 찾을 수 없습니다: {e}")
        return None, None, None

def interpolate_growth(forest_df, species_id, years=30):
    """
    5년 단위 데이터를 연 단위로 선형 보간 (0, 5, 10, ... 50년)
    """
    species_data = forest_df[forest_df['id'] == species_id].iloc[0]
    
    # [중요] CSV 데이터 컬럼에 맞춘 X축 포인트 (0년 ~ 50년, 5년 간격)
    x_points = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    
    # 해당 컬럼의 값 가져오기
    y_points = []
    for y in x_points:
        col_name = f'co2_yr_{y}'
        # 데이터가 있으면 가져오고, 없으면(범위 초과시) 마지막 값 유지
        if col_name in species_data:
            y_points.append(species_data[col_name])
        else:
            y_points.append(y_points[-1] if y_points else 0)

    # 보간 함수 생성 (Linear Interpolation)
    f = interp1d(x_points, y_points, kind='linear', fill_value="extrapolate")
    
    # 1년 ~ 설정된 사업기간(years)까지의 값 반환
    return f(np.arange(1, years + 1))

df_forest, df_price, df_benefit = load_data()

if df_forest is None:
    st.stop()

# -----------------------------------------------------------
# 3. 사이드바: 조건 변수 입력
# -----------------------------------------------------------
with st.sidebar:
    st.header("🎛️ 시뮬레이션 조건 설정")
    
    # A. 수종 선택 (안전한 기본값 설정 로직)
    st.subheader("1. 사업 개요")
    
    # 데이터에 있는 모든 수종 이름 가져오기
    available_names = df_forest['name'].tolist()
    
    # 우리가 원하는 기본값 후보
    desired_defaults = ['상수리나무', '백합나무']
    
    # 실제 데이터에 존재하는 것만 걸러내기 (에러 방지 핵심!)
    valid_defaults = [name for name in desired_defaults if name in available_names]
    
    # 만약 데이터에 원하는 게 하나도 없다면? 목록의 첫 번째를 기본값으로 사용
    if not valid_defaults and available_names:
        valid_defaults = [available_names[0]]

    selected_names = st.multiselect(
        "식재 수종 선택 (혼효림 조성)", 
        options=available_names,
        default=valid_defaults,  # 안전하게 걸러진 기본값 적용
        max_selections=5,
        help="여러 수종을 선택하면 면적을 균등하게 분할하여 식재한다고 가정합니다. (예: 교목 + 관목 혼합 식재)"
    )
    
    if not selected_names:
        st.warning("최소 1개 이상의 수종을 선택해주세요.")
        st.stop()

    # 선택된 수종들의 ID 추출
    selected_ids = df_forest[df_forest['name'].isin(selected_names)]['id'].values
    
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        area_ha = st.number_input("부지 면적 (ha)", value=10.0, step=0.1)
    with col_input2:
        # 식재 밀도
        density_ratio = st.number_input("식재 밀도 (%)", value=100, step=10, help="산림청 표준(3,000본/ha) 대비 식재 비율. 공원형/가로수는 50% 이하 권장") / 100

    sim_years = st.slider("사업 기간 (년)", 10, 40, 30)

    # 예상 식재 본수 계산 (표준: ha당 3,000그루 가정)
    standard_density_per_ha = 3000 
    estimated_trees = int(area_ha * standard_density_per_ha * density_ratio)
    st.caption(f"🌲 예상 식재 본수: 약 **{estimated_trees:,.0f} 그루**")

    # B. 리스크 및 재무 설정
    st.subheader("2. 리스크 & 재무")
    survival_rate = st.slider("평균 생존율 (%)", 50, 100, 90) / 100
    
    initial_cost = st.number_input("초기 조성비 (백만원)", value=100) * 1000000
    maintenance_cost = st.number_input("연간 관리비 (백만원)", value=5) * 1000000
    discount_rate = st.slider("할인율 (Discount Rate, %)", 0.0, 10.0, 3.0, 0.1) / 100
    price_adj = st.slider("탄소가격 추가 상승률 (CAGR, %)", -5.0, 10.0, 0.0, 0.1) / 100
    
    st.markdown("---")
    st.caption("Developed by Zigubon Lab")

# -----------------------------------------------------------
# 4. 시뮬레이션 엔진 (Core Logic)
# -----------------------------------------------------------
# A. 탄소 흡수량 (혼효림 로직: 평균 성장곡선 생성)
combined_growth = np.zeros(sim_years)

for s_id in selected_ids:
    g_curve = interpolate_growth(df_forest, s_id, sim_years)
    combined_growth += g_curve

# 수종별 평균 흡수량 (균등 면적 분할 가정)
avg_growth_curve = combined_growth / len(selected_names)

# [최종 흡수량] = 평균곡선 * 면적 * 생존율 * 식재밀도
adjusted_growth = avg_growth_curve * area_ha * survival_rate * density_ratio 

df_sim = pd.DataFrame({
    'year': range(2026, 2026 + sim_years),
    'age': range(1, sim_years + 1),
    'absorption_t': adjusted_growth
})
df_sim['cum_absorption'] = df_sim['absorption_t'].cumsum()

# B. 재무 분석
# 가격 데이터 매핑
price_base = df_price['price_base'].values
if len(price_base) < sim_years:
    price_base = np.pad(price_base, (0, sim_years - len(price_base)), 'edge')
else:
    price_base = price_base[:sim_years]

# 사용자 가격 조정 적용
user_price_curve = price_base * ((1 + price_adj) ** np.arange(sim_years))

df_sim['revenue'] = df_sim['absorption_t'] * user_price_curve
df_sim['cost'] = maintenance_cost
df_sim.loc[0, 'cost'] += initial_cost 

df_sim['net_cashflow'] = df_sim['revenue'] - df_sim['cost']
df_sim['cum_cashflow'] = df_sim['net_cashflow'].cumsum()

# NPV & ROI
df_sim['discount_factor'] = 1 / ((1 + discount_rate) ** np.arange(sim_years))
df_sim['pv'] = df_sim['net_cashflow'] * df_sim['discount_factor']
npv = df_sim['pv'].sum()
roi = (df_sim['net_cashflow'].sum() / (initial_cost + maintenance_cost * sim_years)) * 100

# -----------------------------------------------------------
# 5. 대시보드 출력
# -----------------------------------------------------------
species_title = ", ".join(selected_names)
if len(selected_names) > 3:
    species_title = f"{selected_names[0]} 외 {len(selected_names)-1}종"

st.title(f"📊 {species_title} NbS 투자 시뮬레이터")
st.markdown(f"**조건:** {area_ha}ha (밀도 {density_ratio*100:.0f}%) | 생존율 {survival_rate*100:.0f}% | 할인율 {discount_rate*100:.1f}%")

# KPI Cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("총 탄소 흡수량", f"{df_sim['cum_absorption'].iloc[-1]:,.0f} tCO₂", 
          delta=f"{len(selected_names)}종 혼합 식재", delta_color="inverse")
c2.metric("총 매출액", f"{df_sim['revenue'].sum()/100000000:.2f} 억원", 
          delta=f"가격조정 {price_adj*100:+.1f}%")
c3.metric("순현재가치 (NPV)", f"{npv/100000000:.2f} 억원", help="0보다 크면 투자 가치 있음")
c4.metric("투자 수익률 (ROI)", f"{roi:.1f} %", delta="손익분기 체크", delta_color="normal" if roi>0 else "inverse")

# Tabs
tab1, tab2 = st.tabs(["📈 재무/수익성 분석", "🌿 탄소/ESG 분석"])

with tab1:
    col_g1, col_g2 = st.columns([2, 1])
    with col_g1:
        fig_cf = go.Figure()
        fig_cf.add_trace(go.Bar(x=df_sim['year'], y=df_sim['revenue'], name='매출', marker_color='#27ae60'))
        fig_cf.add_trace(go.Bar(x=df_sim['year'], y=-df_sim['cost'], name='비용', marker_color='#e74c3c'))
        fig_cf.add_trace(go.Scatter(x=df_sim['year'], y=df_sim['cum_cashflow'], name='누적 현금흐름', line=dict(color='#2c3e50', width=3)))
        fig_cf.update_layout(title="연도별 현금흐름", barmode='relative', yaxis_title="금액 (원)", hovermode="x unified")
        st.plotly_chart(fig_cf, use_container_width=True)
    
    with col_g2:
        st.subheader("💡 투자 포인트")
        if npv > 0:
            st.success(f"**투자 적격**\n\n현재 가치 기준 **{npv/1000000:,.0f}백만원**의 순이익이 예상됩니다.")
        else:
            st.error(f"**투자 주의**\n\n비용이 수익보다 큽니다. 장기적 관점이 필요합니다.")
        st.dataframe(df_sim[['year', 'revenue', 'cost', 'net_cashflow']].style.format("{:,.0f}"), height=300)

with tab2:
    # ESG Data (선택된 수종 평균)
    selected_benefits = df_benefit[df_benefit['id'].isin(selected_ids)]
    
    # 데이터가 없을 경우를 대비한 안전장치
    if not selected_benefits.empty:
        avg_bio = selected_benefits['biodiversity_index'].mean()
        avg_water = selected_benefits['water_index'].mean()
        avg_fire = selected_benefits['fire_resistance'].mean()
    else:
        avg_bio, avg_water, avg_fire = 0, 0, 0

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("ESG Impact")
        st.markdown(f"""
        - **생물다양성 지수:** ⭐ {avg_bio:.1f} / 5.0
        - **수원 함양 기능:** 💧 {avg_water:.1f} / 5.0
        - **내화성(산불저항):** 🔥 {avg_fire:.1f} / 3.0
        """)
        if len(selected_names) > 1:
            st.info(f"✅ **혼효림 효과:** {len(selected_names)}종 혼합 식재로 생태계 복원력이 강화되었습니다.")
            
    with c2:
        st.subheader("🚗 생활 속 체감 효과")
        # 승용차 1대 = 2.4톤 기준
        avg_absorption = df_sim['absorption_t'].mean()
        cars_offset = avg_absorption / 2.4
        
        st.metric("연간 승용차 배출 상쇄", f"{cars_offset:,.0f} 대", help="기준: 연간 2.4tCO2 배출 (국립산림과학원)")
        st.caption(f"이 숲은 매년 승용차 **{int(cars_offset)}대**의 배출량을 지웁니다.")
        st.progress(min(1.0, cars_offset / 100))
