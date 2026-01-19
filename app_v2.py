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
# 2. 데이터 로드 및 함수
# -----------------------------------------------------------
@st.cache_data
def load_data():
    try:
        forest = pd.read_csv('forest_data_2026.csv')
        price = pd.read_csv('carbon_price_scenarios.csv')
        benefit = pd.read_csv('co_benefits.csv')
        return forest, price, benefit
    except FileNotFoundError as e:
        st.error(f"❌ 데이터 파일을 찾을 수 없습니다: {e}")
        return None, None, None

def interpolate_growth(forest_df, species_id, years=30):
    species_data = forest_df[forest_df['id'] == species_id].iloc[0]
    # 5년 단위 데이터 보간 (0, 5, 10 ... 50)
    x_points = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    y_points = []
    for y in x_points:
        col_name = f'co2_yr_{y}'
        if col_name in species_data:
            y_points.append(species_data[col_name])
        else:
            y_points.append(y_points[-1] if y_points else 0)
    
    f = interp1d(x_points, y_points, kind='linear', fill_value="extrapolate")
    return f(np.arange(1, years + 1))

df_forest, df_price, df_benefit = load_data()

if df_forest is None:
    st.stop()

# -----------------------------------------------------------
# 3. 사이드바 (Inputs)
# -----------------------------------------------------------
with st.sidebar:
    st.header("🎛️ 시뮬레이션 조건")
    
    # [1] 수종 선택 (안전한 기본값 로직)
    st.subheader("1. 식재 계획 (Planting)")
    
    available_names = df_forest['name'].tolist()
    # 추천 조합: 교목 + 관목
    default_cands = ['상수리나무', '화살나무(관목)'] 
    valid_defaults = [n for n in default_cands if n in available_names]
    
    # 추천 조합이 없으면 첫 번째 수종 선택
    if not valid_defaults and available_names:
        valid_defaults = [available_names[0]]

    selected_names = st.multiselect(
        "수종 선택 (다층 식재)", 
        options=available_names,
        default=valid_defaults,
        help="교목(나무)과 관목(덤불)을 함께 선택하면 '다층 식재'로 인식하여 탄소 흡수량이 합산(Bonus)됩니다."
    )
    
    if not selected_names:
        st.warning("최소 1개 이상의 수종을 선택해주세요.")
        st.stop()

    # 선택된 ID 추출
    selected_ids = df_forest[df_forest['name'].isin(selected_names)]['id'].values

    c1, c2 = st.columns(2)
    area_ha = c1.number_input("면적 (ha)", value=10.0, step=0.1)
    density_ratio = c2.number_input("밀도 (%)", value=100, step=10, help="산림청 표준(3,000본/ha) 대비 식재 비율") / 100
    sim_years = st.slider("사업 기간 (년)", 10, 40, 30)

    # [2] 경제성 지표 가이드 (CAGR)
    st.subheader("2. 경제성 시나리오 (CAGR)")
    st.info("""
    **💡 시나리오 설정 가이드**
    * **0.0% (Base):** 물가상승률 수준 유지
    * **+3.0% (High):** 2030 NDC 및 규제 강화
    * **-1.0% (Low):** 경기 침체 및 규제 완화
    """)
    
    price_adj = st.slider("탄소가격 추가 상승률 (%)", -5.0, 10.0, 0.0, 0.5) / 100
    
    # [3] 리스크 및 비용
    st.subheader("3. 재무 및 리스크")
    survival_rate = st.slider("평균 생존율 (%)", 50, 100, 90) / 100
    discount_rate = st.slider("할인율 (%)", 0.0, 10.0, 3.0, 0.1) / 100
    
    initial_cost = st.number_input("초기 조성비 (백만원)", value=100) * 1e6
    maintenance_cost = st.number_input("연간 관리비 (백만원)", value=5) * 1e6
    
    st.markdown("---")
    st.caption("Developed by Zigubon Lab")

# -----------------------------------------------------------
# 4. 엔진 계산 (Core Logic: 다층 식재 합산)
# -----------------------------------------------------------

# 선택된 데이터 필터링
selected_rows = df_forest[df_forest['name'].isin(selected_names)]
trees = selected_rows[selected_rows['type'] == 'Tree']
shrubs = selected_rows[selected_rows['type'] == 'Shrub']

# 1) 교목층 (Tree Layer): 면적 분할 (Average)
tree_growth = np.zeros(sim_years)
if not trees.empty:
    for t_id in trees['id']:
        tree_growth += interpolate_growth(df_forest, t_id, sim_years)
    tree_growth /= len(trees)

# 2) 관목층 (Shrub Layer): 면적 분할 (Average)
shrub_growth = np.zeros(sim_years)
if not shrubs.empty:
    for s_id in shrubs['id']:
        shrub_growth += interpolate_growth(df_forest, s_id, sim_years)
    shrub_growth /= len(shrubs)

# 3) 최종 합산 (Layering): 교목 + 관목 (Additive)
total_growth_curve = tree_growth + shrub_growth

# 최종 흡수량: 곡선 * 면적 * 생존율 * 밀도
adjusted_growth = total_growth_curve * area_ha * survival_rate * density_ratio

# 데이터프레임 생성
df_sim = pd.DataFrame({
    'year': range(2026, 2026 + sim_years),
    'age': range(1, sim_years + 1),
    'absorption_t': adjusted_growth
})
df_sim['cum_absorption'] = df_sim['absorption_t'].cumsum()

# 재무 계산
price_base = df_price['price_base'].values
if len(price_base) < sim_years:
    price_base = np.pad(price_base, (0, sim_years - len(price_base)), 'edge')
else:
    price_base = price_base[:sim_years]

user_price = price_base * ((1 + price_adj) ** np.arange(sim_years))

df_sim['revenue'] = df_sim['absorption_t'] * user_price
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
# 5. 대시보드 (UI Output)
# -----------------------------------------------------------
species_title = ", ".join(selected_names[:2])
if len(selected_names) > 2:
    species_title += f" 외 {len(selected_names)-2}종"

st.title(f"📊 {species_title} NbS 투자 시뮬레이터")
st.markdown(f"**조건:** {area_ha}ha (밀도 {density_ratio*100:.0f}%) | 생존율 {survival_rate*100:.0f}% | 할인율 {discount_rate*100:.1f}%")

# KPI Cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("총 탄소 흡수량", f"{df_sim['cum_absorption'].iloc[-1]:,.0f} tCO₂", 
          delta="다층 식재 효과 적용됨" if not shrubs.empty and not trees.empty else None)
c2.metric("총 매출액", f"{df_sim['revenue'].sum()/1e8:.2f} 억원", 
          delta=f"CAGR {price_adj*100:+.1f}%")
c3.metric("순현재가치 (NPV)", f"{npv/1e8:.2f} 억원", 
          help="0보다 크면 투자 가치 있음")
c4.metric("투자 수익률 (ROI)", f"{roi:.1f} %", 
          delta="손익분기 달성" if roi > 0 else "손익분기 미달",
          delta_color="normal" if roi > 0 else "inverse")

# Tabs
tab1, tab2 = st.tabs(["📈 수익성 분석", "🌿 탄소/ESG 분석"])

# Tab 1: Financials
with tab1:
    col_l, col_r = st.columns([2,1])
    with col_l:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_sim['year'], y=df_sim['revenue'], name='매출', marker_color='#27ae60'))
        fig.add_trace(go.Bar(x=df_sim['year'], y=-df_sim['cost'], name='비용', marker_color='#e74c3c'))
        fig.add_trace(go.Scatter(x=df_sim['year'], y=df_sim['cum_cashflow'], name='누적현금', line=dict(color='#2c3e50', width=3)))
        fig.update_layout(title="연도별 현금흐름 (Cash Flow)", barmode='relative', yaxis_title="금액 (원)", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    
    with col_r:
        st.subheader("💡 투자 포인트")
        if npv > 0:
            st.success(f"**투자 적격 (Positive NPV)**\n\n이 프로젝트는 현재 가치 기준으로 **약 {npv/1e6:,.0f}백만원**의 초과 이익을 창출합니다.")
        else:
            st.error(f"**투자 주의 (Negative NPV)**\n\n현재 조건에서는 비용이 수익보다 큽니다. 초기 비용을 줄이거나 탄소 가격 상승을 기다려야 합니다.")
        
        # 다층 식재 성공 메시지
        if not shrubs.empty and not trees.empty:
            st.info("✅ **다층 식재(Multi-layer) 효과:**\n\n교목 하부에 관목을 식재하여, 단일 수종 대비 공간 효율과 탄소 흡수량이 극대화되었습니다.")
            
        st.dataframe(df_sim[['year','revenue','cost']].style.format("{:,.0f}"), height=250)

# Tab 2: ESG Details
with tab2:
    # 선택된 수종의 ESG 점수 평균 계산
    selected_benefits = df_benefit[df_benefit['id'].isin(selected_ids)]
    
    if not selected_benefits.empty:
        avg_bio = selected_benefits['biodiversity_index'].mean()
        avg_water = selected_benefits['water_index'].mean()
        avg_fire = selected_benefits['fire_resistance'].mean()
        
        # 혼효림/다층 식재 보너스 점수 (로직 반영)
        bonus = 1.1 if len(selected_names) > 1 else 1.0
        avg_bio = min(5.0, avg_bio * bonus)
    else:
        avg_bio, avg_water, avg_fire = 0, 0, 0

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("ESG Impact & Co-benefits")
        st.markdown(f"""
        - **생물다양성 지수:** ⭐ {avg_bio:.1f} / 5.0
        - **수원 함양 기능:** 💧 {avg_water:.1f} / 5.0
        - **내화성(산불저항):** 🔥 {avg_fire:.1f} / 3.0
        """)
        
        if len(selected_names) > 1:
             st.success(f"🌿 **생태 복원력 강화:** {len(selected_names)}종 이상의 수종 혼합으로 단일림 대비 생태적 가치가 상승했습니다.")

        with st.expander("ℹ️ 수종별 생태적 특성 보기"):
            for idx, row in selected_benefits.iterrows():
                st.write(f"**{row['name']}:** {row['logic_note']}")

    with c2:
        st.subheader("🚗 생활 속 체감 효과")
        # 승용차 1대 = 2.4톤 기준
        avg_absorption = df_sim['absorption_t'].mean()
        cars_offset = avg_absorption / 2.4
        
        st.metric(
            label="연간 승용차 배출 상쇄", 
            value=f"{cars_offset:,.0f} 대",
            help="출처: 국립산림과학원 (승용차 1대 연간 배출량 약 2.4tCO2 기준)"
        )
        st.caption(f"이 숲({area_ha}ha)은 매년 승용차 **{int(cars_offset)}대**가 뿜어내는 탄소를 0으로 만듭니다.")
        st.progress(min(1.0, cars_offset / 100))
