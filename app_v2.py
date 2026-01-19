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
    # 추천 조합: 교목 + 관목 (다층 식재)
    default_cands = ['상수리나무', '화살나무(관목)'] 
    valid_defaults = [n for n in default_cands if n in available_names]
    
    # 추천 조합이 없으면 첫 번째 수종 선택
    if not valid_defaults and available_names:
        valid_defaults = [available_names[0]]

    selected_names = st.multiselect(
        "수종 선택 (다층 식재)", 
        options=available_names,
        default=valid_defaults,
        help="교목(Tree)과 관목(Shrub)을 혼합 식재 시 '다층 식재'로 인식하여 흡수량을 합산합니다."
    )
    
    if not selected_names:
        st.warning("최소 1개 이상의 수종을 선택해주세요.")
        st.stop()

    c1, c2 = st.columns(2)
    area_ha = c1.number_input("면적 (ha)", value=10.0, step=0.1)
    density_ratio = c2.number_input("밀도 (%)", value=100, step=10, help="산림청 표준(3,000본/ha) 대비 식재 비율") / 100
    sim_years = st.slider("사업 기간 (년)", 10, 40, 30)

    # [2] 표준 방법론 적용 (Methodology Factors) - [복구 및 강화]
    st.subheader("2. 방법론 차감 계수 (Deduction)")
    with st.expander("ℹ️ 순흡수량(Net) 산정 기준"):
        st.markdown("""
        **표준 방법론(Standard Methodology) 적용:**
        * [cite_start]**사업 배출량 (Project Emissions):** 장비 가동, 비료 사용 등 사업 수행 중 발생하는 배출량 차감[cite: 12].
        * [cite_start]**누출 및 버퍼 (Leakage & Buffer):** 자연재해(화재, 병해충) 및 외부 배출 증가를 대비한 유보 물량 차감[cite: 12].
        """)

    col_m1, col_m2 = st.columns(2)
    project_emission_rate = col_m1.number_input("사업 배출 (%)", value=5.0, step=1.0) / 100
    buffer_rate = col_m2.number_input("버퍼(Risk) (%)", value=10.0, step=1.0) / 100

    # [3] 경제성 지표
    st.subheader("3. 경제성 시나리오")
    price_adj = st.slider("탄소가격 상승률 (CAGR, %)", -5.0, 10.0, 0.0, 0.5) / 100
    discount_rate = st.slider("할인율 (%)", 0.0, 10.0, 3.0, 0.1) / 100
    
    initial_cost = st.number_input("초기 조성비 (백만원)", value=100) * 1e6
    maintenance_cost = st.number_input("연간 관리비 (백만원)", value=5) * 1e6

    st.markdown("---")
    st.caption("Developed by Zigubon Lab")

# -----------------------------------------------------------
# 4. 엔진 계산 (Core Logic: 다층 식재 + Net Credit)
# -----------------------------------------------------------

selected_rows = df_forest[df_forest['name'].isin(selected_names)]
trees = selected_rows[selected_rows['type'] == 'Tree']
shrubs = selected_rows[selected_rows['type'] == 'Shrub']

# 1) Gross Absorption (총 흡수량)
# 다층 식재 로직: 교목(평균) + 관목(평균) = 합산(Layering)
tree_growth = np.zeros(sim_years)
if not trees.empty:
    for t_id in trees['id']:
        tree_growth += interpolate_growth(df_forest, t_id, sim_years)
    tree_growth /= len(trees) # 교목끼리는 공간 분할

shrub_growth = np.zeros(sim_years)
if not shrubs.empty:
    for s_id in shrubs['id']:
        shrub_growth += interpolate_growth(df_forest, s_id, sim_years)
    shrub_growth /= len(shrubs) # 관목끼리는 공간 분할

total_gross_curve = tree_growth + shrub_growth
gross_absorption = total_gross_curve * area_ha * density_ratio * 0.9 # 생존율 90%

# 2) Net Absorption (순 흡수량) - 방법론 적용
# Net = Gross * (1 - 사업배출 - 버퍼)
net_absorption = gross_absorption * (1 - project_emission_rate - buffer_rate)

# 데이터프레임
df_sim = pd.DataFrame({
    'year': range(2026, 2026 + sim_years),
    'gross_t': gross_absorption,
    'net_t': net_absorption
})
df_sim['cum_net'] = df_sim['net_t'].cumsum()
df_sim['cum_gross'] = df_sim['gross_t'].cumsum()

# 재무 계산 (Net 기준 수익)
price_base = df_price['price_base'].values[:sim_years]
if len(price_base) < sim_years:
     price_base = np.pad(price_base, (0, sim_years - len(price_base)), 'edge')

user_price = price_base * ((1 + price_adj) ** np.arange(sim_years))

df_sim['revenue'] = df_sim['net_t'] * user_price
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
st.title(f"📊 {', '.join(selected_names[:2])} NbS 투자 시뮬레이터")
st.markdown(f"**조건:** {area_ha}ha | 밀도 {density_ratio*100:.0f}% | 차감율(배출+버퍼) {(project_emission_rate+buffer_rate)*100:.0f}%")

# KPI Cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("총 순흡수량 (Net)", f"{df_sim['cum_net'].iloc[-1]:,.0f} tCO₂", 
          delta=f"총 흡수량(Gross) {df_sim['cum_gross'].iloc[-1]:,.0f} t", delta_color="normal")
c2.metric("예상 매출액", f"{df_sim['revenue'].sum()/1e8:.2f} 억원", 
          delta=f"CAGR {price_adj*100:+.1f}%")
c3.metric("순현재가치 (NPV)", f"{npv/1e8:.2f} 억원", help="순 흡수량(Net) 기준 평가")
c4.metric("투자 수익률 (ROI)", f"{roi:.1f} %", 
          delta="투자 적격" if roi > 0 else "투자 주의", delta_color="normal" if roi > 0 else "inverse")

# Tabs
tab1, tab2 = st.tabs(["📈 경제성/방법론 분석", "🌿 ESG/상세 효과"])

# Tab 1: Methodology & Finance
with tab1:
    col_l, col_r = st.columns([2,1])
    with col_l:
        st.markdown("##### 📉 흡수량 차감 분석 (Gross vs Net)")
        fig_area = go.Figure()
        fig_area.add_trace(go.Scatter(x=df_sim['year'], y=df_sim['cum_gross'], fill='tozeroy', name='총 흡수량(Gross)', line=dict(color='#bdc3c7')))
        fig_area.add_trace(go.Scatter(x=df_sim['year'], y=df_sim['cum_net'], fill='tozeroy', name='순 흡수량(Net)', line=dict(color='#27ae60')))
        fig_area.update_layout(height=350, yaxis_title="누적 탄소 흡수량 (tCO₂)", margin=dict(t=20, b=20), hovermode="x unified")
        st.plotly_chart(fig_area, use_container_width=True)
        
        st.markdown("##### 💰 연도별 현금흐름 (Cash Flow)")
        fig_cf = go.Figure()
        fig_cf.add_trace(go.Bar(x=df_sim['year'], y=df_sim['revenue'], name='매출', marker_color='#2ecc71'))
        fig_cf.add_trace(go.Bar(x=df_sim['year'], y=-df_sim['cost'], name='비용', marker_color='#e74c3c'))
        fig_cf.add_trace(go.Scatter(x=df_sim['year'], y=df_sim['cum_cashflow'], name='누적현금', line=dict(color='#2c3e50', width=3)))
        fig_cf.update_layout(height=300, barmode='relative', yaxis_title="금액 (원)", margin=dict(t=20, b=20))
        st.plotly_chart(fig_cf, use_container_width=True)

    with col_r:
        st.info(f"""
        **📋 방법론 적용 결과**
        
        표준 방법론에 따라 총 흡수량에서 **{(project_emission_rate+buffer_rate)*100:.0f}%** 가 차감되었습니다.
        
        * **총 흡수량(Gross):** {df_sim['cum_gross'].iloc[-1]:,.0f} tCO₂
        * **(-) 사업 배출:** -{df_sim['cum_gross'].iloc[-1]*project_emission_rate:,.0f} tCO₂
        * **(-) 버퍼(Risk):** -{df_sim['cum_gross'].iloc[-1]*buffer_rate:,.0f} tCO₂
        * **(=) 인증 가능량(Net):** {df_sim['cum_net'].iloc[-1]:,.0f} Credit
        """)
        
        if npv > 0:
            st.success(f"**투자 적격 (Positive NPV)**\n\n약 **{npv/1e6:,.0f}백만원**의 순이익 예상")
        else:
            st.error("**투자 주의 (Negative NPV)**\n\n비용이 수익을 초과함")
            
        st.dataframe(df_sim[['year', 'revenue', 'cost', 'net_cashflow']].style.format("{:,.0f}"), height=200)

# Tab 2: ESG Details [복구된 부분]
with tab2:
    selected_ids = df_forest[df_forest['name'].isin(selected_names)]['id'].values
    selected_benefits = df_benefit[df_benefit['id'].isin(selected_ids)]
    
    if not selected_benefits.empty:
        # 혼효림일 경우 생물다양성 가산점 (10%)
        bonus = 1.1 if len(selected_names) > 1 else 1.0
        avg_bio = min(5.0, selected_benefits['biodiversity_index'].mean() * bonus)
        avg_water = selected_benefits['water_index'].mean()
        avg_fire = selected_benefits['fire_resistance'].mean()
    else:
        avg_bio, avg_water, avg_fire = 0, 0, 0

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("ESG Impact")
        st.markdown(f"""
        - **생물다양성:** ⭐ {avg_bio:.1f} / 5.0
        - **수원함양:** 💧 {avg_water:.1f} / 5.0
        - **내화성:** 🔥 {avg_fire:.1f} / 3.0
        """)
        if len(selected_names) > 1:
            st.success(f"✅ **다층 식재 효과:** {len(selected_names)}종 혼합으로 생태 가치가 강화되었습니다.")

        # [복구] 수종별 상세 정보 Expander
        with st.expander("ℹ️ 수종별 생태적 특성 상세 보기", expanded=True):
            for idx, row in selected_benefits.iterrows():
                st.markdown(f"**🌲 {row['name']}**")
                st.caption(f"{row['logic_note']}")
                st.divider()

    with c2:
        st.subheader("🚗 생활 체감 효과 (Net 기준)")
        offset_cars = df_sim['net_t'].mean() / 2.4
        
        st.metric("연간 승용차 상쇄", f"{offset_cars:,.0f} 대")
        
        st.caption(f"이 숲({area_ha}ha)은 매년 승용차 **{int(offset_cars)}대**가 뿜어내는 탄소를 0으로 만듭니다.")
        st.progress(min(1.0, offset_cars/100))
        st.info("기준: 국립산림과학원 (승용차 1대 = 2.4 tCO2/년)")
