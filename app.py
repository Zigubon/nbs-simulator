import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import interp1d

# -----------------------------------------------------------
# 1. 환경 설정
# -----------------------------------------------------------
st.set_page_config(page_title="ZIGUBON Simulator", page_icon="🌲", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f4f6f9; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #2c3e50; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# 2. 데이터 로드
# -----------------------------------------------------------
@st.cache_data
def load_data():
    # index.html에서 가상 파일 시스템으로 넘겨주므로, 로컬 파일처럼 읽으면 됩니다.
    forest = pd.read_csv("forest_data_2026.csv")
    price = pd.read_csv("carbon_price_scenarios.csv")
    benefit = pd.read_csv("co_benefits.csv")
    return forest, price, benefit

try:
    df_forest, df_price, df_benefit = load_data()
except Exception as e:
    st.error(f"데이터 파일 로드 실패. index.html 설정을 확인하세요.\n에러: {e}")
    st.stop()

# -----------------------------------------------------------
# 3. 사이드바 UI
# -----------------------------------------------------------
st.sidebar.title("🌲 산림탄소상쇄 시뮬레이터")
st.sidebar.caption("Powered by ZIGUBON")

area = st.sidebar.number_input("사업 면적 (ha)", min_value=1, value=10)
project_period = st.sidebar.slider("사업 기간 (년)", 10, 50, 30)

species_list = df_forest['name'].unique()
selected_species = st.sidebar.multiselect("식재 수종 (혼효림 구성)", species_list, default=[species_list[0]])

price_scenario = st.sidebar.selectbox("탄소 가격 전망", ["Base (기본)", "High (낙관)", "Low (보수)"], index=0)
price_col_map = {"Base (기본)": "price_base", "High (낙관)": "price_high", "Low (보수)": "price_low"}
price_col = price_col_map[price_scenario]

if not selected_species:
    st.warning("수종을 하나 이상 선택해주세요.")
    st.stop()

# -----------------------------------------------------------
# 4. 시뮬레이션 로직
# -----------------------------------------------------------
st.title(f"🌲 {', '.join(selected_species)} 혼효림 탄소상쇄 시뮬레이션")

years = list(range(2026, 2026 + project_period + 1))
chart_data = []
total_last_uptake = 0
species_results = {}

# 면적 N빵 (단순 균등 분배)
area_per_species = area / len(selected_species)

for sp in selected_species:
    sp_row = df_forest[df_forest['name'] == sp].iloc[0]
    
    # 5년 단위 데이터를 가져와서 1년 단위로 선형 보간
    x_points = list(range(0, 51, 5))
    y_points = [sp_row[f'co2_yr_{y}'] for y in x_points]
    
    f_interp = interp1d(x_points, y_points, kind='linear')
    yearly_uptake_per_ha = f_interp(range(project_period + 1))
    
    # 면적 적용
    final_uptake = yearly_uptake_per_ha * area_per_species
    total_last_uptake += final_uptake[-1]
    species_results[sp] = final_uptake[-1]
    
    chart_data.append(go.Scatter(x=years, y=final_uptake, name=sp, stackgroup='one'))

# -----------------------------------------------------------
# 5. 결과 시각화
# -----------------------------------------------------------
# 차트
fig = go.Figure(data=chart_data)
fig.update_layout(title="연도별 누적 탄소 흡수량", xaxis_title="연도", yaxis_title="tCO2", height=450, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# 경제성 분석
end_year = 2026 + project_period
if end_year > df_price['year'].max():
    unit_price = df_price.iloc[-1][price_col]
else:
    unit_price = df_price[df_price['year'] == end_year][price_col].values[0]

total_value = total_last_uptake * unit_price

# ESG 분석 (ID 매핑 로직은 간소화하여 평균값 적용)
# 실제 데이터에 'id' 컬럼이 매칭된다고 가정
try:
    selected_ids = df_forest[df_forest['name'].isin(selected_species)]['id']
    # co_benefits의 id와 forest의 id가 일치해야 정확함. 여기서는 예외처리 추가.
    avg_bio = df_benefit['biodiversity_index'].mean() 
    if len(selected_species) > 1: avg_bio = min(5.0, avg_bio + 0.5)
except:
    avg_bio = 3.0 # 매칭 실패 시 기본값

# KPI 카드
c1, c2, c3 = st.columns(3)
c1.metric("총 예상 흡수량", f"{total_last_uptake:,.1f} tCO₂")
c2.metric("예상 가치", f"₩{total_value:,.0f}")
c3.metric("ESG 지수", f"{avg_bio:.1f} / 5.0")

with st.expander("📊 상세 데이터 보기"):
    st.table(pd.DataFrame(list(species_results.items()), columns=['수종', '기여 흡수량(tCO2)']))
