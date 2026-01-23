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
    project_period = st.slider("사업 기간 (년)", 20, 100, 30)
    
    st.markdown("---")
    
    # [섹션 2] 수종 및 비율 (신규 기능)
    st.subheader("2️⃣ 수종 및 구성비")
    species_list = df_forest['name'].unique()
    
    # 기본값: 데이터가 있으면 상위 2개, 아니면 1개
    default_sp = [species_list[0], species_list[1]] if len(species_list) > 1 else [species_list[0]]
    selected_species = st.multiselect("식재 수종 선택", species_list, default=default_sp)
    
    if not selected_species:
        st.warning("⚠️ 최소 1개 이상의 수종을 선택해주세요.")
        st.stop()
    
    # 수종별 점유 비율 슬라이더 생성
    species_ratios = {}
    if len(selected_species) > 1:
        st.caption("👇 수종별 점유 비율(%)을 설정하세요")
        total_ratio = 0
        for sp in selected_species:
            default_val = int(100 / len(selected_species))
            # 마지막 수종은 남은 비율 자동 할당 등의 로직이 복잡하므로, 사용자 자율에 맡기고 경고만 표시
            ratio = st.slider(f"{sp} 비율", 0, 100, default_val, key=f"ratio_{sp}")
            species_ratios[sp] = ratio / 100.0
            total_ratio += ratio
        
        if total_ratio != 100:
            st.error(f"⚠️ 비율 합계가 {total_ratio}%입니다. 100%에 맞춰주세요.")
    else:
        species_ratios[selected_species[0]] = 1.0

    st.markdown("---")
    
    # [섹션 3] 식재 밀도 (신규 기능)
    st.subheader("3️⃣ 식재 밀도 (Density)")
    density_help = """
    국립산림과학원 표준 흡수량은 '표준 밀도(약 3,000본/ha)' 기준입니다.
    - 100%: 표준 식재
    - 120%: 밀식 (흡수량 증가)
    - 80%: 소식 (흡수량 감소)
    """
    density_factor = st.slider("식재 밀도 지수 (%)", 50, 150, 100, help=density_help) / 100.0
    
    # 총 식재 본수 역산 (KPI용)
    estimated_trees = int(area * 3000 * density_factor)
    st.caption(f"🌲 총 추정 식재 본수: **{estimated_trees:,} 그루**")

    st.markdown("---")
    
    # [섹션 4] 경제성 가정 (기존 기능)
    st.subheader("4️⃣ 경제성 시나리오")
    price_scenario = st.selectbox("탄소배출권 가격", ["Base (기준)", "High (낙관)", "Low (보수)"])
    price_col_map = {"Base (기준)": "price_base", "High (낙관)": "price_high", "Low (보수)": "price_low"}
    price_col = price_col_map[price_scenario]

    # [방법론 명시]
    with st.expander("ℹ️ 방법론 (Methodology)"):
        st.info("""
        **국립산림과학원 산림탄소상쇄 표준 방법론 적용**
        1. **입목 바이오매스**: FBDC 임분수확표 기반 보간
        2. **기타 저장고**: 토양/낙엽/고사목 (확장계수법 적용)
        3. **베이스라인**: 무관리 시나리오 대비 순흡수량 산정
        """)

# -----------------------------------------------------------
# 4. 타이틀 및 로직 분기
# -----------------------------------------------------------
forest_type = "혼효림 (Mixed Forest)" if len(selected_species) > 1 else "단순림 (Monoculture)"
st.title(f"🌲 {forest_type} 탄소상쇄 시뮬레이터")
st.markdown(f"**{area}ha** 면적에 **{', '.join(selected_species)}**을 식재하는 프로젝트의 환경·경제적 가치를 분석합니다.")

# -----------------------------------------------------------
# 5. 통합 계산 엔진 (Tier 1 + New Features)
# -----------------------------------------------------------
years = list(range(2026, 2026 + project_period + 1))

# 결과 저장용 배열
total_biomass_carbon = np.zeros(project_period + 1)
total_soil_carbon = np.zeros(project_period + 1)
species_contributions = {} # 수종별 기여량 (파이차트용)

for sp in selected_species:
    sp_row = df_forest[df_forest['name'] == sp].iloc[0]
    
    # 1) 표준 성장 곡선 보간 (ha당)
    x_points = list(range(0, 51, 5))
    y_points = [sp_row[f'co2_yr_{y}'] for y in x_points]
    f_interp = interp1d(x_points, y_points, kind='linear', fill_value="extrapolate")
    standard_uptake_per_ha = f_interp(range(project_period + 1))
    
    # 2) [신규] 실제 면적 및 밀도 적용
    # 해당 수종의 실제 식재 면적 = 전체 면적 * 설정한 비율
    real_area = area * species_ratios[sp]
    
    # 밀도 보정 적용 (단순 선형 비례 가정)
    adjusted_uptake = standard_uptake_per_ha * real_area * density_factor
    
    # 3) [기존 Tier 1] 토양 및 기타 저장고 계산 (바이오매스의 35% 가정)
    soil_uptake = adjusted_uptake * 0.35
    
    # 합산
    total_biomass_carbon += adjusted_uptake
    total_soil_carbon += soil_uptake
    
    # 수종별 총 기여량 저장 (마지막 해 기준 누적량)
    species_contributions[sp] = adjusted_uptake[-1] + soil_uptake[-1]

# 총 프로젝트 탄소량
total_project_carbon = total_biomass_carbon + total_soil_carbon

# [기존] 베이스라인 (Baseline) 계산 - 무관리 시 70% 수준 가정
baseline_carbon = total_project_carbon * 0.7 
net_credit = total_project_carbon - baseline_carbon # 순 감축량

# -----------------------------------------------------------
# 6. 결과 대시보드 (KPIs)
# -----------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

final_carbon = total_project_carbon[-1]

# 경제 가치 (기간 마지막 해의 가격 적용)
target_year = 2026 + project_period
if target_year > df_price['year'].max():
    unit_price = df_price.iloc[-1][price_col]
else:
    unit_price = df_price[df_price['year'] == target_year][price_col].values[0]
final_value = final_carbon * unit_price

# 승용차 상쇄 (연 2.43톤)
cars_offset = (final_carbon / project_period) / 2.43

# ESG 점수 (혼효림 가산점 + Tier 2 논리)
diversity_base = 3.5
mix_bonus = (len(selected_species) - 1) * 0.5
esg_score = min(5.0, diversity_base + mix_bonus)

with col1:
    st.metric("총 탄소 순흡수량", f"{final_carbon:,.0f} tCO₂", f"연평균 {final_carbon/project_period:,.0f}톤")
with col2:
    st.metric("예상 경제적 가치", f"₩{final_value/100000000:.1f} 억", f"톤당 {unit_price:,.0f}원 ({price_scenario})")
with col3:
    st.metric("승용차 배출 상쇄", f"{cars_offset:,.0f} 대/년", "1대당 2.43tCO₂ 기준")
with col4:
    st.metric("총 식재 본수", f"{estimated_trees:,} 본", f"밀도 {int(density_factor*100)}% 적용")

st.markdown("---")

# -----------------------------------------------------------
# 7. 통합 시각화 (Tier 1 + Tier 2)
# -----------------------------------------------------------
c_main, c_sub = st.columns([2, 1])

# [왼쪽] Tier 1: 누적 영역 차트 (저장고별 + 베이스라인)
with c_main:
    st.subheader("📊 탄소 저장고 및 베이스라인 분석")
    fig = go.Figure()
    
    # 1. 입목 바이오매스 (Layer 1)
    fig.add_trace(go.Scatter(
        x=years, y=total_biomass_carbon,
        mode='lines', name='🌲 입목 바이오매스',
        stackgroup='one',
        line=dict(width=0, color='#27ae60')
    ))
    
    # 2. 토양/낙엽/고사목 (Layer 2)
    fig.add_trace(go.Scatter(
        x=years, y=total_soil_carbon,
        mode='lines', name='🟤 토양 및 기타 저장고',
        stackgroup='one',
        line=dict(width=0, color='#8d6e63')
    ))
    
    # 3. 베이스라인 (비교선)
    fig.add_trace(go.Scatter(
        x=years, y=baseline_carbon,
        mode='lines', name='📉 베이스라인 (무관리)',
        line=dict(color='#7f8c8d', width=2, dash='dash')
    ))
    
    fig.update_layout(
        xaxis_title="연도", yaxis_title="누적 탄소 흡수량 (tCO₂)",
        height=400, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("✅ **추가성(Additionality):** 실선(프로젝트)과 점선(베이스라인)의 차이가 본 사업의 순수한 탄소 감축 효과입니다.")

# [오른쪽] Tier 2: ESG 레이더 차트 (복구됨)
with c_sub:
    st.subheader("🕸️ ESG Co-benefits")
    
    # 레이더 차트 점수 계산 (혼효림일수록 점수 상승)
    mix_ratio = len(selected_species)
    
    biodiversity = min(5.0, 3.0 + (mix_ratio * 0.5)) # 생물다양성
    water = 4.0 # 수자원 (기본 우수)
    disaster = min(5.0, 3.0 + (mix_ratio * 0.4)) # 재해방지 (혼효림 유리)
    recreation = 3.5 + (mix_ratio * 0.2) # 휴양
    economy = min(5.0, 3.5 + (final_value / 1000000000)) # 경제성 (매출 연동)

    categories = ['생물다양성', '수자원 함양', '재해 방지', '산림 휴양', '경제적 가치']
    r_values = [biodiversity, water, disaster, recreation, economy]
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=r_values, theta=categories, fill='toself',
        name='Project Score',
        line=dict(color='#145A32')
    ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=False, height=350,
        margin=dict(l=40, r=40, t=20, b=20)
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    
    if mix_ratio > 1:
        st.success(f"✨ **혼효림 효과:** {mix_ratio}종 혼합 식재로 **생물다양성** 및 **재해 방지** 기능이 강화되었습니다.")
    else:
        st.info("💡 **팁:** 수종을 추가하여 혼효림으로 구성하면 ESG 점수를 높일 수 있습니다.")

# -----------------------------------------------------------
# 8. 데이터 다운로드
# -----------------------------------------------------------
with st.expander("📥 상세 리포트 데이터 다운로드"):
    df_res = pd.DataFrame({
        "Year": years,
        "Total_Carbon": total_project_carbon,
        "Biomass_Carbon": total_biomass_carbon,
        "Soil_Carbon": total_soil_carbon,
        "Baseline": baseline_carbon,
        "Net_Credit": net_credit
    })
    st.dataframe(df_res, use_container_width=True)
    st.download_button("CSV 다운로드", df_res.to_csv(index=False).encode('utf-8-sig'), "simulation_full_report.csv")
