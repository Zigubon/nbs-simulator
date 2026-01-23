import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import interp1d

# ==============================================================================
# 1. 페이지 및 스타일 설정 (Global Config)
# ==============================================================================
st.set_page_config(
    page_title="ZIGUBON | Forest Carbon & Finance Simulator",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS (UI 개선)
st.markdown("""
    <style>
    /* 전체 배경 및 폰트 */
    .main { background-color: #f8f9fa; font-family: 'Pretendard', sans-serif; }
    
    /* 카드 스타일 */
    .stCard { background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
    
    /* 메트릭 스타일 */
    div[data-testid="stMetricValue"] { font-size: 28px !important; color: #145A32; font-weight: 800; }
    div[data-testid="stMetricLabel"] { font-size: 15px !important; color: #555; font-weight: 600; }
    
    /* Expander 스타일 */
    div[data-testid="stExpander"] { border: 1px solid #e0e0e0; border-radius: 8px; background-color: #ffffff; }
    
    /* 사이드바 스타일 */
    section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #eee; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. 데이터 로드 및 전처리 (Data Layer)
# ==============================================================================
@st.cache_data
def load_data():
    """
    index.html에서 가상 파일 시스템으로 넘겨준 CSV 파일들을 로드합니다.
    실패 시 None을 반환하여 에러 처리를 유도합니다.
    """
    try:
        # 파일 경로: index.html과 같은 루트에 있다고 가정 (Pyodide 환경)
        forest = pd.read_csv("forest_data_2026.csv")
        price = pd.read_csv("carbon_price_scenarios.csv")
        benefit = pd.read_csv("co_benefits.csv")
        return forest, price, benefit
    except Exception as e:
        return None, None, None

df_forest, df_price, df_benefit = load_data()

# 데이터 로드 실패 시 중단
if df_forest is None:
    st.error("""
    ❌ **치명적인 오류: 데이터를 불러올 수 없습니다.**
    
    1. `forest_data_2026.csv`, `carbon_price_scenarios.csv`, `co_benefits.csv` 파일이 깃허브 저장소 최상위에 있는지 확인하세요.
    2. `index.html` 파일 내 `fileList` 변수에 파일명 오타가 없는지 확인하세요.
    """)
    st.stop()

# ==============================================================================
# 3. 헬퍼 함수 정의 (Logic Layer)
# ==============================================================================

def check_native(species_name):
    """
    CBI 지표 4번: 자생종 여부를 판별합니다.
    키워드 매칭 방식을 사용합니다.
    """
    native_keywords = ["소나무", "상수리", "신갈", "졸참", "굴참", "잣나무", "산벚", "전나무"]
    # 리기다, 낙엽송, 편백, 백합나무 등은 도입종으로 간주
    return any(k in species_name for k in native_keywords)

def get_co_benefit_score(species_name, benefit_df, column):
    """
    특정 수종의 ESG 점수(수자원, 내화성 등)를 조회합니다.
    데이터가 없으면 기본값(3.0)을 반환합니다.
    """
    # 1. 산림 데이터에서 ID 찾기
    try:
        sp_id = df_forest[df_forest['name'] == species_name]['id'].values[0]
        # 2. 베네핏 데이터에서 점수 찾기
        # 데이터 구조상 id가 1,2,3... 정수라고 가정
        score = benefit_df.loc[benefit_df['id'] == sp_id, column].values[0]
        return score
    except:
        return 3.0 # 데이터 매칭 실패 시 중간값

# ==============================================================================
# 4. 사이드바 UI - 입력 제어 (Control Layer)
# ==============================================================================
with st.sidebar:
    st.title("🌲 시뮬레이션 설정")
    st.markdown("---")
    
    # --------------------------------------------------------------------------
    # 섹션 1: 사업 기본 개요
    # --------------------------------------------------------------------------
    st.subheader("1️⃣ 사업 개요 (Project Basics)")
    area = st.number_input("사업 면적 (ha)", min_value=1.0, value=50.0, step=1.0, help="전체 사업 대상지의 면적입니다.")
    project_period = st.slider("사업 기간 (년)", 5, 50, 30, help="사업 시작부터 종료까지의 기간입니다 (최대 50년).")
    
    st.markdown("---")
    
    # --------------------------------------------------------------------------
    # 섹션 2: 수종 및 비율 (정밀 제어)
    # --------------------------------------------------------------------------
    st.subheader("2️⃣ 식재 포트폴리오 (Portfolio)")
    species_list = df_forest['name'].unique()
    
    # 기본 선택값 로직
    default_sp = [species_list[0], species_list[1]] if len(species_list) > 1 else [species_list[0]]
    selected_species = st.multiselect("수종 선택 (혼효림 권장)", species_list, default=default_sp)
    
    if not selected_species:
        st.warning("⚠️ 최소 1개 이상의 수종을 선택해야 합니다.")
        st.stop()
    
    # 수종별 비율 슬라이더 동적 생성
    species_ratios = {}
    st.caption("👇 수종별 식재 비율 (%)")
    
    if len(selected_species) > 1:
        total_ratio = 0
        for i, sp in enumerate(selected_species):
            # 남은 비율 자동 계산 로직은 UX상 복잡하므로 사용자 자율 조정 유도
            default_val = int(100 / len(selected_species))
            ratio = st.slider(f"{sp}", 0, 100, default_val, key=f"ratio_{sp}")
            species_ratios[sp] = ratio / 100.0
            total_ratio += ratio
        
        if total_ratio != 100:
            st.error(f"⚠️ 비율 합계가 {total_ratio}%입니다. 100%에 맞춰주세요.")
    else:
        # 단일 수종
        st.info(f"단일 수종: {selected_species[0]} 100%")
        species_ratios[selected_species[0]] = 1.0

    st.markdown("---")
    
    # --------------------------------------------------------------------------
    # 섹션 3: 생태 및 관리 (Density & Ecology)
    # --------------------------------------------------------------------------
    st.subheader("3️⃣ 생태 및 관리 (Ecology)")
    
    # 식재 밀도
    density_factor = st.slider(
        "식재 밀도 지수 (%)", 50, 150, 100, 
        help="표준 밀도(약 3,000본/ha) 대비 식재 비율. 120%는 밀식, 80%는 소식을 의미합니다."
    ) / 100.0
    
    # 총 본수 계산 (KPI용)
    total_trees = int(area * 3000 * density_factor)
    st.caption(f"🌱 총 식재 본수: **{total_trees:,} 그루**")
    
    # 생태 연결성 (CBI 지표 2)
    st.markdown("<br>", unsafe_allow_html=True)
    connectivity_score = st.select_slider(
        "생태 연결성 (Connectivity)",
        options=["고립 (낮음)", "일부 연결 (보통)", "핵심 축 연결 (높음)"],
        value="일부 연결 (보통)",
        help="대상지가 주변 산림 생태축(예: 백두대간, 정맥)과 얼마나 연결되어 있는지 평가합니다."
    )
    conn_map = {"고립 (낮음)": 1.0, "일부 연결 (보통)": 3.0, "핵심 축 연결 (높음)": 5.0}
    cbi_conn_val = conn_map[connectivity_score]

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 섹션 4: 재무 및 리스크 (Financial)
    # --------------------------------------------------------------------------
    st.subheader("4️⃣ 재무 및 리스크 (Financial)")
    
    # 버퍼 비율
    buffer_ratio = st.slider(
        "리스크 버퍼 (Buffer %)", 0, 30, 10,
        help="산불, 병해충 등 영구적 손실에 대비해 의무적으로 적립(판매 불가)하는 크레딧 비율입니다."
    ) / 100.0
    
    # 비용 입력 (ROI 계산용)
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        initial_cost = st.number_input("초기 조성비", value=1500, step=100, help="만원/ha (묘목, 식재, 설계비)")
    with col_c2:
        annual_cost = st.number_input("연간 관리비", value=50, step=10, help="만원/ha (모니터링, 풀베기)")
        
    # 탄소 가격 시나리오
    st.markdown("<br>", unsafe_allow_html=True)
    price_scenario = st.selectbox(
        "탄소 가격 전망 (Scenario)", 
        ["Base (기준)", "High (낙관)", "Low (보수)"]
    )
    price_col_map = {"Base (기준)": "price_base", "High (낙관)": "price_high", "Low (보수)": "price_low"}
    selected_price_col = price_col_map[price_scenario]

# ==============================================================================
# 5. 핵심 계산 엔진 (Calculation Engine)
# ==============================================================================

# 타이틀 및 헤더
forest_label = "혼효림 (Mixed Forest)" if len(selected_species) > 1 else "단순림 (Monoculture)"
st.title(f"🌲 {forest_label} 탄소·금융 시뮬레이터")
st.markdown(f"**{area}ha** 면적, **{project_period}년** 기간, **{', '.join(selected_species)}** 식재 사업에 대한 종합 가치 평가")

# 5-1. 물리적 흡수량 계산 (Carbon Physics)
years = list(range(2026, 2026 + project_period + 1)) # X축 (0년차 ~ N년차)
project_len = len(years)

# 배열 초기화
arr_biomass = np.zeros(project_len) # 입목 바이오매스
arr_soil = np.zeros(project_len)    # 토양/기타
cbi_native_score_acc = 0            # 자생종 점수 누적용
cbi_water_score_acc = 0             # 수자원 점수 누적용

for sp in selected_species:
    # 1) 데이터 추출
    row = df_forest[df_forest['name'] == sp].iloc[0]
    ratio = species_ratios[sp]
    
    # 2) 5년 단위 데이터 -> 1년 단위 선형 보간 (Interpolation)
    x_raw = list(range(0, 51, 5)) # 0, 5, 10 ... 50
    y_raw = [row[f'co2_yr_{y}'] for y in x_raw]
    
    # 보간 함수 생성 (ha당 누적 흡수량)
    f_interp = interp1d(x_raw, y_raw, kind='linear', fill_value="extrapolate")
    uptake_per_ha = f_interp(range(project_period + 1))
    
    # 3) 실제 흡수량 = (ha당 흡수량) * (실제 면적) * (밀도 계수)
    real_sp_area = area * ratio
    sp_uptake = uptake_per_ha * real_sp_area * density_factor
    
    # 4) 토양 탄소 (Tier 1 간이법: 바이오매스의 35% 추가 축적 가정)
    sp_soil = sp_uptake * 0.35
    
    # 5) 합산
    arr_biomass += sp_uptake
    arr_soil += sp_soil
    
    # 6) CBI 가중치 계산
    if check_native(sp):
        cbi_native_score_acc += (ratio * 100) # 비율만큼 가점 (최대 100)
    
    water_idx = get_co_benefit_score(sp, df_benefit, 'water_index')
    cbi_water_score_acc += (water_idx * ratio)

# 총 흡수량 (Gross)
arr_total_gross = arr_biomass + arr_soil

# 베이스라인 (Baseline) - 무관리 시 자연 생장 및 쇠퇴 고려 (70% 수준 가정)
arr_baseline = arr_total_gross * 0.7

# 순 감축량 (Net Credit) - 버퍼 차감 전
arr_net_gross = arr_total_gross - arr_baseline

# 버퍼 차감 (Buffer Deduction)
arr_buffer = arr_net_gross * buffer_ratio
arr_issuable = arr_net_gross - arr_buffer # 최종 발급 가능 크레딧

# 5-2. 재무 분석 (Financial Analysis)
# (1) 총 비용 (단위: 원)
cost_initial = initial_cost * area * 10000 
cost_annual_total = annual_cost * area * project_period * 10000
cost_total = cost_initial + cost_annual_total

# (2) 총 수익 (단위: 원)
# 간소화를 위해 '마지막 해의 누적 크레딧'을 '마지막 해의 가격'으로 평가
# (현금흐름할인법 DCF까지 가면 너무 복잡해지므로, 누적 관점의 ROI 산출)
target_year = 2026 + project_period
if target_year > df_price['year'].max():
    final_price = df_price.iloc[-1][selected_price_col]
else:
    final_price = df_price[df_price['year'] == target_year][selected_price_col].values[0]

revenue_total = arr_issuable[-1] * final_price

# (3) 수익성 지표
profit_net = revenue_total - cost_total
roi_percent = (profit_net / cost_total * 100) if cost_total > 0 else 0

# 5-3. CBI 및 ESG 점수 산출
# (1) 자생종 점수 (0~5)
score_native = (cbi_native_score_acc / 100.0) * 5.0

# (2) 수자원 점수 (1~5)
score_water = cbi_water_score_acc

# (3) 연결성 점수 (사용자 입력)
score_conn = cbi_conn_val

# (4) 경제성 점수 (ROI 연동)
# ROI 0% 이하 = 1점, 200% 이상 = 5점, 그 사이 선형 보간
if roi_percent <= 0:
    score_econ = 1.0
elif roi_percent >= 200:
    score_econ = 5.0
else:
    score_econ = 1.0 + (roi_percent / 50.0)

# (5) 생물다양성 보너스 (혼효림)
score_diversity = min(5.0, 2.0 + (len(selected_species) * 0.6))

# 종합 CBI
score_cbi_avg = (score_native + score_water + score_conn + score_econ + score_diversity) / 5.0

# 승용차 상쇄 대수 (국립산림과학원 2.43톤 기준)
cars_offset = (arr_issuable[-1] / project_period) / 2.43


# ==============================================================================
# 6. 결과 대시보드 (Visualization Layer)
# ==============================================================================

# 6-1. KPI 카드 (4 Columns)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "발급 가능 크레딧 (Net)", 
        f"{arr_issuable[-1]:,.0f} Credit",
        f"총량 {arr_total_gross[-1]:,.0f} - 버퍼 {buffer_ratio*100:.0f}%",
        help="베이스라인과 리스크 버퍼를 모두 차감한 후 실제 판매 가능한 크레딧 수량입니다."
    )

with col2:
    st.metric(
        "예상 순수익 (Profit)", 
        f"₩{profit_net/100000000:.1f} 억",
        f"ROI {roi_percent:.1f}%",
        help="총 매출에서 초기 조성비와 연간 관리비를 제외한 순수익입니다."
    )

with col3:
    st.metric(
        "CBI 종합 등급", 
        f"{score_cbi_avg:.1f} / 5.0",
        f"경제성 {score_econ:.1f}점 반영",
        help="싱가포르 지수(CBI)에 기반하여 생태, 사회, 경제적 가치를 종합 평가한 점수입니다."
    )

with col4:
    st.metric(
        "승용차 배출 상쇄", 
        f"{cars_offset:,.0f} 대/년",
        "1대당 2.43 tCO₂",
        help="연평균 순 흡수량을 승용차 1대의 연간 배출량으로 환산한 수치입니다."
    )

st.markdown("---")

# 6-2. 메인 차트 (2 Columns Layout)
chart_col_1, chart_col_2 = st.columns([2, 1])

with chart_col_1:
    st.subheader("📊 탄소 저장고 및 추가성 분석")
    
    fig_area = go.Figure()
    
    # Layer 1: 입목 바이오매스
    fig_area.add_trace(go.Scatter(
        x=years, y=arr_biomass,
        mode='lines', name='🌲 입목 바이오매스',
        stackgroup='one',
        line=dict(width=0, color='#27ae60')
    ))
    
    # Layer 2: 토양/기타
    fig_area.add_trace(go.Scatter(
        x=years, y=arr_soil,
        mode='lines', name='🟤 토양 및 기타 저장고',
        stackgroup='one',
        line=dict(width=0, color='#8d6e63')
    ))
    
    # Line: 베이스라인
    fig_area.add_trace(go.Scatter(
        x=years, y=arr_baseline,
        mode='lines', name='📉 베이스라인 (무관리)',
        line=dict(color='#34495e', width=2, dash='dash')
    ))
    
    # 레이아웃 조정 (범례 상단 배치)
    fig_area.update_layout(
        xaxis_title="연도 (Year)", 
        yaxis_title="누적 탄소 흡수량 (tCO₂)", 
        height=400,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        margin=dict(t=30)
    )
    st.plotly_chart(fig_area, use_container_width=True)

with chart_col_2:
    st.subheader("🕸️ CBI 가치 평가 (Radar)")
    
    categories = ['자생종 비율', '수자원 함양', '생태 연결성', '종 다양성', '경제성(ROI)']
    r_vals = [score_native, score_water, score_conn, score_diversity, score_econ]
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=r_vals, theta=categories,
        fill='toself', name='Project Score',
        line=dict(color='#145A32')
    ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=False,
        height=350,
        margin=dict(l=30, r=30, t=30, b=30)
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # 경제성 팁
    if score_econ < 3.0:
        st.caption("💡 **Tip:** 관리비를 줄이거나 탄소 가격 시나리오가 상승하면 경제성 점수가 올라갑니다.")

# 6-3. 상세 데이터 다운로드 (Expander)
with st.expander("📥 상세 리포트 데이터 (CSV 다운로드)"):
    df_result = pd.DataFrame({
        "Year": years,
        "Total_Gross_CO2": arr_total_gross,
        "Biomass_CO2": arr_biomass,
        "Soil_CO2": arr_soil,
        "Baseline": arr_baseline,
        "Issuable_Credit": arr_issuable,
        "Estimated_Revenue_Cum": arr_issuable * final_price # 단순 누적 매출 추정
    })
    st.dataframe(df_result, use_container_width=True)
    
    csv_data = df_result.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="CSV 파일 다운로드",
        data=csv_data,
        file_name="zigubon_forest_simulation.csv",
        mime="text/csv"
    )

# ==============================================================================
# 7. 방법론 및 설명 섹션 (Documentation Layer)
# ==============================================================================
# 사용자 요청 HTML/CSS 디자인 그대로 적용

st.markdown("""
<hr style="margin-top: 50px; margin-bottom: 30px; border-top: 1px solid #ddd;">

<style>
    /* 1. 기본 컨테이너 스타일 */
    .nbs-container {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 30px;
        color: #333;
        font-family: 'Pretendard', 'Apple SD Gothic Neo', sans-serif;
        line-height: 1.6;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
    }

    /* 2. 제목 스타일 */
    .nbs-header {
        color: #2C3E50;
        margin-top: 0;
        border-bottom: 2px solid #27AE60;
        padding-bottom: 12px;
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .nbs-sub-header {
        font-size: 1.1rem;
        margin-bottom: 15px;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* 3. 플렉스 박스 (가로 배치용) */
    .nbs-flex-wrapper {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        margin-top: 15px;
    }

    /* 4. 카드 박스 공통 스타일 */
    .nbs-card {
        flex: 1;
        min-width: 280px;
        background: #f8f9fa;
        border: 1px solid #eee;
        padding: 20px;
        border-radius: 10px;
        transition: transform 0.2s;
    }
    .nbs-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }

    /* 5. 태그 스타일 */
    .nbs-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 5px;
        margin-bottom: 5px;
    }
    .tag-green { background: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; }
    .tag-blue { background: #e3f2fd; color: #1565c0; border: 1px solid #bbdefb; }
    .tag-orange { background: #fff3e0; color: #e65100; border: 1px solid #ffe0b2; }

    /* 모바일 대응 */
    @media (max-width: 768px) {
        .nbs-container { padding: 20px; }
        .nbs-flex-wrapper { flex-direction: column; }
    }
</style>

<div class="nbs-container">
    
    <h3 class="nbs-header">
        🧬 1. 분석 방법론 (Methodology)
    </h3>
    <p style="color: #666; margin-bottom: 25px;">
        본 시뮬레이터는 <strong>국립산림과학원(NIFOS)</strong> 표준 데이터와 <strong>싱가포르 지수(CBI)</strong>를 기반으로 
        산림의 탄소 흡수량과 생태적 가치를 정량적으로 분석합니다.
    </p>

    <div class="nbs-flex-wrapper">
        <div class="nbs-card" style="border-top: 4px solid #27AE60;">
            <h4 style="margin:0 0 10px 0; color:#27AE60; font-size:1.1rem;">🌲 탄소 흡수 (Carbon)</h4>
            <ul style="font-size: 0.9rem; padding-left: 20px; color: #555; margin-bottom:0;">
                <li><strong>FBDC 모델:</strong> 현실림 임분수확표 기반 보간법(Interpolation) 적용</li>
                <li><strong>저장고 확장:</strong> 입목 바이오매스 + <span class="nbs-tag tag-green">토양/낙엽/고사목</span> 포함</li>
                <li><strong>추가성 검증:</strong> 베이스라인(무관리) 대비 순흡수량 산출</li>
            </ul>
        </div>

        <div class="nbs-card" style="border-top: 4px solid #2980B9;">
            <h4 style="margin:0 0 10px 0; color:#2980B9; font-size:1.1rem;">🦋 생물다양성 (Biodiversity)</h4>
            <ul style="font-size: 0.9rem; padding-left: 20px; color: #555; margin-bottom:0;">
                <li><strong>CBI 지수 적용:</strong> 도시생물다양성지수(Singapore Index) 기반 평가</li>
                <li><strong>자생종 가중치:</strong> <span class="nbs-tag tag-blue">Native Species</span> 비율에 따른 점수화</li>
                <li><strong>연결성 평가:</strong> 생태 네트워크 연결 수준 반영</li>
            </ul>
        </div>
    </div>

    <h3 class="nbs-header" style="margin-top: 40px; border-bottom-color: #E67E22;">
        💰 2. 재무 분석 모델 (Financial Engine)
    </h3>
    
    <div style="background: #fff8e1; border: 1px solid #ffe0b2; padding: 20px; border-radius: 8px; margin-top: 15px;">
        <strong style="color: #d35400; font-size: 1rem;">📊 ROI 및 수익성 산출 로직</strong>
        <div style="display: flex; gap: 20px; margin-top: 10px; align-items: center; flex-wrap: wrap;">
            <div style="flex: 1;">
                <p style="font-size: 0.9rem; margin: 0; color: #555;">
                    <strong>① 순 크레딧(Net Credit)</strong><br>
                    = 총 흡수량 × (1 - <span class="nbs-tag tag-orange">Buffer Risk %</span>)
                </p>
            </div>
            <div style="font-size: 1.5rem; color: #bbb;">➜</div>
            <div style="flex: 1;">
                <p style="font-size: 0.9rem; margin: 0; color: #555;">
                    <strong>② 순수익(Net Profit)</strong><br>
                    = (순 크레딧 × 예상 가격) - (조성비 + 관리비)
                </p>
            </div>
        </div>
    </div>

    <h3 class="nbs-header" style="margin-top: 40px; border-bottom-color: #34495E;">
        🛠️ 3. 기술 스택 및 핵심 알고리즘 (Tech Spec)
    </h3>
    
    <div style="background-color: #f1f3f5; padding: 20px; border-radius: 8px; margin-bottom: 25px;">
        <strong style="color: #2C3E50; font-size: 1rem; display: block; margin-bottom: 10px;">💻 Architecture: Serverless Wasm</strong>
        <p style="font-size: 0.9rem; color: #555; margin-bottom: 10px;">
            본 시뮬레이터는 <strong>Pyodide</strong> 엔진을 통해 브라우저 내에서 Python을 직접 실행하는 
            <strong>Client-side Computing</strong> 기술을 적용했습니다. 별도의 서버 통신 없이 즉각적인 연산이 가능합니다.
        </p>
        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            <span class="nbs-tag" style="background:#306998; color:white;">Python 3.11</span>
            <span class="nbs-tag" style="background:#FF4B4B; color:white;">Stlite (Streamlit)</span>
            <span class="nbs-tag" style="background:#150458; color:white;">NumPy/Pandas</span>
            <span class="nbs-tag" style="background:#3F4F75; color:white;">SciPy (Interpolation)</span>
            <span class="nbs-tag" style="background:#8e44ad; color:white;">Plotly JS</span>
        </div>
    </div>

    <div class="nbs-flex-wrapper">
        <div class="nbs-card">
            <strong style="color: #2980b9;">📐 생장 예측 알고리즘 (Interpolation)</strong>
            <p style="font-size: 0.85rem; color: #666; margin: 5px 0 10px;">
                5년 단위 표준 데이터를 <strong>선형 보간법(Linear Interpolation)</strong>으로 재구성하여 연 단위 시계열 데이터를 생성합니다.
            </p>
            <div style="background: #fff; padding: 8px; border: 1px solid #ddd; border-radius: 4px; font-family: monospace; font-size: 0.8rem; color: #333;">
                f = interp1d(x_5yr, y_carbon)<br>
                y_annual = f(years_1_to_50)
            </div>
        </div>

        <div class="nbs-card">
            <strong style="color: #c0392b;">💰 경제성 분석 알고리즘 (ROI Model)</strong>
            <p style="font-size: 0.85rem; color: #666; margin: 5px 0 10px;">
                순수익(Net Profit)과 투자 비용을 기반으로 <strong>투자대비수익률(ROI)</strong>을 실시간 산출합니다.
            </p>
            <div style="background: #fff; padding: 8px; border: 1px solid #ddd; border-radius: 4px; font-family: monospace; font-size: 0.8rem; color: #333;">
                ROI (%) = (Net_Profit / Total_Cost) * 100<br>
                <span style="color:#888;">* Net_Profit = (Credits × Price) - Cost</span>
            </div>
        </div>

        <div class="nbs-card">
            <strong style="color: #27ae60;">⚖️ CBI 생태 가치 알고리즘</strong>
            <p style="font-size: 0.85rem; color: #666; margin: 5px 0 10px;">
                싱가포르 지수(CBI) 방법론을 적용하여 <strong>자생종 비율</strong>과 <strong>연결성</strong>을 가중 평균하여 지수화합니다.
            </p>
            <div style="background: #fff; padding: 8px; border: 1px solid #ddd; border-radius: 4px; font-family: monospace; font-size: 0.8rem; color: #333;">
                Score = (Native_Ratio + Conn_Score + Diversity) / 3
            </div>
        </div>
    </div>

    <div style="text-align: right; font-size: 0.85rem; color: #999; margin-top: 25px; border-top: 1px solid #eee; padding-top: 15px;">
        Powered by <strong>Zigubon Lab</strong> | Data Sources: NIFOS, CBD
    </div>

</div>
""", unsafe_allow_html=True)
