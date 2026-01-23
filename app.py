import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import interp1d

# ==============================================================================
# 1. 시스템 설정 및 디자인 (CSS)
# ==============================================================================
st.set_page_config(
    page_title="ZIGUBON | Forest Carbon & ESG Simulator",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 통합 스타일시트 정의
st.markdown("""
    <style>
    /* 전체 배경 및 폰트 */
    .main { background-color: #f8f9fa; font-family: 'Pretendard', 'Apple SD Gothic Neo', sans-serif; }
    
    /* KPI 카드 스타일 */
    div[data-testid="stMetricValue"] { font-size: 26px; color: #145A32; font-weight: 800; }
    div[data-testid="stMetricLabel"] { font-size: 14px; color: #555; font-weight: 600; }
    div[data-testid="stCard"] { 
        background-color: white; 
        border-radius: 12px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.04); 
        padding: 1rem;
    }

    /* 탭 및 익스팬더 스타일 */
    .stExpander { border: 1px solid #e0e0e0; border-radius: 8px; background: white; }
    
    /* 하단 설명 섹션 컨테이너 (통합 스타일) */
    .nbs-footer-container {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 40px;
        margin-top: 50px;
        color: #333;
        box-shadow: 0 10px 30px rgba(0,0,0,0.03);
    }

    /* 섹션 헤더 */
    .nbs-section-header {
        font-size: 1.5rem;
        font-weight: 800;
        color: #2c3e50;
        border-bottom: 2px solid #ddd;
        padding-bottom: 15px;
        margin-bottom: 25px;
        margin-top: 40px;
        letter-spacing: -0.5px;
    }
    .nbs-header-green { border-bottom-color: #27ae60; color: #27ae60; }
    .nbs-header-blue { border-bottom-color: #2980b9; color: #2980b9; }
    .nbs-header-orange { border-bottom-color: #d35400; color: #d35400; }
    .nbs-header-dark { border-bottom-color: #34495e; color: #34495e; }

    /* 서브 헤더 */
    .nbs-sub-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #555;
        margin-bottom: 15px;
        display: flex; align-items: center; gap: 8px;
    }

    /* 카드 그리드 시스템 */
    .nbs-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 25px;
        margin-bottom: 30px;
    }

    /* 정보 카드 */
    .nbs-info-card {
        background: #f8f9fa;
        border: 1px solid #eee;
        border-radius: 10px;
        padding: 25px;
        transition: all 0.2s ease;
    }
    .nbs-info-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        border-color: #ddd;
    }

    /* 태그 스타일 */
    .nbs-tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .tag-green { background: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; }
    .tag-blue { background: #e3f2fd; color: #1565c0; border: 1px solid #bbdefb; }
    .tag-orange { background: #fff3e0; color: #e65100; border: 1px solid #ffe0b2; }
    .tag-gray { background: #f1f3f5; color: #495057; border: 1px solid #dee2e6; }
    .tag-tech { background: #343a40; color: #fff; border: 1px solid #343a40; }

    /* 수식 박스 */
    .nbs-formula {
        font-family: 'Consolas', 'Monaco', monospace;
        background: #fff;
        padding: 15px;
        border-radius: 6px;
        border: 1px solid #e9ecef;
        color: #c0392b;
        font-size: 0.9rem;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. 데이터 로드 및 전처리
# ==============================================================================
@st.cache_data
def load_data():
    """
    index.html에서 가상 파일 시스템으로 주입된 CSV 파일들을 로드합니다.
    로컬 환경 테스트 시 파일 부재에 대한 예외 처리를 포함합니다.
    """
    try:
        forest = pd.read_csv("forest_data_2026.csv")
        price = pd.read_csv("carbon_price_scenarios.csv")
        benefit = pd.read_csv("co_benefits.csv")
        return forest, price, benefit
    except Exception as e:
        return None, None, None

df_forest, df_price, df_benefit = load_data()

if df_forest is None:
    st.error("🚨 중요: 데이터 파일을 찾을 수 없습니다. GitHub 저장소에 CSV 파일이 업로드되어 있는지 확인해주세요.")
    st.stop()

# ==============================================================================
# 3. 사이드바 (입력 제어 패널)
# ==============================================================================
with st.sidebar:
    st.title("🌲 시뮬레이션 설정")
    st.markdown("---")
    
    # [섹션 1] 기본 사업 개요
    st.subheader("1️⃣ 사업 개요 (Project Basics)")
    area = st.number_input("사업 면적 (ha)", min_value=1.0, value=50.0, step=1.0, help="전체 사업 대상지의 면적입니다.")
    project_period = st.slider("사업 기간 (년)", 5, 50, 30, help="탄소 흡수량을 산정할 전체 사업 기간입니다.")
    
    st.markdown("---")
    
    # [섹션 2] 수종 및 포트폴리오 (비율 조정 기능)
    st.subheader("2️⃣ 수종 포트폴리오 (Species Mix)")
    species_list = df_forest['name'].unique()
    # 기본값: 데이터가 있으면 상위 2개 자동 선택
    default_sp = [species_list[0], species_list[1]] if len(species_list) > 1 else [species_list[0]]
    selected_species = st.multiselect("식재 수종 선택", species_list, default=default_sp)
    
    if not selected_species:
        st.warning("⚠️ 최소 1개 이상의 수종을 선택해주세요.")
        st.stop()
    
    # 수종별 점유 비율 슬라이더 생성
    species_ratios = {}
    if len(selected_species) > 1:
        st.info("👇 수종별 점유 비율(%)을 설정하세요")
        total_ratio = 0
        for i, sp in enumerate(selected_species):
            # 기본 비율 균등 배분
            default_val = int(100 / len(selected_species))
            # 마지막 수종은 남은 비율을 자동 계산하면 좋겠지만, Streamlit UI 한계상 사용자 조절 유도
            ratio = st.slider(f"{sp} 비율", 0, 100, default_val, key=f"ratio_{sp}")
            species_ratios[sp] = ratio / 100.0
            total_ratio += ratio
        
        if total_ratio != 100:
            st.error(f"⚠️ 현재 비율 합계: {total_ratio}% (100%에 맞춰주세요)")
    else:
        species_ratios[selected_species[0]] = 1.0

    st.markdown("---")

    # [섹션 3] 생태 연결성 (CBI) & 밀도
    st.subheader("3️⃣ 생태 및 기술 요소 (Tech & Bio)")
    
    # CBI 지표 2번: 연결성 평가
    connectivity_score = st.select_slider(
        "생태 연결성 (Connectivity)",
        options=["고립 (낮음)", "일부 연결 (보통)", "핵심 축 연결 (높음)"],
        value="일부 연결 (보통)",
        help="대상지가 백두대간 등 주요 생태축과 연결되어 있는지 평가합니다 (CBI 지표)."
    )
    conn_map = {"고립 (낮음)": 1.0, "일부 연결 (보통)": 3.0, "핵심 축 연결 (높음)": 5.0}
    conn_value = conn_map[connectivity_score]
    
    # 식재 밀도
    density_factor = st.slider("식재 밀도 지수 (%)", 50, 150, 100, help="표준 식재본수(3,000본/ha) 대비 밀도입니다. 100%가 표준입니다.") / 100.0
    estimated_trees = int(area * 3000 * density_factor)
    st.caption(f"🌱 총 추정 식재 본수: {estimated_trees:,} 본")

    st.markdown("---")
    
    # [섹션 4] 재무 및 리스크 (Financial)
    st.subheader("4️⃣ 재무 및 리스크 (Financials)")
    
    # 리스크 버퍼
    buffer_ratio = st.slider("리스크 버퍼 (Buffer %)", 0, 30, 15, help="산불 등 영구 손실에 대비해 유보하는 크레딧 비율입니다.") / 100.0
    
    # 비용 입력
    c1, c2 = st.columns(2)
    with c1:
        initial_cost_per_ha = st.number_input("초기 조성비 (만원/ha)", value=1500, step=100)
    with c2:
        annual_cost_per_ha = st.number_input("연 관리비 (만원/ha)", value=50, step=10)
    
    discount_rate = 0.045 # 사회적 할인율 4.5% 가정 (NPV 계산용)

    st.markdown("---")

    # [섹션 5] 탄소 가격 시나리오
    st.subheader("5️⃣ 시장 전망 (Market View)")
    price_scenario = st.selectbox("탄소배출권 가격 전망", ["Base (기준)", "High (낙관)", "Low (보수)"])
    price_col_map = {"Base (기준)": "price_base", "High (낙관)": "price_high", "Low (보수)": "price_low"}
    price_col = price_col_map[price_scenario]


# ==============================================================================
# 4. 시뮬레이션 계산 엔진 (Physics & Financial Engine)
# ==============================================================================

# 자생종 확인 함수 (CBI 지표용)
def check_native(name):
    native_keywords = ["소나무", "상수리", "신갈", "졸참", "굴참", "잣나무", "느티나무"] 
    return any(k in name for k in native_keywords)

# 시간축 생성
years = list(range(2026, 2026 + project_period + 1))

# 결과 저장용 배열 초기화
total_biomass_carbon = np.zeros(project_period + 1)
total_soil_carbon = np.zeros(project_period + 1)

# CBI 점수 집계용 변수
total_native_ratio = 0
weighted_water_score = 0

# --- [Core Loop] 수종별 계산 ---
for sp in selected_species:
    sp_row = df_forest[df_forest['name'] == sp].iloc[0]
    ratio = species_ratios[sp]
    
    # A. 탄소 흡수량 보간 (Interpolation)
    x_points = list(range(0, 51, 5)) # 0, 5, 10 ... 50년
    y_points = [sp_row[f'co2_yr_{y}'] for y in x_points]
    
    # 선형 보간 함수 생성 (연 단위 데이터 생성)
    f_interp = interp1d(x_points, y_points, kind='linear', fill_value="extrapolate")
    standard_uptake_per_ha = f_interp(range(project_period + 1))
    
    # B. 면적 및 밀도 적용 (Physical scaling)
    real_area = area * ratio
    adjusted_uptake = standard_uptake_per_ha * real_area * density_factor
    
    # C. 토양 탄소 추정 (Tier 1 확장계수법: 바이오매스의 35% 가정)
    soil_uptake = adjusted_uptake * 0.35
    
    # 합산
    total_biomass_carbon += adjusted_uptake
    total_soil_carbon += soil_uptake
    
    # D. CBI 가중치 계산
    if check_native(sp):
        total_native_ratio += ratio * 100
    
    # 공편익 데이터 매핑
    try:
        ben_row = df_benefit.iloc[sp_row['id']-1] # ID 매핑
        weighted_water_score += ben_row['water_index'] * ratio
    except:
        weighted_water_score += 3.0 * ratio # 데이터 없을 시 기본값

# --- [Credit Logic] 순 감축량 계산 ---
total_project_carbon = total_biomass_carbon + total_soil_carbon
baseline_carbon = total_project_carbon * 0.7 # 베이스라인 (무관리 시 70% 가정)
gross_credit = total_project_carbon - baseline_carbon # 총 감축량 (Gross)
buffer_amount = gross_credit * buffer_ratio # 버퍼 차감
net_issuable_credit = gross_credit - buffer_amount # 발행 가능 크레딧 (Net)

# --- [Financial Logic] ROI & NPV 계산 ---
# 1. 비용 흐름 (Cost Flow)
total_initial_cost = initial_cost_per_ha * area * 10000 # 원 단위
annual_cost_year = annual_cost_per_ha * area * 10000
total_cost_nominal = total_initial_cost + (annual_cost_year * project_period)

# 2. 수익 흐름 (Revenue Flow)
# 가격 데이터 매핑 (연도별 가격 적용)
revenue_stream = []
cost_stream = []
net_cash_flow = []

cost_stream.append(total_initial_cost) # 0년차 초기비용
net_cash_flow.append(-total_initial_cost) 

cumulative_net_credit = 0

for i, yr in enumerate(years):
    if i == 0: continue # 0년차는 초기비용만
    
    # 해당 연도 발생 크레딧 (누적 차이)
    annual_credit = net_issuable_credit[i] - net_issuable_credit[i-1]
    
    # 해당 연도 가격
    if yr > df_price['year'].max():
        curr_price = df_price.iloc[-1][price_col]
    else:
        curr_price = df_price[df_price['year'] == yr][price_col].values[0]
        
    rev = annual_credit * curr_price
    revenue_stream.append(rev)
    cost_stream.append(annual_cost_year)
    
    net_flow = rev - annual_cost_year
    net_cash_flow.append(net_flow)

total_revenue_nominal = sum(revenue_stream)
net_profit_nominal = total_revenue_nominal - total_cost_nominal

# ROI (단순 수익률)
roi = (net_profit_nominal / total_cost_nominal) * 100 if total_cost_nominal > 0 else 0

# NPV (순현재가치)
npv = -total_initial_cost
for t, flow in enumerate(net_cash_flow[1:], start=1): # 1년차부터 할인
    npv += flow / ((1 + discount_rate) ** t)


# --- [ESG Logic] CBI 점수 산출 ---
cbi_native_score = (total_native_ratio / 100.0) * 5.0
cbi_water_score = weighted_water_score
cbi_conn_score = conn_value
cbi_diversity_score = min(5.0, 2.0 + (len(selected_species) * 0.6)) # 혼효림 가산점

# 경제성 점수 (ROI 연동)
if roi <= 0: cbi_econ_score = 1.0
elif roi >= 200: cbi_econ_score = 5.0
else: cbi_econ_score = 1.0 + (roi / 50.0)

# 종합 점수
final_cbi_score = (cbi_native_score + cbi_water_score + cbi_conn_score + cbi_diversity_score + cbi_econ_score) / 5.0


# ==============================================================================
# 5. 메인 대시보드 UI
# ==============================================================================
forest_type = "혼효림 (Mixed Forest)" if len(selected_species) > 1 else "단순림 (Monoculture)"
st.title(f"🌲 {forest_type} 사업성 분석 시뮬레이터")
st.markdown(f"**{area}ha** 면적 / **{project_period}년** 사업 / **{', '.join(selected_species)}** 식재 시나리오 분석")

# [KPI Metrics]
col1, col2, col3, col4 = st.columns(4)
final_credit = net_issuable_credit[-1]
cars_offset = (final_credit / project_period) / 2.43

with col1:
    st.metric("순 발행 크레딧 (Net Credit)", f"{final_credit:,.0f} tCO₂", f"버퍼 {int(buffer_ratio*100)}% 차감됨")
with col2:
    st.metric("예상 순수익 (Net Profit)", f"₩{net_profit_nominal/100000000:.1f} 억", f"ROI {roi:.1f}%")
with col3:
    st.metric("순현재가치 (NPV)", f"₩{npv/100000000:.1f} 억", f"할인율 {discount_rate*100}% 적용")
with col4:
    st.metric("CBI 종합 등급", f"{final_cbi_score:.1f} / 5.0", f"생물다양성+경제성")

st.markdown("---")

# [Charts Layout]
c_main, c_sub = st.columns([2, 1])

# 왼쪽: 탄소 차트
with c_main:
    st.subheader("📊 탄소 저장 및 추가성 분석 (Additionality)")
    fig = go.Figure()
    
    # 1. 입목
    fig.add_trace(go.Scatter(x=years, y=total_biomass_carbon, mode='lines', name='🌲 입목 바이오매스', stackgroup='one', line=dict(width=0, color='#27ae60')))
    # 2. 토양
    fig.add_trace(go.Scatter(x=years, y=total_soil_carbon, mode='lines', name='🟤 토양/기타 저장고', stackgroup='one', line=dict(width=0, color='#8d6e63')))
    # 3. 베이스라인
    fig.add_trace(go.Scatter(x=years, y=baseline_carbon, mode='lines', name='📉 베이스라인 (무관리)', line=dict(color='#34495e', width=2, dash='dash')))
    
    fig.update_layout(
        xaxis_title="연도", yaxis_title="누적 흡수량 (tCO₂)",
        height=400, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5), # 범례 상단 이동
        margin=dict(t=30)
    )
    st.plotly_chart(fig, use_container_width=True)

# 오른쪽: CBI 레이더 차트
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
    
    with st.expander("💡 CBI 점수 상세 해석"):
        st.write(f"- **자생종:** {total_native_ratio:.0f}% 구성")
        st.write(f"- **수익성:** ROI {roi:.1f}% 달성")
        st.write(f"- **연결성:** {connectivity_score}")

# [Data Download]
with st.expander("📥 상세 데이터 테이블 다운로드"):
    df_res = pd.DataFrame({
        "Year": years,
        "Total_Carbon": total_project_carbon,
        "Baseline": baseline_carbon,
        "Net_Credit": net_issuable_credit,
        "Cumulative_Cash_Flow": np.cumsum(net_cash_flow[1:]) # 현금흐름 누적
    })
    st.dataframe(df_res, use_container_width=True)
    st.download_button("CSV 다운로드", df_res.to_csv(index=False).encode('utf-8-sig'), "simulation_report.csv")


# ==============================================================================
# 6. 하단 통합 설명 섹션 (Unified Footer)
# ==============================================================================
st.markdown(f"""
<div class="nbs-footer-container">

    <h3 class="nbs-section-header nbs-header-green">
        🧬 1. 분석 방법론 (Methodology)
    </h3>
    <p style="margin-bottom: 25px;">
        본 시뮬레이터는 <strong>국립산림과학원(NIFOS)</strong> 표준 데이터와 <strong>싱가포르 지수(CBI)</strong>를 기반으로 
        산림의 탄소 흡수량과 생태적 가치를 정량적으로 분석합니다.
    </p>

    <div class="nbs-grid">
        <div class="nbs-info-card">
            <h4 style="color:#27AE60; margin-top:0;">🌲 탄소 흡수 (Carbon)</h4>
            <div style="margin-top:10px;">
                <span class="nbs-tag tag-green">FBDC 모델</span>
                <span class="nbs-tag tag-green">Tier 2</span>
            </div>
            <ul style="font-size: 0.9rem; margin-top: 15px; padding-left: 20px; color: #555;">
                <li><strong>성장 예측:</strong> 현실림 임분수확표 기반 보간(Interpolation)</li>
                <li><strong>저장고 확장:</strong> 입목 바이오매스 + <span style="background:#e8f5e9; padding:0 4px;">토양/낙엽/고사목 (35% 가산)</span></li>
                <li><strong>추가성(Additionality):</strong> 무관리 베이스라인 대비 순흡수량 산출</li>
            </ul>
        </div>

        <div class="nbs-info-card">
            <h4 style="color:#2980B9; margin-top:0;">🦋 생물다양성 (Biodiversity)</h4>
            <div style="margin-top:10px;">
                <span class="nbs-tag tag-blue">CBI Index</span>
                <span class="nbs-tag tag-blue">Singapore Index</span>
            </div>
            <ul style="font-size: 0.9rem; margin-top: 15px; padding-left: 20px; color: #555;">
                <li><strong>자생종 가중치:</strong> Native Species 비율에 따른 점수화</li>
                <li><strong>생태 연결성:</strong> 주변 생태축과의 연결 수준 평가</li>
                <li><strong>공편익(Co-benefits):</strong> 수자원 함양 및 산불 저항성 반영</li>
            </ul>
        </div>
    </div>

    <h3 class="nbs-section-header nbs-header-orange">
        💰 2. 재무 분석 모델 (Financial Engine)
    </h3>
    
    <div style="background: #fff8e1; border: 1px solid #ffe0b2; padding: 25px; border-radius: 10px;">
        <div class="nbs-sub-title" style="color:#d35400;">📊 순현재가치(NPV) 및 수익성 산출 로직</div>
        <div style="display: flex; gap: 30px; align-items: flex-start; flex-wrap: wrap;">
            <div style="flex: 1;">
                <strong style="display:block; margin-bottom:5px; color:#555;">Step 1. 순 크레딧 산출</strong>
                <div class="nbs-formula">
                    Net_Credit = Total_Uptake × (1 - Buffer_Ratio)
                </div>
                <p style="font-size:0.85rem; color:#666; margin-top:5px;">* 버퍼(Buffer): {int(buffer_ratio*100)}% (산불 등 영구 손실 대비 유보)</p>
            </div>
            <div style="flex: 1;">
                <strong style="display:block; margin-bottom:5px; color:#555;">Step 2. 현금 흐름 (Cash Flow)</strong>
                <div class="nbs-formula">
                    Profit = (Credit × Price) - (Init_Cost + Ann_Cost)
                </div>
                <p style="font-size:0.85rem; color:#666; margin-top:5px;">* 할인율(Discount Rate): {discount_rate*100}% 적용 (NPV 산출 시)</p>
            </div>
        </div>
    </div>

    <h3 class="nbs-section-header nbs-header-dark">
        🛠️ 3. 기술 스택 및 알고리즘 (Tech Spec)
    </h3>

    <div style="background-color: #f1f3f5; padding: 20px; border-radius: 10px; margin-bottom: 25px;">
        <strong style="color: #2C3E50; display: block; margin-bottom: 10px;">💻 Architecture: Serverless Wasm (Pyodide)</strong>
        <p style="font-size: 0.9rem; color: #555; margin-bottom: 15px;">
            본 시뮬레이터는 브라우저 내에서 Python을 직접 실행하는 <strong>Client-side Computing</strong> 기술을 적용하여, 
            별도의 서버 통신 없이 즉각적인 재무/환경 시뮬레이션이 가능합니다.
        </p>
        <div>
            <span class="nbs-tag tag-tech">Python 3.11</span>
            <span class="nbs-tag tag-tech">Stlite (Wasm)</span>
            <span class="nbs-tag tag-tech">Pandas</span>
            <span class="nbs-tag tag-tech">SciPy</span>
            <span class="nbs-tag tag-tech">Plotly JS</span>
        </div>
    </div>

    <div class="nbs-grid">
        <div class="nbs-info-card">
            <strong style="color: #2980b9;">📐 생장 예측 (Interpolation)</strong>
            <p style="font-size: 0.85rem; color: #666; margin: 5px 0 10px;">
                5년 단위 표준 데이터를 <strong>선형 보간법(Linear Interpolation)</strong>으로 재구성하여 연 단위 시계열 데이터를 생성합니다.
            </p>
        </div>
        <div class="nbs-info-card">
            <strong style="color: #c0392b;">📈 투자대비수익률 (ROI)</strong>
            <p style="font-size: 0.85rem; color: #666; margin: 5px 0 10px;">
                순수익(Net Profit)과 총 투자 비용(Total Cost)을 기반으로 ROI를 산출하고 이를 CBI 경제성 지표로 환산합니다.
            </p>
        </div>
        <div class="nbs-info-card">
            <strong style="color: #27ae60;">⚖️ CBI 복합 지표</strong>
            <p style="font-size: 0.85rem; color: #666; margin: 5px 0 10px;">
                자생종 비율, 수자원 인덱스, 연결성, 수익성 등 이질적인 데이터를 <strong>5점 척도 정규화(Normalization)</strong>하여 레이더 차트로 시각화합니다.
            </p>
        </div>
    </div>

    <div style="text-align: right; font-size: 0.85rem; color: #999; margin-top: 30px; border-top: 1px solid #eee; padding-top: 20px;">
        Data Sources: <strong>NIFOS</strong> (National Institute of Forest Science), <strong>CBD</strong> (Convention on Biological Diversity) <br>
        Powered by <strong>ZIGUBON Lab</strong>
    </div>

</div>
""", unsafe_allow_html=True)
