import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os

# 1. 데이터 로드
try:
    df_forest = pd.read_csv('forest_data_2026.csv')
    df_price = pd.read_csv('carbon_price_scenarios.csv')
    df_benefit = pd.read_csv('co_benefits.csv')
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# 2. 시뮬레이션 로직 (매일 자동 계산되는 시나리오)
# 가정: 모든 수종을 각각 1ha씩 심었을 때의 포트폴리오 효과 분석
years_cols = [col for col in df_forest.columns if 'co2_yr_' in col]
years_cols.sort(key=lambda x: int(x.split('_')[2])) # 연도순 정렬
years_int = [int(col.split('_')[2]) + 2026 for col in years_cols] # 실제 연도 (2026, 2031...)

# 전체 수종 합계 계산
total_uptake = df_forest[years_cols].sum(axis=0).values

# 경제적 가치 (2026년 기준 베이스 시나리오)
base_price_2026 = df_price.loc[df_price['year'] == 2026, 'price_base'].values[0]
estimated_value = total_uptake[-1] * base_price_2026 # 50년 누적 가치

# ESG 지수 평균
avg_bio = df_benefit['biodiversity_index'].mean()
avg_water = df_benefit['water_index'].mean()

# 3. 차트 생성 (Plotly)
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=years_int,
    y=total_uptake,
    mode='lines+markers',
    name='누적 탄소 흡수량',
    line=dict(color='#145A32', width=4),
    marker=dict(size=8)
))

fig.update_layout(
    title="🌲 모든 수종 혼효 식재 시 예상 탄소 흡수량 (포트폴리오)",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Pretendard, sans-serif"),
    xaxis=dict(title="연도"),
    yaxis=dict(title="누적 흡수량 (tCO2)")
)

chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
today_date = datetime.now().strftime("%Y-%m-%d")

# 4. HTML 리포트 생성
html_template = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Forest MRV Daily Report</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background-color: #f8f9fa; margin: 0; padding: 20px; color: #333; }}
        .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 40px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); }}
        .header {{ text-align: center; margin-bottom: 40px; border-bottom: 2px solid #f1f1f1; padding-bottom: 20px; }}
        .header h1 {{ margin: 0; color: #145A32; font-size: 1.8rem; letter-spacing: -0.5px; }}
        .badge {{ background: #e8f5e9; color: #145A32; padding: 5px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: bold; }}
        
        .kpi-container {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 30px; }}
        .kpi-card {{ background: #fff; border: 1px solid #eee; padding: 20px; border-radius: 12px; text-align: center; transition: transform 0.2s; }}
        .kpi-card:hover {{ transform: translateY(-5px); box-shadow: 0 5px 15px rgba(0,0,0,0.05); border-color: #145A32; }}
        .kpi-icon {{ font-size: 1.5rem; color: #27ae60; margin-bottom: 10px; }}
        .kpi-value {{ font-size: 1.5rem; font-weight: 800; color: #2c3e50; margin: 5px 0; }}
        .kpi-label {{ font-size: 0.85rem; color: #888; }}

        .chart-box {{ border: 1px solid #eee; border-radius: 12px; padding: 10px; margin-top: 20px; }}
        .footer {{ text-align: center; margin-top: 50px; font-size: 0.8rem; color: #aaa; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <span class="badge">Daily Update</span>
            <h1>🌲 Forest MRV Analysis</h1>
            <p style="color:#666; margin-top:10px;">자동 생성 리포트 • {today_date}</p>
        </div>

        <div class="kpi-container">
            <div class="kpi-card">
                <div class="kpi-icon"><i class="fa-solid fa-tree"></i></div>
                <div class="kpi-label">총 예상 흡수량 (50년)</div>
                <div class="kpi-value">{total_uptake[-1]:,.0f} <span style="font-size:1rem">tCO₂</span></div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon"><i class="fa-solid fa-coins"></i></div>
                <div class="kpi-label">예상 경제적 가치</div>
                <div class="kpi-value">₩{estimated_value/100000000:.1f} <span style="font-size:1rem">억</span></div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon"><i class="fa-solid fa-leaf"></i></div>
                <div class="kpi-label">평균 ESG 지수</div>
                <div class="kpi-value">{avg_bio:.1f} <span style="font-size:1rem">/ 5.0</span></div>
            </div>
        </div>

        <div class="chart-box">
            {chart_html}
        </div>

        <div class="footer">
            Data Source: NIFOS Standard (2026) • Powered by ZIGUBON & GitHub Actions
        </div>
    </div>
</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_template)
