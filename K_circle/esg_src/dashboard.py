import os
import pandas as pd
import streamlit as st
import plotly.express as px

# =========================
# 0) 데이터 로드
# =========================
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(BASE_PATH, "results.csv")

df = pd.read_csv(RESULTS_PATH, encoding="utf-8-sig")

# 숫자형 변환
for col in ["year", "esg_avg", "esg_last", "stock_price", "market_cap", "debt"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================
# 1) 헤더
# =========================
st.title("📊 ESG 기업 분석 대시보드")
st.markdown("데모 버전 made by 비비빅")
st.write("")

# =========================
# 2) 사이드바 (Filters)
# =========================
st.sidebar.header("⚙️ 필터")

# 기업 선택
company = st.sidebar.selectbox("기업 선택", df["company"].unique())

# 기업 비교용
companies = df["company"].unique().tolist()
if len(companies) >= 2:
    c1 = st.sidebar.selectbox("비교 기업 A", companies, index=0, key="cmpA")
    c2 = st.sidebar.selectbox("비교 기업 B", companies, index=1, key="cmpB")
else:
    c1, c2 = None, None

# =========================
# 3) Top 5 ESG 기업
# =========================
st.subheader("🏆 Top 5 ESG 기업")
top5 = (
    df.groupby("company", as_index=False)["esg_avg"]
      .mean()
      .nlargest(5, "esg_avg")
)
st.bar_chart(top5.set_index("company")["esg_avg"])

# =========================
# 4) Company Details
# =========================
st.subheader(f"📌 Company Details: {company}")
latest = df[df["company"] == company].sort_values("year").iloc[-1]

if "market_cap" in df.columns:
    st.metric("시가총액 (단위: 조원)", f"{latest['market_cap']}")
if "debt" in df.columns:
    st.metric("부채비율 (%)", f"{latest['debt']}")

# ESG 추세
st.subheader(f"{company} ESG 점수 추세")
esg_trend = (
    df[df["company"] == company]
      .groupby("year")["esg_last"]
      .mean()
      .sort_index()
)
st.line_chart(esg_trend)

# 주가 추세
st.subheader(f"{company} 주가 추이")
stock_trend = (
    df[df["company"] == company]
      .groupby("year")["stock_price"]
      .mean()
      .sort_index()
)
st.line_chart(stock_trend)

# =========================
# 5) Comparison (Radar Chart)
# =========================
st.subheader("📊 기업 비교 (Radar Chart)")
if c1 and c2:
    def _agg_one(name):
        sub = df[df["company"] == name]
        return {
            "ESG 평균": sub["esg_avg"].mean(),
            "ESG 최신": sub["esg_last"].mean(),
            "문서 수": sub["n_docs"].mean(),
        }

    a = _agg_one(c1)
    b = _agg_one(c2)

    radar_df = pd.DataFrame({
        "Category": ["ESG 평균", "ESG 최신", "문서 수"],
        c1: [a["ESG 평균"], a["ESG 최신"], a["문서 수"]],
        c2: [b["ESG 평균"], b["ESG 최신"], b["문서 수"]],
    })

    fig = px.line_polar(radar_df, r=c1, theta="Category", line_close=True)
    fig.add_scatterpolar(r=radar_df[c2], theta=radar_df["Category"], name=c2)
    st.plotly_chart(fig, use_container_width=True)

# =========================
# 6) 기업 추천
# =========================
st.subheader("✅ 추천 기업")
col1, col2, col3 = st.columns(3)
top3 = top5.head(3)

for i, col in enumerate([col1, col2, col3]):
    if i < len(top3):
        row = top3.iloc[i]
        with col:
            st.metric(row["company"], round(row["esg_avg"], 2), "ESG 평균 점수")
