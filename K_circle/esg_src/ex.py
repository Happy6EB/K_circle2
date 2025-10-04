import os
import pandas as pd
import streamlit as st
import plotly.express as px



# =========================
# 0) 데이터 로드 (results.csv만 사용)
# =========================
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(BASE_PATH, "results.csv")

# CSV 읽기
df = pd.read_csv(RESULTS_PATH, encoding="utf-8-sig")

# 컬럼 이름 점검 (디버그용) - 필요 없으면 주석 처리해도 됨
# st.write("columns:", list(df.columns))

# 숫자형 변환 (문자열로 들어온 경우 대비)
for col in ["year", "esg_avg", "esg_last", "stock_price", "market_cap", "debt"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================
# 1) 헤더
# =========================
st.title("📊 ESG 기업 분석 대시보드")
st.markdown("데모 버전 made by 비비빅")
st.write("")
st.write("")

# =========================
# 2) Top 5 ESG 기업 (esg_avg 평균 기준)
# =========================
st.subheader("Top 5 ESG 기업")
top5 = (
    df.groupby("company", as_index=False)["esg_avg"]
      .mean()
      .nlargest(5, "esg_avg")
)
st.bar_chart(top5.set_index("company")["esg_avg"])

# =========================
# 3) Company Details (results.csv 기반)
# =========================
company = st.selectbox("기업 선택", df["company"].unique())
st.subheader(f"Company Details: {company}")

# 선택 기업 데이터 중 최신 연도 한 줄
latest = (
    df[df["company"] == company]
    .sort_values("year")
    .iloc[-1]
)

# 시가총액 & 부채비율 (csv 값 그대로 표기)
if "market_cap" in df.columns:
    st.metric("시가총액 (단위: 조원)", f"{latest['market_cap']}")
if "debt" in df.columns:
    st.metric("부채비율 (%)", f"{latest['debt']}")

# ESG 연도별 추세
st.subheader(f"{company} ESG 점수 추세")
esg_trend = (
    df[df["company"] == company]
      .groupby("year")["esg_last"]
      .mean()
      .sort_index()
)
st.line_chart(esg_trend)

# 주가 연도별 추세
st.subheader(f"{company} 주가 추이")
price_trend = (
    df[df["company"] == company]
      .groupby("year")["stock_price"]
      .mean()
      .sort_index()
)
st.line_chart(price_trend)

# =========================
# 4) 기업 비교 (레이더 차트 예시) - esg_avg / esg_last / n_docs 비교
# =========================
st.subheader("Comparison between companies")
companies = df["company"].unique().tolist()
if len(companies) >= 2:
    c1 = st.selectbox("비교 기업 A", companies, index=0, key="cmpA")
    c2 = st.selectbox("비교 기업 B", companies, index=1, key="cmpB")

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
else:
    st.info("비교를 위해서는 최소 2개 기업 데이터가 필요합니다.")
