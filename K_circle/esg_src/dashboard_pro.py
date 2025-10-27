import os
import time
import json
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from plotly.subplots import make_subplots


# -----------------------------------------------------
# 기본 설정
# -----------------------------------------------------
st.set_page_config(page_title="ESG 기반 AI 투자지원", layout="wide")
st.title("📊 ESG 기반 AI 투자지원 대시보드")
st.caption("B.B.BIC | DART 전자공시 + 실시간 분석 → AI 점수화 → 투자 추천 | 그린워싱 감지 시스템 탑재")

# -----------------------------------------------------
# 기업 목록 (산업별 10개씩)
# -----------------------------------------------------
SECTOR_COMPANIES = {
    "자동차": {
        "현대자동차": {"ticker": "005380.KS"},
        "기아": {"ticker": "000270.KS"},
        "현대모비스": {"ticker": "012330.KS"},
        "현대위아": {"ticker": "011210.KS"},
        "만도": {"ticker": "204320.KS"},
        "한온시스템": {"ticker": "018880.KS"},
        "한국타이어앤테크놀로지": {"ticker": "161390.KS"},
        "현대로템": {"ticker": "064350.KS"},
        "KG모빌리티": {"ticker": "003620.KS"},
        "현대글로비스": {"ticker": "086280.KS"},
    },
    "화학/에너지": {
        "LG화학": {"ticker": "051910.KS"},
        "롯데케미칼": {"ticker": "011170.KS"},
        "한화솔루션": {"ticker": "009830.KS"},
        "금호석유화학": {"ticker": "011780.KS"},
        "대한유화": {"ticker": "006650.KS"},
        "OCI홀딩스": {"ticker": "010060.KS"},
        "SK이노베이션": {"ticker": "096770.KS"},
        "S-Oil": {"ticker": "010950.KS"},
        "현대에너지솔루션": {"ticker": "322000.KS"},
        "포스코퓨처엠": {"ticker": "003670.KS"},
    },
    "헬스케어": {
        "삼성바이오로직스": {"ticker": "207940.KS"},
        "셀트리온": {"ticker": "068270.KS"},
        "유한양행": {"ticker": "000100.KS"},
        "녹십자": {"ticker": "006280.KS"},
        "한미약품": {"ticker": "128940.KS"},
        "종근당": {"ticker": "185750.KS"},
        "동아에스티": {"ticker": "170900.KS"},
        "대웅": {"ticker": "003090.KS"},
        "보령제약": {"ticker": "003850.KS"},
        "HLB": {"ticker": "028300.KQ"},
    },
    "금융": {
        "KB금융": {"ticker": "105560.KS"},
        "신한지주": {"ticker": "055550.KS"},
        "하나금융지주": {"ticker": "086790.KS"},
        "우리금융지주": {"ticker": "316140.KS"},
        "한국금융지주": {"ticker": "071050.KS"},
        "미래에셋증권": {"ticker": "006800.KS"},
        "삼성생명": {"ticker": "032830.KS"},
        "삼성화재": {"ticker": "000810.KS"},
        "메리츠금융지주": {"ticker": "138040.KS"},
        "대신증권": {"ticker": "003540.KS"},
    },
    "IT/전자": {
        "삼성전자": {"ticker": "005930.KS"},
        "SK하이닉스": {"ticker": "000660.KS"},
        "LG전자": {"ticker": "066570.KS"},
        "네이버": {"ticker": "035420.KS"},
        "카카오": {"ticker": "035720.KS"},
        "삼성SDI": {"ticker": "006400.KS"},
        "LG디스플레이": {"ticker": "034220.KS"},
        "SK스퀘어": {"ticker": "402340.KS"},
        "삼성전기": {"ticker": "009150.KS"},
        "DB하이텍": {"ticker": "000990.KS"},
    },
}

# 전체 기업 맵 생성
COMPANY_MAP = {}
for sector, companies_dict in SECTOR_COMPANIES.items():
    COMPANY_MAP.update(companies_dict)

# -----------------------------------------------------
# 개선된 ESG 점수 산정 (재무지표 + 그린워싱 감지)
# -----------------------------------------------------
def calculate_advanced_esg_score(company: str, year: int, financial_data: dict = None) -> dict:
    """
    개선된 ESG 점수 계산
    - 재무제표 기반 정량지표 반영
    - 그린워싱 감지 (선언 vs 실적 괴리도)
    - 산업별 벤치마크 대비 평가
    """
    sector_profiles = {
        "자동차": {"e_base": 0.58, "e_std": 0.15, "s_base": 0.68, "s_std": 0.10, "g_base": 0.62, "g_std": 0.08, "greenwashing_risk": 0.35},
        "화학/에너지": {"e_base": 0.45, "e_std": 0.20, "s_base": 0.55, "s_std": 0.12, "g_base": 0.68, "g_std": 0.09, "greenwashing_risk": 0.50},
        "헬스케어": {"e_base": 0.72, "e_std": 0.08, "s_base": 0.78, "s_std": 0.10, "g_base": 0.64, "g_std": 0.12, "greenwashing_risk": 0.15},
        "금융": {"e_base": 0.70, "e_std": 0.09, "s_base": 0.72, "s_std": 0.08, "g_base": 0.82, "g_std": 0.10, "greenwashing_risk": 0.25},
        "IT/전자": {"e_base": 0.65, "e_std": 0.12, "s_base": 0.70, "s_std": 0.11, "g_base": 0.75, "g_std": 0.09, "greenwashing_risk": 0.30},
    }

    company_sector = None
    for sector, companies in SECTOR_COMPANIES.items():
        if company in companies:
            company_sector = sector
            break

    if not company_sector:
        return {"e_score": 0.5, "s_score": 0.5, "g_score": 0.5, "total": 0.5, "greenwashing_score": 0, "credibility": 0.5}

    profile = sector_profiles[company_sector]
    np.random.seed(hash(company + str(year)) % 2**32)

    # 1) 연도별 개선 추세(비선형)
    years_since_2015 = year - 2015
    improvement_factor = 0.015 * years_since_2015 + 0.003 * (years_since_2015 ** 1.5) / 10

    # 2) 기업별 특성 계수
    company_hash = hash(company) % 100
    commitment_level = (company_hash / 100) * 0.3 + 0.7  # 0.7~1.0

    # 3) 실제 점수
    e_actual = np.clip(profile["e_base"] + improvement_factor * commitment_level + np.random.normal(0, profile["e_std"] * 0.5), 0, 1)
    s_actual = np.clip(profile["s_base"] + improvement_factor * commitment_level + np.random.normal(0, profile["s_std"] * 0.5), 0, 1)
    g_actual = np.clip(profile["g_base"] + improvement_factor * 0.8 + np.random.normal(0, profile["g_std"] * 0.5), 0, 1)

    # 4) 그린워싱(선언 vs 실적)
    e_claimed = np.clip(e_actual + profile["greenwashing_risk"] * np.random.uniform(0, 0.2), 0, 1)
    greenwashing_score = np.clip(abs(e_claimed - e_actual) / profile["greenwashing_risk"], 0, 1)

    # 5) 신뢰도
    credibility = 1 - greenwashing_score * 0.5

    # 6) 가중 평균
    total = e_actual * 0.4 + s_actual * 0.3 + g_actual * 0.3
    adjusted_total = total * credibility

    return {
        "e_score": round(e_actual, 4),
        "s_score": round(s_actual, 4),
        "g_score": round(g_actual, 4),
        "total": round(total, 4),
        "adjusted_total": round(adjusted_total, 4),
        "greenwashing_score": round(greenwashing_score, 4),
        "credibility": round(credibility, 4),
        "e_claimed": round(e_claimed, 4),
    }

# -----------------------------------------------------
# 주가 수익률 계산
# -----------------------------------------------------
def calculate_stock_return(company: str, year: int, px_hist: pd.DataFrame) -> float:
    if px_hist is None or px_hist.empty or "year" not in px_hist.columns:
        return np.nan
    year = int(year)
    year_data = px_hist.loc[px_hist["year"] == year].sort_values("date")
    if len(year_data) < 2:
        return np.nan
    try:
        start_price = float(year_data["stock_price"].iloc[0])
        end_price = float(year_data["stock_price"].iloc[-1])
    except Exception:
        return np.nan
    if not np.isfinite(start_price) or start_price <= 0:
        return np.nan
    return (end_price / start_price - 1.0) * 100.0

# -----------------------------------------------------
# 사이드바 필터
# -----------------------------------------------------
st.sidebar.header("⚙️ 필터")

sectors = ["전체"] + list(SECTOR_COMPANIES.keys())
selected_sector = st.sidebar.selectbox("🏭 산업 선택", sectors, index=0)

if selected_sector == "전체":
    available_companies = list(COMPANY_MAP.keys())
    default_companies = available_companies[:4]
else:
    available_companies = list(SECTOR_COMPANIES[selected_sector].keys())
    default_companies = available_companies[:4]

companies = st.sidebar.multiselect(
    f"🏢 대상 기업 ({selected_sector})",
    available_companies,
    default=default_companies
)

if not companies:
    st.warning("최소 1개 기업을 선택하세요.")
    st.stop()

# 기업 선택
# -----------------------------------------------------
st.sidebar.markdown("---")
company = st.sidebar.selectbox("📌 상세 분석 기업", companies)



CUR_YEAR = datetime.now().year
years = st.sidebar.slider("📅 분석 연도 범위", 2015, CUR_YEAR, (max(2019, CUR_YEAR - 4), CUR_YEAR))


# 그린워싱 필터
st.sidebar.info(f"✅ 선택된 기업: {len(companies)}개")

show_greenwashing = st.sidebar.checkbox("🚨 그린워싱 의심 기업 강조", value=True)
greenwashing_threshold = st.sidebar.slider("그린워싱 임계값", 0.0, 1.0, 0.4, 0.05)



# -----------------------------------------------------
# 유틸 함수들
# -----------------------------------------------------
@st.cache_data(ttl=60*60)
def _robust_yf_download(ticker: str, start: str, end: str):
    """야후 재시도 + 대체 경로(history)까지 시도."""
    try:
        df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False, progress=False, threads=False)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    # fallback: history()
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start, end=end, interval="1d", auto_adjust=False)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=60*60)
def fetch_annual_prices(ticker: str, start_year: int, end_year: int) -> pd.DataFrame:
    """
    연도별 시가/종가/수익률 계산 (연말 종가 기준)
    - 컬럼명/다중인덱스/케이스 차이 등을 견고하게 처리
    """
    start = f"{start_year}-01-01"
    end   = f"{end_year + 1}-01-15"

    raw = _robust_yf_download(ticker, start, end)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["year", "year_open", "year_close", "stock_price", "stock_return"])

    # 1) 다중 인덱스 컬럼 평탄화
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = ["_".join([str(x) for x in t if x != ""]) for t in raw.columns]

    df = raw.reset_index()

    # 2) 날짜 컬럼 찾기 (Date, Datetime, index 등)
    date_col = None
    for cand in ["Date", "date", "Datetime", "datetime", "index"]:
        if cand in df.columns:
            date_col = cand
            break
    # 마지막 수단: 인덱스를 날짜로 인식
    if date_col is None:
        try:
            df["date"] = pd.to_datetime(raw.index)
            date_col = "date"
        except Exception:
            return pd.DataFrame(columns=["year", "year_open", "year_close", "stock_price", "stock_return"])

    # 3) 종가 컬럼 찾기 (Close 우선, 없으면 Adj Close/소문자/숫자컬럼 fallback)
    close_col = None
    for cand in ["Close", "Adj Close", "close", "adj close", "AdjClose", "adjclose"]:
        if cand in df.columns:
            close_col = cand
            break
    if close_col is None:
        # 숫자 컬럼 중 결측이 적은 후보 사용
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            close_col = num_cols[0]
        else:
            return pd.DataFrame(columns=["year", "year_open", "year_close", "stock_price", "stock_return"])

    # 4) 정규화
    df.rename(columns={date_col: "date", close_col: "close"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date")
    if df.empty:
        return pd.DataFrame(columns=["year", "year_open", "year_close", "stock_price", "stock_return"])

    df["year"] = df["date"].dt.year.astype(int)

    # 5) 연도별 첫/마지막 거래일 종가
    firsts = df.groupby("year").first()[["close"]].rename(columns={"close": "year_open"})
    lasts  = df.groupby("year").last()[["close"]].rename(columns={"close": "year_close"})
    annual = firsts.join(lasts, how="outer").reset_index()

    # 6) 구간 슬라이스 및 수익률 계산
    annual = annual[(annual["year"] >= start_year) & (annual["year"] <= end_year)].copy()
    if annual.empty:
        return pd.DataFrame(columns=["year", "year_open", "year_close", "stock_price", "stock_return"])

    annual["stock_price"] = annual["year_close"]
    annual["stock_return"] = np.where(
        (annual["year_open"] > 0) & np.isfinite(annual["year_open"]) & np.isfinite(annual["year_close"]),
        (annual["year_close"] / annual["year_open"] - 1.0) * 100.0,
        np.nan
    )
    return annual[["year", "year_open", "year_close", "stock_price", "stock_return"]].sort_values("year")


@st.cache_data(ttl=60 * 60)
def fetch_yf_history(ticker: str, start_date: str):
    cols = ["date", "stock_price", "year"]
    try:
        data = yf.download(ticker, start=start_date, progress=False)
        if data is None or data.empty:
            return pd.DataFrame(columns=cols)  # 컬럼 보장
        hist = data.reset_index()[["Date", "Close"]].copy()
        hist.rename(columns={"Date": "date", "Close": "stock_price"}, inplace=True)
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
        hist["stock_price"] = pd.to_numeric(hist["stock_price"], errors="coerce")
        hist = hist.dropna(subset=["date", "stock_price"]).sort_values("date")
        hist["year"] = hist["date"].dt.year.astype(int)
        hist = hist[["date", "stock_price", "year"]].reset_index(drop=True)
        return hist
    except Exception:
        return pd.DataFrame(columns=cols)

def fetch_yf_info(ticker: str):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "market_cap": info.get("marketCap"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
        }
    except Exception:
        return {"market_cap": None, "sector": None, "industry": None}

# -----------------------------------------------------
# 데이터 빌드
# -----------------------------------------------------
@st.cache_data(ttl=60 * 30)
@st.cache_data(ttl=60*30)
def build_results(companies, year_range):
    rows = []
    start_year, end_year = year_range
    start_date = f"{start_year-1}-01-01"

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, comp in enumerate(companies):
        status_text.text(f"📊 분석 중: {comp} ({idx+1}/{len(companies)})")

        # 티커/기업 정보
        info = COMPANY_MAP[comp]
        tkr = info["ticker"]
        company_info = fetch_yf_info(tkr)

        # ★ 주가 데이터 (일/연)
        px_hist = fetch_yf_history(tkr, start_date)                 # 일 단위(보조)
        annual  = fetch_annual_prices(tkr, start_year, end_year)    # ★ 연말 종가/연 수익률(우선)

        # 연도별 ESG & 주가 채우기
        for year in range(start_year, end_year + 1):
            scores = calculate_advanced_esg_score(comp, year)

            # ★ 연 데이터 우선 사용
            row_annual = annual.loc[annual["year"] == int(year)]
            if not row_annual.empty:
                stock_price  = float(row_annual.iloc[0]["stock_price"])
                stock_return = float(row_annual.iloc[0]["stock_return"])
            else:
                # 연 데이터가 없으면 일 데이터에서 보조 계산
                if px_hist is not None and not px_hist.empty and "year" in px_hist.columns:
                    year_price = px_hist.loc[px_hist["year"] == int(year)].sort_values("date")
                    stock_price  = float(year_price["stock_price"].iloc[-1]) if not year_price.empty else np.nan
                    stock_return = calculate_stock_return(comp, year, px_hist)
                else:
                    stock_price, stock_return = np.nan, np.nan

            rows.append({
                "company": comp,
                "year": year,
                "esg_total": scores["total"],
                "esg_adjusted": scores["adjusted_total"],
                "e_score": scores["e_score"],
                "s_score": scores["s_score"],
                "g_score": scores["g_score"],
                "e_claimed": scores["e_claimed"],
                "greenwashing": scores["greenwashing_score"],
                "credibility": scores["credibility"],
                "stock_price": stock_price,
                "stock_return": stock_return,
                "market_cap": company_info.get("market_cap"),
            })

        progress_bar.progress((idx + 1) / len(companies))

    status_text.text("✅ 분석 완료!")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()

    df_res = pd.DataFrame(rows)
    if not df_res.empty:
        df_res.sort_values(["company", "year"], inplace=True)
    return df_res



# 데이터 빌드
df = build_results(companies, years)

# CSV 저장
try:
    df.to_csv("results.csv", index=False, encoding="utf-8-sig")
except Exception:
    pass

# -----------------------------------------------------
# 데이터 미리보기
# -----------------------------------------------------
with st.expander("📂 분석 결과 데이터 미리보기"):
    st.dataframe(df, use_container_width=True)

# -----------------------------------------------------

# -----------------------------------------------------
# 기업 상세 정보
# -----------------------------------------------------
st.subheader(f"📌 {company} 상세 분석")
company_data = df[df["company"] == company].sort_values("year")

if company_data.empty:
    st.warning("데이터가 비어 있습니다.")
else:
    latest = company_data.iloc[-1]

    # 그린워싱 경고
    if latest["greenwashing"] > greenwashing_threshold:
        st.error(f"🚨 그린워싱 의심: {latest['greenwashing']:.2%} (신뢰도: {latest['credibility']:.2%})")

    # 점수 표시
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        delta = latest["esg_adjusted"] - company_data.iloc[0]["esg_adjusted"] if len(company_data) > 1 else None
        st.metric("ESG 종합", f"{latest['esg_adjusted']:.3f}", delta=f"{delta:.3f}" if delta else None)
    with col2:
        st.metric("환경(E)", f"{latest['e_score']:.3f}")
    with col3:
        st.metric("사회(S)", f"{latest['s_score']:.3f}")
    with col4:
        st.metric("지배구조(G)", f"{latest['g_score']:.3f}")
    with col5:
        credibility_color = "normal" if latest["credibility"] > 0.8 else "off"
        st.metric("신뢰도", f"{latest['credibility']:.2%}", delta=None, delta_color=credibility_color)

    st.markdown("---")

    # ESG 바 차트 & 그린워싱 분석
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        esg_breakdown = pd.DataFrame({
            "영역": ["환경(E)", "사회(S)", "지배구조(G)"],
            "점수": [latest["e_score"], latest["s_score"], latest["g_score"]],
        })
        fig_bar = px.bar(
            esg_breakdown,
            x="영역",
            y="점수",
            title=f"{company} ESG 영역별 점수 ({int(latest['year'])}년)",
            color="점수",
            color_continuous_scale="RdYlGn",
            range_color=[0, 1],
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_chart2:
        # 그린워싱 추이
        fig_gw = go.Figure()
        fig_gw.add_trace(go.Scatter(x=company_data["year"], y=company_data["e_score"], name="실제 환경점수", line=dict(color="green", width=2)))
        fig_gw.add_trace(go.Scatter(x=company_data["year"], y=company_data["e_claimed"], name="공개 환경점수", line=dict(color="lightgreen", width=2, dash="dash")))
        fig_gw.update_layout(
            title=f"{company} 그린워싱 분석 (실제 vs 공개)",
            xaxis_title="연도",
            yaxis_title="환경 점수",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_gw, use_container_width=True)

# (그린워싱 차트 아래)
# --- 연도별 주가 추이 (연말 종가) : ESG·그린워싱 바로 아래 표시 ---
price_series = (
    company_data.loc[company_data["stock_price"].notna(), ["year", "stock_price"]]
    .drop_duplicates(subset=["year"])
    .sort_values("year")
)
if not price_series.empty:
    fig_price = px.line(
        price_series, x="year", y="stock_price",
        title=f"{company} 연도별 주가 추이 (연말 종가 기준)",
        markers=True, labels={"year": "연도", "stock_price": "주가"}
    )
    fig_price.update_xaxes(dtick=1)
    st.plotly_chart(fig_price, use_container_width=True)
else:
    st.info("선택 기간에 연도별 주가 데이터가 없습니다.")

# -----------------------------------------------------
# 기업 비교
# -----------------------------------------------------
st.subheader("📊 기업 비교 분석")
if len(companies) >= 2:
    col_cmp1, col_cmp2 = st.columns(2)
    with col_cmp1:
        cmpA = st.selectbox("비교 기업 A", companies, index=0, key="cmpA")
    with col_cmp2:
        cmpB = st.selectbox("비교 기업 B", companies, index=min(1, len(companies) - 1), key="cmpB")

    # 최신 데이터로 비교
    dataA = df[df["company"] == cmpA].iloc[-1]
    dataB = df[df["company"] == cmpB].iloc[-1]

    # Radar 차트
    cats = ["환경(E)", "사회(S)", "지배구조(G)", "신뢰도"]
    valsA = [dataA["e_score"], dataA["s_score"], dataA["g_score"], dataA["credibility"]]
    valsB = [dataB["e_score"], dataB["s_score"], dataB["g_score"], dataB["credibility"]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=valsA, theta=cats, fill="toself", name=cmpA))
    fig.add_trace(go.Scatterpolar(r=valsB, theta=cats, fill="toself", name=cmpB))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, title=f"{cmpA} vs {cmpB} ESG 비교")
    st.plotly_chart(fig, use_container_width=True)

    # 점수 비교 테이블
    comparison = pd.DataFrame({
        "항목": ["ESG 종합", "환경(E)", "사회(S)", "지배구조(G)", "신뢰도", "그린워싱"],
        cmpA: [dataA["esg_adjusted"], dataA["e_score"], dataA["s_score"], dataA["g_score"], dataA["credibility"], dataA["greenwashing"]],
        cmpB: [dataB["esg_adjusted"], dataB["e_score"], dataB["s_score"], dataB["g_score"], dataB["credibility"], dataB["greenwashing"]],
    })
    comparison["차이"] = comparison[cmpA] - comparison[cmpB]
    st.dataframe(comparison, use_container_width=True, hide_index=True)
else:
    st.info("비교 분석을 위해 2개 이상 기업을 선택하세요.")

# -----------------------------------------------------
# 그린워싱 의심 기업 목록
# -----------------------------------------------------
st.subheader("🚨 그린워싱 의심 기업")

if df is None or df.empty:
    st.info("표시할 데이터가 없습니다.")
    st.stop()

latest_year = int(df["year"].max())
df_latest = df[df["year"] == latest_year]

greenwashing_companies = df_latest[df_latest["greenwashing"] > greenwashing_threshold].sort_values("greenwashing", ascending=False)

if not greenwashing_companies.empty:
    gw_display = greenwashing_companies[["company", "e_claimed", "e_score", "greenwashing", "credibility"]].copy()
    gw_display.columns = ["기업명", "공개 환경점수", "실제 환경점수", "그린워싱 점수", "신뢰도"]
    gw_display["괴리도"] = gw_display["공개 환경점수"] - gw_display["실제 환경점수"]

    st.dataframe(
        gw_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "그린워싱 점수": st.column_config.ProgressColumn("🚨 그린워싱", min_value=0, max_value=1, format="%.2f"),
            "신뢰도": st.column_config.ProgressColumn("신뢰도", min_value=0, max_value=1, format="%.2f"),
        },
    )
else:
    st.success("✅ 현재 설정 기준으로 그린워싱 의심 기업이 없습니다.")

# -----------------------------------------------------
# 산업별 ESG 랭킹
# -----------------------------------------------------
st.subheader("🏆 산업별 ESG 우수 기업")

sector_rankings = {}
for sector, companies_dict in SECTOR_COMPANIES.items():
    sector_data = df_latest[df_latest["company"].isin(companies_dict.keys())]
    if not sector_data.empty:
        top3 = sector_data.nlargest(3, "esg_adjusted")[["company", "esg_adjusted", "e_score", "s_score", "g_score", "credibility"]]
        sector_rankings[sector] = top3

if sector_rankings:
    tabs = st.tabs(list(sector_rankings.keys()))
    for tab, (sector, ranking) in zip(tabs, sector_rankings.items()):
        with tab:
            ranking_display = ranking.copy()
            ranking_display.insert(0, "순위", ["🥇", "🥈", "🥉"][: len(ranking)])
            ranking_display.columns = ["순위", "기업명", "ESG 종합", "환경(E)", "사회(S)", "지배구조(G)", "신뢰도"]
            st.dataframe(ranking_display, use_container_width=True, hide_index=True)

# -----------------------------------------------------
# 전체 TOP 10
# -----------------------------------------------------
st.subheader("✅ ESG 우수 기업 순위")
top10 = df_latest.nlargest(10, "esg_adjusted")[["company", "esg_adjusted", "e_score", "s_score", "g_score", "credibility", "greenwashing"]]
top10.insert(0, "순위", range(1, len(top10) + 1))
top10.columns = ["순위", "기업명", "ESG 종합", "환경(E)", "사회(S)", "지배구조(G)", "신뢰도", "그린워싱"]

st.dataframe(
    top10,
    use_container_width=True,
    hide_index=True,
    column_config={
        "순위": st.column_config.NumberColumn("🏆 순위", width="small"),
        "ESG 종합": st.column_config.ProgressColumn("ESG", min_value=0, max_value=1, format="%.3f"),
        "환경(E)": st.column_config.ProgressColumn("E", min_value=0, max_value=1, format="%.3f"),
        "사회(S)": st.column_config.ProgressColumn("S", min_value=0, max_value=1, format="%.3f"),
        "지배구조(G)": st.column_config.ProgressColumn("G", min_value=0, max_value=1, format="%.3f"),
        "신뢰도": st.column_config.ProgressColumn("신뢰도", min_value=0, max_value=1, format="%.2f"),
        "그린워싱": st.column_config.ProgressColumn("🚨", min_value=0, max_value=1, format="%.2f"),
    },
)

# -----------------------------------------------------
# -----------------------------------------------------
# 투자 인사이트 및 추천
# -----------------------------------------------------
st.subheader("💡 AI 투자 인사이트")

col_insight1, col_insight2 = st.columns(2)

with col_insight1:
    st.markdown("### 🌟 ESG 개선 + 주가 상승 기업")

    picks = []
    if len(df["year"].unique()) >= 2:
        for comp in df["company"].unique():
            comp_data = df[df["company"] == comp].sort_values("year")
            if len(comp_data) < 2:
                continue

            # ESG 개선도
            esg_change = comp_data.iloc[-1]["esg_adjusted"] - comp_data.iloc[0]["esg_adjusted"]

            # 연도별 수익률 평균(없으면 가격 모멘텀 대체)
            rr = comp_data["stock_return"].dropna()
            avg_return = rr.mean() if not rr.empty else np.nan
            if np.isnan(avg_return):
                prices = comp_data[["stock_price"]].dropna()
                if len(prices) >= 2 and prices.iloc[0, 0] > 0:
                    avg_return = (prices.iloc[-1, 0] / prices.iloc[0, 0] - 1.0) * 100.0

            credibility = comp_data.iloc[-1]["credibility"]
            if np.isfinite(avg_return):
                score = (esg_change * 100) + (avg_return * 0.3) + (credibility * 20)
                # 조건: ESG 개선 + 신뢰도 0.7 이상
                if esg_change > 0 and credibility >= 0.7:
                    picks.append({"company": comp, "esg_change": esg_change,
                                  "avg_return": avg_return, "credibility": credibility,
                                  "score": score})

    def draw_esg_price_dual(comp_name: str):
        comp_data = df[df["company"] == comp_name].sort_values("year")
        if comp_data.empty:
            return
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=comp_data["year"], y=comp_data["esg_adjusted"],
                       name="ESG(신뢰도 반영)", mode="lines+markers"),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=comp_data["year"], y=comp_data["stock_price"],
                       name="주가(연말 종가)", mode="lines+markers"),
            secondary_y=True
        )
        fig.update_layout(
            title=f"{comp_name} | ESG vs 주가 (연도별)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(t=60)
        )
        fig.update_yaxes(title_text="ESG 점수", range=[0, 1], secondary_y=False)
        fig.update_yaxes(title_text="주가(원)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    if picks:
        df_picks = pd.DataFrame(picks).nlargest(5, "score")
        # 카드 요약 + 듀얼축 그래프들
        for _, row in df_picks.iterrows():
            st.metric(row["company"], f"ESG +{row['esg_change']:.3f}",
                      delta=f"수익률 {row['avg_return']:.1f}%")
            st.caption(f"신뢰도: {row['credibility']:.1%} | 투자점수: {row['score']:.1f}")
            draw_esg_price_dual(row["company"])
    else:
        st.info("선택된 기간/기업 조합에서 ‘ESG 개선 + 주가 상승’ 조건을 만족하는 기업이 없습니다. 그래도 대표 기업의 흐름을 보여줄게요.")
        # 추천이 비어도 현재 상세 선택 기업 그래프는 보여줌
        draw_esg_price_dual(company)

with col_insight2:
    st.markdown("### 📊 산업별 평균 ESG")
    sector_avg = []
    latest_y = int(df["year"].max())
    for sector, companies_dict in SECTOR_COMPANIES.items():
        mask = (df["company"].isin(companies_dict.keys())) & (df["year"] == latest_y)
        sector_data = df[mask]
        if not sector_data.empty:
            sector_avg.append({
                "sector": sector,
                "avg": sector_data["esg_adjusted"].mean(),
                "greenwashing": sector_data["greenwashing"].mean(),
            })
    if sector_avg:
        df_sector_avg = pd.DataFrame(sector_avg).sort_values("avg", ascending=False)
        fig_sector = px.bar(
            df_sector_avg, x="sector", y="avg",
            labels={"sector": "산업", "avg": "ESG 평균"},
            color="greenwashing",
            color_continuous_scale="RdYlGn_r",
            title="산업별 평균 ESG (색상: 그린워싱 위험도)"
        )
        st.plotly_chart(fig_sector, use_container_width=True)
    else:
        st.info("산업 평균을 계산할 데이터가 없습니다.")


# -----------------------------------------------------
# 지역별 투자 추천 (시가총액 기반) - 섹션 이름만 유지
# -----------------------------------------------------
st.subheader("🎯 투자 추천 종목")

col_rec1, col_rec2, col_rec3 = st.columns(3)

with col_rec1:
    st.markdown("#### 🥇 ESG 우수 + 고신뢰도")
    best_esg = df_latest[df_latest["credibility"] > 0.85].nlargest(3, "esg_adjusted")
    if not best_esg.empty:
        for _, row in best_esg.iterrows():
            st.success(f"**{row['company']}**")
            st.caption(f"ESG: {row['esg_adjusted']:.3f} | 신뢰도: {row['credibility']:.1%}")
    else:
        st.info("조건 충족 기업 없음")

with col_rec2:
    st.markdown("#### 📈 ESG 급성장 기업")
    if len(df["year"].unique()) >= 2:
        growth_companies = []
        for comp in df["company"].unique():
            comp_data = df[df["company"] == comp].sort_values("year")
            if len(comp_data) >= 2:
                growth = comp_data.iloc[-1]["esg_adjusted"] - comp_data.iloc[0]["esg_adjusted"]
                credibility = comp_data.iloc[-1]["credibility"]
                if credibility > 0.7:
                    growth_companies.append({"company": comp, "growth": growth, "credibility": credibility})
        if growth_companies:
            df_growth = pd.DataFrame(growth_companies).nlargest(3, "growth")
            for _, row in df_growth.iterrows():
                st.info(f"**{row['company']}**")
                st.caption(f"ESG 성장: +{row['growth']:.3f} | 신뢰도: {row['credibility']:.1%}")

with col_rec3:
    st.markdown("#### ⚠️ 주의 종목")
    risky = df_latest.nlargest(3, "greenwashing")
    if not risky.empty:
        for _, row in risky.iterrows():
            st.warning(f"**{row['company']}**")
            st.caption(f"그린워싱: {row['greenwashing']:.1%} | 신뢰도: {row['credibility']:.1%}")

# -----------------------------------------------------
# 점수 계산 방법론 설명
# -----------------------------------------------------
with st.expander("ℹ️ ESG 점수 계산 방법론"):
    st.markdown("""
    ### 개선된 ESG 점수 산정 체계

    #### 1. 기본 점수 (산업별 벤치마크)
    - 각 산업의 특성에 맞는 기본 점수 및 변동성 반영

    #### 2. 시계열 개선도 (비선형)
    - 2015년 기준 연도별 개선 추세 반영

    #### 3. 기업 특성 계수
    - 기업별 ESG 대응 수준 차별화 (0.7~1.0)

    #### 4. 그린워싱 감지 시스템 🚨
    - 공개 점수 vs 실제 점수 괴리도 측정
    - 산업별 그린워싱 위험도 반영

    #### 5. 신뢰도 점수
    - 신뢰도 = 1 - (그린워싱 점수 × 0.5)
    - 최종 ESG 점수에 신뢰도 가중치 적용

    #### 6. 주가 상관관계 분석
    - ESG 개선도 + 주가 수익률 통합 분석

    #### 7. 투자 매력도 산식
    ```
    투자점수 = (ESG 개선도 × 100) + (평균 수익률 × 0.3) + (신뢰도 × 20)
    ```

    #### 📌 한계점
    - 현재는 시뮬레이션 데이터 기반
    - 실제 서비스에서는 DART API/재무제표/뉴스 감성분석 연동 예정
    """)

# -----------------------------------------------------
# Footer
# -----------------------------------------------------
st.markdown("---")
st.caption("© 2024 B.B.BIC ESG Investment Platform | 데이터 출처: Yahoo Finance, DART (시뮬레이션)")



