import os
import time
import json
import math
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

# -----------------------------------------------------
# 기본 설정
# -----------------------------------------------------
st.set_page_config(page_title="ESG 기반 AI 투자지원", layout="wide")
st.title("📊 ESG 기반 AI 투자지원 대시보드")
st.caption("B.B.BIC | 실시간(또는 준실시간) 수집 → AI 점수화 → 대시보드")

# 기업 목록과 티커 매핑 (원하는 기업 자유 추가)
COMPANY_MAP = {
    "삼성전자": {"ticker": "005930.KS", "query": "삼성전자 ESG OR 친환경 OR 탄소중립"},
    "SK하이닉스": {"ticker": "000660.KS", "query": "SK하이닉스 ESG OR 친환경 OR 탄소중립"},
    "LG전자": {"ticker": "066570.KS", "query": "LG전자 ESG OR 친환경 OR 탄소중립"},
    "현대자동차": {"ticker": "005380.KS", "query": "현대자동차 ESG OR 친환경 OR 탄소중립"},
    "NAVER": {"ticker": "035420.KS", "query": "네이버 ESG OR 친환경 OR 탄소중립"},
    "한화솔루션": {"ticker": "009830.KS", "query": "한화솔루션 ESG OR 친환경 OR 탄소중립"},
    
    # 여기에 기업 추가!
    "카카오": {"ticker": "035720.KS", "query": "카카오 ESG OR 친환경 OR 탄소중립"},
    "삼성SDI": {"ticker": "006400.KS", "query": "삼성SDI ESG OR 친환경 OR 탄소중립"},
    "LG화학": {"ticker": "051910.KS", "query": "LG화학 ESG OR 친환경 OR 탄소중립"},
    "포스코홀딩스": {"ticker": "005490.KS", "query": "포스코 ESG OR 친환경 OR 탄소중립"},
    "기아": {"ticker": "000270.KS", "query": "기아 ESG OR 친환경 OR 탄소중립"},
    "KB금융": {"ticker": "105560.KS", "query": "KB금융 ESG OR 친환경 OR 탄소중립"},
    "신한지주": {"ticker": "055550.KS", "query": "신한지주 ESG OR 친환경 OR 탄소중립"},
    "셀트리온": {"ticker": "068270.KS", "query": "셀트리온 ESG OR 친환경 OR 탄소중립"},
    "현대모비스": {"ticker": "012330.KS", "query": "현대모비스 ESG OR 친환경 OR 탄소중립"},
}
# -----------------------------------------------------
# 사이드바 필터
# -----------------------------------------------------
st.sidebar.header("⚙️ 필터")
days = st.sidebar.slider("뉴스 분석 기간(일)", 7, 60, 14, 1)
companies = st.sidebar.multiselect(
    "대상 기업", list(COMPANY_MAP.keys()), default=list(COMPANY_MAP.keys())[:4]
)
if not companies:
    st.warning("최소 1개 기업을 선택하세요.")
    st.stop()

# -----------------------------------------------------
# 유틸: 캐시 옵션
# -----------------------------------------------------
@st.cache_data(ttl=60*30)  # 30분 캐시
def cached_json_get(url, headers=None, params=None):
    r = requests.get(url, headers=headers, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=60*60)  # 1시간 캐시
def fetch_yf_history(ticker: str, start_date: str):
    try:
        data = yf.download(ticker, start=start_date)
        if data is None or data.empty:
            return pd.DataFrame()
        hist = data.reset_index()[["Date", "Close"]]
        hist.rename(columns={"Date":"date","Close":"stock_price"}, inplace=True)
        hist["year"] = hist["date"].dt.year
        return hist
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60*60)
def fetch_yf_fastinfo(ticker: str):
    try:
        t = yf.Ticker(ticker)
        fi = getattr(t, "fast_info", None) or {}
        # yfinance fast_info가 없는 종목도 있어 방어적 처리
        market_cap = fi.get("market_cap")
        return market_cap
    except Exception:
        return None

# -----------------------------------------------------
# 뉴스 수집 (NewsAPI → 없으면 Naver → 없으면 샘플)
# -----------------------------------------------------
def fetch_news(company: str, query: str, days_back: int):
    until = datetime.utcnow()
    since = until - timedelta(days=days_back)
    news = []

    # API 키 직접 입력
    news_api_key = st.secrets.get("NEWS_API_KEY")  # toml에서 읽기
    
    if news_api_key:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "from": since.strftime("%Y-%m-%d"),
            "to": until.strftime("%Y-%m-%d"),
            "sortBy": "relevancy",
            "language": "ko",
            "pageSize": 50,
            "apiKey": news_api_key,
        }
        try:
            data = cached_json_get(url, params=params)
            for a in data.get("articles", []):
                title = a.get("title", "")
                desc = a.get("description", "")
                content = (title or "") + " " + (desc or "")
                news.append({"company":company, "date":a.get("publishedAt","")[:10], "source":"newsapi", "content":content})
        except Exception:
            pass

    # 샘플 데이터 (백업)
    if not news:
        samples = [
            f"{company} 탄소중립 로드맵 발표, 재생에너지 사용 확대",
            f"{company} 협력사와의 상생 프로그램 강화, 사회적 책임 확대",
            f"{company} ESG 위원회 확대 개편 및 독립성 강화",
            f"{company} 환경오염 논란에 대한 개선 계획 발표",
        ]
        for s in samples:
            news.append({"company":company, "date": datetime.utcnow().strftime("%Y-%m-%d"), "source":"sample", "content": s})

    return pd.DataFrame(news)

    # 2) 네이버 검색 API (선택)
    if not news:
        cid = st.secrets.get("NAVER_CLIENT_ID")
        csec = st.secrets.get("NAVER_CLIENT_SECRET")
        if cid and csec:
            url = "https://openapi.naver.com/v1/search/news.json"
            headers = {"X-Naver-Client-Id": cid, "X-Naver-Client-Secret": csec}
            params = {"query": query, "display": 40, "sort": "sim"}
            try:
                data = cached_json_get(url, headers=headers, params=params)
                for it in data.get("items", []):
                    title = it.get("title","").replace("<b>","").replace("</b>","")
                    desc = it.get("description","").replace("<b>","").replace("</b>","")
                    news.append({"company":company, "date":datetime.utcnow().strftime("%Y-%m-%d"), "source":"naver", "content":f"{title} {desc}"})
            except Exception:
                pass

    # 3) 샘플(백업)
    if not news:
        samples = [
            f"{company} 탄소중립 로드맵 발표, 재생에너지 사용 확대",
            f"{company} 협력사와의 상생 프로그램 강화, 사회적 책임 확대",
            f"{company} ESG 위원회 확대 개편 및 독립성 강화",
            f"{company} 환경오염 논란에 대한 개선 계획 발표",
        ]
        for s in samples:
            news.append({"company":company, "date": datetime.utcnow().strftime("%Y-%m-%d"), "source":"sample", "content": s})

    return pd.DataFrame(news)

# -----------------------------------------------------
# ESG 점수화
# -----------------------------------------------------
ANALYZER = SentimentIntensityAnalyzer()

# 간단 키워드 가중치(환경/사회/지배구조)
KEYWORD_W = {
    "환경": ["탄소", "온실가스", "재생에너지", "친환경", "배출권", "넷제로", "수소", "태양광", "풍력"],
    "사회": ["안전", "노동", "상생", "지역사회", "기부", "복지", "윤리", "다양성"],
    "지배구조": ["이사회", "감사", "내부통제", "지배구조", "주주", "공시", "투명성"],
}
def keyword_boost(text: str)->float:
    score = 0.0
    for cat, words in KEYWORD_W.items():
        for w in words:
            if w in text:
                score += 0.1
    return score

def esg_score_one(text: str)->float:
    s = ANALYZER.polarity_scores(text)["compound"]  # -1~1
    s += keyword_boost(text)
    return float(np.clip(s, -1, 1))

# -----------------------------------------------------
# 전체 파이프라인: 수집→점수화→집계→주가/시총 결합
# -----------------------------------------------------
@st.cache_data(ttl=60*30)
def build_results(companies, days_back: int):
    rows = []
    price_frames = []

    start_date = (datetime.today() - timedelta(days=365*5)).strftime("%Y-%m-%d")

    for comp in companies:
        info = COMPANY_MAP[comp]
        q = info["query"]
        tkr = info["ticker"]

        # 뉴스 수집
        df_news = fetch_news(comp, q, days_back)
        # 점수화
        df_news["esg_score"] = df_news["content"].apply(esg_score_one)

        # 회사별 집계 (평균/최신/문서수)
        if not df_news.empty:
            esg_avg = df_news["esg_score"].mean()
            esg_last = df_news.sort_values("date")["esg_score"].iloc[-1]
            n_docs = len(df_news)
        else:
            esg_avg, esg_last, n_docs = 0, 0, 0

        # 주가/시총
        px_hist = fetch_yf_history(tkr, start_date)
        market_cap = fetch_yf_fastinfo(tkr)  # 없으면 None

        # 연도별로 결과 뿌리기 (주가가 있으면 매칭)
        if not px_hist.empty:
            for y, grp in px_hist.groupby("year"):
                stock_price = float(grp.tail(1)["stock_price"].values[0])
                rows.append({
                    "company": comp,
                    "year": int(y),
                    "esg_avg": round(esg_avg,4),
                    "esg_last": round(esg_last,4),
                    "n_docs": int(n_docs),
                    "stock_price": stock_price,
                    "market_cap": market_cap if market_cap is not None else np.nan,
                })
        else:
            # 가격이 전혀 없으면 최근 연도 1건만
            rows.append({
                "company": comp,
                "year": datetime.today().year,
                "esg_avg": round(esg_avg,4),
                "esg_last": round(esg_last,4),
                "n_docs": int(n_docs),
                "stock_price": np.nan,
                "market_cap": market_cap if market_cap is not None else np.nan,
            })

    df_res = pd.DataFrame(rows)
    if not df_res.empty:
        df_res.sort_values(["company","year"], inplace=True)
    return df_res

df = build_results(companies, days)

# 필요하면 CSV로도 보관 (제출물 요구 시)
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_path = os.path.join(base_path, "results.csv")
try:
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
except Exception:
    pass

# -----------------------------------------------------
# 상단: Top5 / 사이드바-본문 동기 선택
# -----------------------------------------------------
st.sidebar.markdown("---")
company_sidebar = st.sidebar.selectbox("📌 상세 기업 선택", companies)

# 본문에도 동일 선택 박스(동기화)
company = st.selectbox(
    "기업 선택 (본문)",
    companies,
    index=companies.index(company_sidebar)
)

if company != company_sidebar:
    st.sidebar.success(f"사이드바 선택도 ‘{company}’로 동기화해 주세요.")

# -----------------------------------------------------
# 미리보기 표
# -----------------------------------------------------
with st.expander("📂 결과 데이터 미리보기 (API 집계 결과)"):
    st.dataframe(df.head(20))

# -----------------------------------------------------
# Company Details
# -----------------------------------------------------
st.subheader(f"📌 기업 정보 : {company}")
company_data = df[df["company"] == company].sort_values("year")
if company_data.empty:
    st.warning("데이터가 비어 있습니다.")
else:
    latest = company_data.iloc[-1]
    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("ESG 평균", f"{latest['esg_avg']}")
    with colB:
        st.metric("ESG 최신", f"{latest['esg_last']}")
    with colC:
        st.metric("문서 수", f"{int(latest['n_docs'])}")

    col1, col2 = st.columns(2)
    with col1:
        st.line_chart(company_data.set_index("year")["esg_last"], height=300)
        st.caption("ESG 점수 추이")
    with col2:
        st.line_chart(company_data.set_index("year")["stock_price"], height=300)
        st.caption("주가 추이")

    # 시장가치(가능한 경우)
    if "market_cap" in company_data.columns and company_data["market_cap"].notna().any():
        mc = company_data["market_cap"].dropna().iloc[-1]
        st.caption(f"참고: 현재 시가총액(원화 환산 전 원단위, 데이터 제공 범위에 따라 공란 가능) ≈ {mc:,}")

# -----------------------------------------------------
# 비교(Radar)
# -----------------------------------------------------
st.subheader("📊 기업 비교 (Radar)")
if len(companies) >= 2:
    cmpA = st.selectbox("비교 기업 A", companies, index=0, key="cmpA")
    cmpB = st.selectbox("비교 기업 B", companies, index=1, key="cmpB")

    cats = ["ESG 평균", "ESG 최신", "문서 수"]
    def stats(c):
        sub = df[df["company"] == c]
        return [
            sub["esg_avg"].mean(skipna=True),
            sub["esg_last"].mean(skipna=True),
            sub["n_docs"].mean(skipna=True),
        ]
    valsA, valsB = stats(cmpA), stats(cmpB)
    df_radar = pd.DataFrame({"Category": cats, cmpA: valsA, cmpB: valsB})

    fig = px.line_polar(df_radar, r=cmpA, theta="Category", line_close=True)
    fig.update_traces(name=cmpA, showlegend=True)
    fig.add_scatterpolar(r=df_radar[cmpB], theta=df_radar["Category"], name=cmpB)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("비교할 기업을 2개 이상 선택하세요.")

# -----------------------------------------------------
# 추천 카드 (상위 3개)
# -----------------------------------------------------
st.subheader("✅ 추천 기업")
top3 = df.groupby("company")["esg_avg"].mean().nlargest(3).reset_index()
c1,c2,c3 = st.columns(3)
for i, c in enumerate([c1,c2,c3]):
    if i < len(top3):
        row = top3.iloc[i]
        with c:
            st.metric(row["company"], round(row["esg_avg"],2), "ESG 평균")
