import os
import time
import json
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
import re
from bs4 import BeautifulSoup

# -----------------------------------------------------
# 기본 설정
# -----------------------------------------------------
st.set_page_config(page_title="ESG 기반 AI 투자지원 (DART 연계)", layout="wide")
st.title("📊 ESG 기반 AI 투자지원 대시보드 (DART 연계)")
st.caption("B.B.BIC | DART 전자공시 + 뉴스 분석 → AI 점수화 → 대시보드")

# -----------------------------------------------------
# 기업 목록 (DART 고유번호 포함)
# -----------------------------------------------------
COMPANY_BY_SECTOR = {
    "IT/전자": {
        "삼성전자": {"ticker": "005930.KS", "dart_code": "00126380"},
        "SK하이닉스": {"ticker": "000660.KS", "dart_code": "00164742"},
        "LG전자": {"ticker": "066570.KS", "dart_code": "00401731"},
        "NAVER": {"ticker": "035420.KS", "dart_code": "00164779"},
    },
    "자동차": {
        "현대자동차": {"ticker": "005380.KS", "dart_code": "00164742"},
        "기아": {"ticker": "000270.KS", "dart_code": "00164529"},
    },
    "화학/에너지": {
        "LG화학": {"ticker": "051910.KS", "dart_code": "00188796"},
        "포스코홀딩스": {"ticker": "005490.KS", "dart_code": "00164958"},
    },
    "금융": {
        "KB금융": {"ticker": "105560.KS", "dart_code": "00434003"},
        "신한지주": {"ticker": "055550.KS", "dart_code": "00190977"},
    },
}

COMPANY_MAP = {}
for sector, companies_dict in COMPANY_BY_SECTOR.items():
    COMPANY_MAP.update(companies_dict)

# -----------------------------------------------------
# DART API 함수들
# -----------------------------------------------------
DART_API_KEY = st.secrets.get("DART_API_KEY", "")

@st.cache_data(ttl=60*60*24)  # 24시간 캐시
def get_dart_reports(corp_code: str, year: int):
    """DART에서 지속가능경영보고서 검색"""
    if not DART_API_KEY:
        return []
    
    url = "https://opendart.fss.or.kr/api/list.json"
    params = {
        "crtfc_key": DART_API_KEY,
        "corp_code": corp_code,
        "bgn_de": f"{year}0101",
        "end_de": f"{year}1231",
        "pblntf_ty": "E",  # 기타공시
        "page_count": 100
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get("status") != "000":
            return []
        
        reports = []
        for item in data.get("list", []):
            title = item.get("report_nm", "")
            # 지속가능, ESG, 통합보고서 검색
            if any(keyword in title for keyword in ["지속가능", "ESG", "통합보고서", "사회책임"]):
                reports.append({
                    "title": title,
                    "date": item.get("rcept_dt"),
                    "url": f"http://dart.fss.or.kr/dsaf001/main.do?rcpNo={item.get('rcept_no')}"
                })
        return reports
    except Exception as e:
        st.warning(f"DART API 오류: {e}")
        return []

def analyze_dart_report_url(url: str) -> dict:
    """
    DART 보고서 URL에서 ESG 관련 텍스트 추출 및 분석
    실제로는 PDF 다운로드 후 파싱 필요하지만, 데모용으로 간소화
    """
    # 실제 구현: PDF 다운로드 → OCR/텍스트 추출 → 정량지표 파싱
    # 여기서는 시뮬레이션
    return {
        "e_metrics": {
            "탄소배출량": np.random.uniform(100000, 500000),
            "재생에너지비율": np.random.uniform(10, 40),
            "폐기물재활용율": np.random.uniform(70, 95)
        },
        "s_metrics": {
            "여성임원비율": np.random.uniform(10, 30),
            "산재율": np.random.uniform(0.1, 0.5),
            "교육시간": np.random.uniform(20, 80)
        },
        "g_metrics": {
            "독립이사비율": np.random.uniform(40, 60),
            "이사회개최": np.random.uniform(8, 15),
            "감사위원": np.random.uniform(3, 5)
        }
    }

# -----------------------------------------------------
# ESG 점수 산정 (정량지표 기반)
# -----------------------------------------------------
def calculate_esg_from_metrics(metrics: dict) -> dict:
    """
    DART에서 추출한 정량지표를 점수로 변환
    벤치마크 대비 상대평가 (업종평균 기준)
    """
    # 환경 점수 (탄소배출량은 낮을수록 좋음)
    carbon_score = 1 - (metrics["e_metrics"]["탄소배출량"] / 500000)
    renewable_score = metrics["e_metrics"]["재생에너지비율"] / 100
    recycle_score = metrics["e_metrics"]["폐기물재활용율"] / 100
    e_score = (carbon_score * 0.5 + renewable_score * 0.3 + recycle_score * 0.2)
    
    # 사회 점수
    female_score = metrics["s_metrics"]["여성임원비율"] / 30
    safety_score = 1 - (metrics["s_metrics"]["산재율"] / 1.0)
    training_score = metrics["s_metrics"]["교육시간"] / 100
    s_score = (female_score * 0.4 + safety_score * 0.4 + training_score * 0.2)
    
    # 지배구조 점수
    independent_score = metrics["g_metrics"]["독립이사비율"] / 60
    meeting_score = min(metrics["g_metrics"]["이사회개최"] / 12, 1.0)
    audit_score = min(metrics["g_metrics"]["감사위원"] / 5, 1.0)
    g_score = (independent_score * 0.5 + meeting_score * 0.3 + audit_score * 0.2)
    
    # 0~1 범위로 정규화
    e_score = np.clip(e_score, 0, 1)
    s_score = np.clip(s_score, 0, 1)
    g_score = np.clip(g_score, 0, 1)
    
    total = e_score * 0.4 + s_score * 0.3 + g_score * 0.3
    
    return {
        "e_score": round(e_score, 4),
        "s_score": round(s_score, 4),
        "g_score": round(g_score, 4),
        "total": round(total, 4)
    }

# -----------------------------------------------------
# 사이드바 필터
# -----------------------------------------------------
st.sidebar.header("⚙️ 필터")

sectors = ["전체"] + list(COMPANY_BY_SECTOR.keys())
selected_sector = st.sidebar.selectbox("🏭 산업 선택", sectors)

if selected_sector == "전체":
    available_companies = list(COMPANY_MAP.keys())
    default_companies = available_companies[:4]
else:
    available_companies = list(COMPANY_BY_SECTOR[selected_sector].keys())
    default_companies = available_companies[:min(4, len(available_companies))]

companies = st.sidebar.multiselect(
    f"🏢 대상 기업 ({selected_sector})",
    available_companies,
    default=default_companies
)

if not companies:
    st.warning("최소 1개 기업을 선택하세요.")
    st.stop()

years = st.sidebar.slider("📅 분석 연도 범위", 2020, 2024, (2022, 2024))

st.sidebar.info(f"✅ 선택된 기업: {len(companies)}개")

# DART API 키 입력
if not DART_API_KEY:
    st.sidebar.warning("⚠️ DART API 키를 secrets.toml에 추가하세요")
    st.sidebar.code('DART_API_KEY = "your_api_key"')

# -----------------------------------------------------
# 유틸 함수들
# -----------------------------------------------------
@st.cache_data(ttl=60*60)
def fetch_yf_history(ticker: str, start_date: str):
    try:
        data = yf.download(ticker, start=start_date, progress=False)
        if data is None or data.empty:
            return pd.DataFrame()
        hist = data.reset_index()[["Date", "Close"]]
        hist.rename(columns={"Date":"date","Close":"stock_price"}, inplace=True)
        hist["year"] = hist["date"].dt.year
        return hist
    except Exception:
        return pd.DataFrame()

# -----------------------------------------------------
# 데이터 빌드
# -----------------------------------------------------
@st.cache_data(ttl=60*30)
def build_results(companies, year_range):
    rows = []
    start_year, end_year = year_range
    start_date = f"{start_year-1}-01-01"
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, comp in enumerate(companies):
        status_text.text(f"분석 중: {comp} ({idx+1}/{len(companies)})")
        
        info = COMPANY_MAP[comp]
        tkr = info["ticker"]
        dart_code = info.get("dart_code", "")
        
        # 주가 데이터
        px_hist = fetch_yf_history(tkr, start_date)
        
        # 연도별 DART 보고서 분석
        for year in range(start_year, end_year + 1):
            # DART 보고서 검색
            reports = get_dart_reports(dart_code, year) if dart_code else []
            
            if reports:
                # 첫 번째 보고서 분석
                metrics = analyze_dart_report_url(reports[0]["url"])
                scores = calculate_esg_from_metrics(metrics)
                report_available = True
            else:
                # 보고서 없으면 기본값 (낮은 점수)
                scores = {"e_score": 0.3, "s_score": 0.3, "g_score": 0.3, "total": 0.3}
                report_available = False
            
            # 주가 데이터 매칭
            year_price = px_hist[px_hist["year"] == year]
            if not year_price.empty:
                stock_price = float(year_price.tail(1)["stock_price"].values[0])
            else:
                stock_price = np.nan
            
            rows.append({
                "company": comp,
                "year": year,
                "esg_total": scores["total"],
                "e_score": scores["e_score"],
                "s_score": scores["s_score"],
                "g_score": scores["g_score"],
                "stock_price": stock_price,
                "dart_available": report_available,
                "report_count": len(reports)
            })
        
        progress_bar.progress((idx + 1) / len(companies))
    
    status_text.text("✅ 분석 완료!")
    progress_bar.empty()
    
    df_res = pd.DataFrame(rows)
    if not df_res.empty:
        df_res.sort_values(["company", "year"], inplace=True)
    return df_res

# 데이터 빌드
df = build_results(companies, years)

# CSV 저장
try:
    df.to_csv("results_dart.csv", index=False, encoding="utf-8-sig")
except Exception:
    pass

# -----------------------------------------------------
# 데이터 미리보기
# -----------------------------------------------------
with st.expander("📂 DART 분석 결과 미리보기"):
    st.dataframe(df, use_container_width=True)

# -----------------------------------------------------
# 기업 선택
# -----------------------------------------------------
st.sidebar.markdown("---")
company = st.sidebar.selectbox("📌 상세 기업 선택", companies)

# -----------------------------------------------------
# 기업 상세 정보
# -----------------------------------------------------
st.subheader(f"📌 기업 정보: {company}")
company_data = df[df["company"] == company].sort_values("year")

if company_data.empty:
    st.warning("데이터가 비어 있습니다.")
else:
    latest = company_data.iloc[-1]
    
    # DART 보고서 상태
    dart_status = "✅ DART 보고서 분석 완료" if latest["dart_available"] else "⚠️ DART 보고서 미발견 (추정치)"
    st.caption(dart_status)
    
    # 점수 표시
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ESG 종합", f"{latest['esg_total']:.3f}")
    with col2:
        st.metric("환경(E)", f"{latest['e_score']:.3f}")
    with col3:
        st.metric("사회(S)", f"{latest['s_score']:.3f}")
    with col4:
        st.metric("지배구조(G)", f"{latest['g_score']:.3f}")
    
    st.markdown("---")
    
    # ESG 바 차트
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        esg_breakdown = pd.DataFrame({
            "영역": ["환경(E)", "사회(S)", "지배구조(G)"],
            "점수": [latest['e_score'], latest['s_score'], latest['g_score']]
        })
        fig_bar = px.bar(esg_breakdown, x="영역", y="점수",
                         title=f"{company} ESG 영역별 점수",
                         color="점수",
                         color_continuous_scale="Viridis",
                         range_color=[0, 1])
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col_chart2:
        # 연도별 추이
        fig_trend = px.line(company_data, x="year", y=["e_score", "s_score", "g_score"],
                           title=f"{company} ESG 연도별 추이",
                           labels={"value": "점수", "variable": "영역"})
        fig_trend.update_layout(legend_title_text="영역")
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # 주가 차트
    if not company_data["stock_price"].isna().all():
        st.line_chart(company_data.set_index("year")["stock_price"], height=300)
        st.caption("주가 추이")

# -----------------------------------------------------
# 기업 비교
# -----------------------------------------------------
st.subheader("📊 기업 비교 (ESG 영역별)")
if len(companies) >= 2:
    col_cmp1, col_cmp2 = st.columns(2)
    with col_cmp1:
        cmpA = st.selectbox("비교 기업 A", companies, index=0)
    with col_cmp2:
        cmpB = st.selectbox("비교 기업 B", companies, index=min(1, len(companies)-1))
    
    # 최신 데이터로 비교
    dataA = df[df["company"] == cmpA].iloc[-1]
    dataB = df[df["company"] == cmpB].iloc[-1]
    
    cats = ["환경(E)", "사회(S)", "지배구조(G)"]
    valsA = [dataA["e_score"], dataA["s_score"], dataA["g_score"]]
    valsB = [dataB["e_score"], dataB["s_score"], dataB["g_score"]]
    
    df_radar = pd.DataFrame({"영역": cats, cmpA: valsA, cmpB: valsB})
    
    fig = px.line_polar(df_radar, r=cmpA, theta="영역", line_close=True)
    fig.update_traces(name=cmpA, showlegend=True)
    fig.add_scatterpolar(r=df_radar[cmpB], theta=df_radar["영역"], name=cmpB)
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------
# TOP 5 기업
# -----------------------------------------------------
st.subheader("✅ ESG 우수 기업 TOP 5")
top5 = df.groupby("company").agg({
    "esg_total": "mean",
    "e_score": "mean",
    "s_score": "mean",
    "g_score": "mean",
    "report_count": "sum"
}).nlargest(5, "esg_total").reset_index()

top5.insert(0, "순위", range(1, len(top5) + 1))
top5.columns = ["순위", "기업명", "ESG 종합", "환경(E)", "사회(S)", "지배구조(G)", "DART 보고서"]

st.dataframe(
    top5,
    use_container_width=True,
    hide_index=True,
    column_config={
        "순위": st.column_config.NumberColumn("🏆", width="small"),
        "ESG 종합": st.column_config.ProgressColumn("ESG", min_value=0, max_value=1),
        "환경(E)": st.column_config.ProgressColumn("E", min_value=0, max_value=1),
        "사회(S)": st.column_config.ProgressColumn("S", min_value=0, max_value=1),
        "지배구조(G)": st.column_config.ProgressColumn("G", min_value=0, max_value=1),
    }
)