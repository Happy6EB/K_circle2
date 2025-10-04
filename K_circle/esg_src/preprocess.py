# preprocess.py
import re
import pandas as pd

def clean_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = t.lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"[^a-z0-9가-힣\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _norm(s) -> str:
    return str(s).replace("\ufeff", "").strip().strip('"').strip("'").lower()

def load_news(path: str) -> pd.DataFrame:
    # ✅ UTF-8 CSV 읽기 (BOM도 자동 처리)
    df = pd.read_csv(path, encoding="utf-8-sig", engine="python")

    df.columns = [_norm(c) for c in df.columns]
    rename = {"제목":"title","기사제목":"title","내용":"content","본문":"content",
              "회사":"company","기업":"company","날짜":"date","작성일":"date","출처":"source"}
    df = df.rename(columns=rename)

    # 필수 컬럼 체크
    for col in ["company","content"]:
        if col not in df.columns:
            raise KeyError(f"CSV에 '{col}' 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")
    if "title" not in df.columns:
        df["title"] = ""

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["text"] = (df["title"].fillna("") + " " + df["content"].fillna("")).apply(clean_text)
    return df.dropna(subset=["company","text"])
