import yaml, pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def load_keywords(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)  # {"E":[...], "S":[...], "G":[...]}

def keyword_score(text: str, kw_map: dict) -> dict:
    scores = {"E":0, "S":0, "G":0}
    for k in ("E","S","G"):
        for w in kw_map.get(k, []):
            if w in text: scores[k] += 1
    return scores

def source_weight(src: str) -> float:
    src = (src or "").lower()
    if any(s in src for s in ["dart","공시","sec","gov"]): return 1.3
    if any(s in src for s in ["reuters","bloomberg","wsj"]): return 1.15
    return 1.0

def esg_score_row(text: str, src: str, kw_map: dict, analyzer) -> float:
    kw = keyword_score(text, kw_map)
    senti = analyzer.polarity_scores(text)["compound"]  # -1~1
    base = 0.6*kw["E"] + 0.5*kw["S"] + 0.4*kw["G"]
    s_adj = 3.0*senti  # 감성 보정
    w = source_weight(src)
    return round((base + s_adj) * w, 3)

def score_frame(df: pd.DataFrame, kw_map: dict) -> pd.DataFrame:
    an = SentimentIntensityAnalyzer()
    df["esg_score"] = [esg_score_row(t, s, kw_map, an) for t,s in zip(df["text"], df.get("source",""))]
    return df
