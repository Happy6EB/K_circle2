import argparse
import pandas as pd
from preprocess import load_news
from scorer import load_keywords, score_frame
from greenwash import is_greenwash

def main(in_csv, kw_yaml, out_csv):
    # 데이터 로드
    df = load_news(in_csv)
    kws = load_keywords(kw_yaml)
    df = score_frame(df, kws)

    # 기업별 집계
    df["date"] = pd.to_datetime(df["date"])
    agg = (df.groupby("company")
              .agg(esg_avg=("esg_score","mean"),
                   esg_last=("esg_score","last"),
                   n_docs=("esg_score","count"))
              .reset_index()
          )

    # 그린워싱 플래그
    flags = []
    for c, g in df.groupby("company"):
        flags.append({"company":c, "greenwash_flag": int(is_greenwash(g))})
    flags = pd.DataFrame(flags)

    out = agg.merge(flags, on="company", how="left").sort_values("esg_last", ascending=False)
    out.to_csv(out_csv, index=False)
    print(f"[DONE] saved: {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", default="data/news_samples.csv")
    ap.add_argument("--kw", dest="kw_yaml", default="data/esg_keywords.yaml")
    ap.add_argument("--out", dest="out_csv", default="results.csv")
    args = ap.parse_args()
    main(args.in_csv, args.kw_yaml, args.out_csv)
