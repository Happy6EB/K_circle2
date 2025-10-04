import numpy as np
import pandas as pd

PROMO = {"혁신","선도","세계적","친환경 경영","지속가능 리더"}

def is_greenwash(group: pd.DataFrame) -> bool:
    # 규칙 예시: 점수 급등 & 실증 키워드(감축량/인증/감사) 언급 부족
    g = group.sort_values("date")
    scores = g["esg_score"].values
    if len(scores) < 3: return False
    jump = (scores[-1] - np.mean(scores[:-1])) > (np.std(scores[:-1])*1.5 + 2)
    promo_hits = g["text"].str.contains("|".join(PROMO)).sum()
    evidence = g["text"].str.contains("감축|배출량|인증|감사|공인|검증").sum()
    return bool(jump and promo_hits >= 2 and evidence == 0)
