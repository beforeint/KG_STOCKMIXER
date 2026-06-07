# 01. Institutional KG 빌더

> 순서: 1/6 (02-Build-Scripts)  
> 파일: `build_kg/build_institutional_kg.py`  
> 출력: `NASDAQ_institutional_relation.npy`  shape=(1026,1026,1)

---

## 핵심 아이디어

> 두 종목이 같은 기관투자자에게 동시에 보유되고 있으면 "관련 종목"으로 본다

- 데이터 소스: `yf.Ticker(ticker).institutional_holders`
- 관계 강도: Jaccard 유사도 (두 종목 기관 집합의 교집합/합집합)

---

## 전체 코드

```python
import numpy as np
import pandas as pd
import time
import yfinance as yf
from tqdm import tqdm

TICKER_FILE = 'NASDAQ_tickers_qualify_dr-0.98_min-5_smooth.csv'
OUTPUT_FILE = 'NASDAQ_institutional_relation.npy'
MIN_OVERLAP_RATIO = 0.01   # Jaccard 임계값 1%

tickers = pd.read_csv(TICKER_FILE, header=None)[0].tolist()
S = len(tickers)  # 1026

# Step 1: 기관 집합 수집
holding_map = {}   # { ticker: set(institution_names) }
for ticker in tqdm(tickers):
    try:
        inst = yf.Ticker(ticker).institutional_holders
        if inst is not None and not inst.empty:
            holding_map[ticker] = set(inst['Holder'].dropna().tolist())
        else:
            holding_map[ticker] = set()
    except Exception:
        holding_map[ticker] = set()
    time.sleep(0.1)

# Step 2: Jaccard 유사도 행렬 계산
relation = np.zeros((S, S, 1), dtype=np.float32)
for i in tqdm(range(S)):
    inst_i = holding_map.get(tickers[i], set())
    if not inst_i:
        continue
    for j in range(i+1, S):
        inst_j = holding_map.get(tickers[j], set())
        if not inst_j:
            continue
        overlap = inst_i & inst_j          # 교집합
        union   = inst_i | inst_j          # 합집합
        ratio   = len(overlap) / len(union) if union else 0.0
        if ratio >= MIN_OVERLAP_RATIO:
            relation[i, j, 0] = ratio
            relation[j, i, 0] = ratio      # 대칭

np.save(OUTPUT_FILE, relation)
```

---

## "동시에 보유"를 어떻게 아는가?

yfinance의 `institutional_holders`는 **SEC 13F 공시** 기반 데이터를 반환한다.

```
Holder              Shares   Date Reported   pctHeld
Vanguard Group      1.2B     2024-09-30      7.5%
BlackRock           0.9B     2024-09-30      5.6%
State Street        0.5B     2024-09-30      3.2%
```

동시에 실시간 확인하는 게 **아니라**, 데이터 수집 시점 기준 각 종목의 대주주 목록을 집합으로 만들어 교집합을 본다:

```python
holding_map["AAPL"] = {"Vanguard", "BlackRock", "State Street"}
holding_map["MSFT"] = {"Vanguard", "BlackRock", "Fidelity"}

overlap = {"Vanguard", "BlackRock"}      # 2개
union   = {"Vanguard", "BlackRock", "State Street", "Fidelity"}  # 4개
Jaccard = 2 / 4 = 0.50  →  엣지 연결 (0.50 ≥ 0.01)
```

---

## 주요 파라미터

| 파라미터 | 값 | 의미 |
|---------|-----|------|
| `MIN_OVERLAP_RATIO` | 0.01 | Jaccard ≥ 1% 인 쌍만 엣지 |
| `time.sleep(0.1)` | 0.1초 | yfinance rate limit 방지 |

---

## 한계

1. yfinance는 최근 분기 13F 기준 **상위 N개** 기관만 반환 (전수 아님)
2. 수집 시점이 모두 동일하지 않을 수 있음 (기관마다 보고 날짜 다름)
3. 기관 이름 문자열이 조금이라도 다르면 (`"Vanguard Group"` vs `"Vanguard Group Inc."`) 다른 기관으로 처리
4. 구 ticker (상장폐지) → yfinance 404 → `holding_map[t] = set()` → 고립 노드
   - NASDAQ 1026개 중 443개(43.2%) 고립 발생

---

## 결과 통계

- shape: (1026, 1026, 1)
- 엣지 수: 154,032 (무방향)
- Density: 29.3%
- 값 범위: [0.053, 1.0]
- 최대 연결요소: 583개 (56.8%)

---

## 실행 환경

```bash
pip install "yfinance==0.2.18" "multitasking==0.0.9"
cd build_kg/
python build_institutional_kg.py
```

> Python 3.8 서버에서 최신 yfinance는 `TypeError: 'type' object is not subscriptable` 발생 → 반드시 0.2.18 사용
