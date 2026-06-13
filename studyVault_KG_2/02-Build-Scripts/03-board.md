# 03. Board Interlock KG 빌더

> 순서: 3/6 (02-Build-Scripts)  
> 파일: `build_kg/build_board_kg.py`  
> 출력: `NASDAQ_board_relation.npy`  shape=(1026,1026,1)

---

## 핵심 아이디어

> 두 회사가 동일한 임원(CEO/CFO/이사 등)을 공유하면 연결한다 (Board Interlock)

---

## 전체 코드

```python
import numpy as np
import pandas as pd
import time
import yfinance as yf
from tqdm import tqdm

TICKER_FILE = 'NASDAQ_tickers_qualify_dr-0.98_min-5_smooth.csv'
OUTPUT_FILE = 'NASDAQ_board_relation.npy'

tickers = pd.read_csv(TICKER_FILE, header=None)[0].tolist()
S = len(tickers)

# Step 1: ticker → 임원 이름 집합
board_map = {}
for ticker in tqdm(tickers):
    try:
        info = yf.Ticker(ticker).info
        officers = info.get('companyOfficers', [])
        names = set(o['name'] for o in officers if 'name' in o)
        board_map[ticker] = names
    except Exception:
        board_map[ticker] = set()
    time.sleep(0.1)

# Step 2: 공유 임원 기반 관계 행렬
relation = np.zeros((S, S, 1), dtype=np.float32)
for i in tqdm(range(S)):
    board_i = board_map.get(tickers[i], set())
    if not board_i:
        continue
    for j in range(i+1, S):
        board_j = board_map.get(tickers[j], set())
        shared = board_i & board_j   # 공유 임원 집합
        if shared:
            # 값 = 공유 임원 수 / min(임원 수)
            w = len(shared) / min(len(board_i), len(board_j))
            relation[i, j, 0] = w
            relation[j, i, 0] = w

np.save(OUTPUT_FILE, relation)
```

---

## 가중치 계산 방식

```
w = |shared| / min(|board_i|, |board_j|)
```

- 두 회사 중 더 작은 이사회 기준 공유 비율
- 값 범위: (0, 1]

---

## 결과 통계

- shape: (1026, 1026, 1)
- 엣지 수: **21쌍** (매우 sparse)
- 고립 노드: 1,002개 (97.7%)
- 최대 연결요소: 5개 (0.5%)
- 값 범위: [0.1, 1.0]

---

## 주의: 실험 신뢰도 낮음

Board KG는 엣지가 21개에 불과해 GCN/GAT 실효성이 거의 없다.

- 원인: 구 ticker(상장폐지) 97.7%가 `companyOfficers` 반환 없음
- 실제 board interlock은 존재하지만 yfinance가 현재 임원 데이터만 반환

**실험 시 참고 수준으로만 사용할 것.**

---

## Institutional과의 비교

| | Institutional | Board |
|--|--------------|-------|
| 관계 기준 | 공통 기관투자자 | 공통 임원 |
| 엣지 수 | 154,032 | 21 |
| 커버리지 | 56.8% | 0.5% |
| 실용성 | 높음 | 낮음 |
