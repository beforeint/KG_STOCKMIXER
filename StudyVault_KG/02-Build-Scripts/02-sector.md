# 02. Sector/Industry KG 빌더

> 순서: 2/6 (02-Build-Scripts)  
> 파일: `build_kg/build_sector_kg.py`  
> 출력: `NASDAQ_industry_relation.npy`  shape=(1026,1026,1)

---

## 핵심 아이디어

> 같은 업종(industry)에 속한 종목끼리 연결한다. 값은 이진(0/1).

---

## 전체 코드

```python
import numpy as np
import pandas as pd
import time
import yfinance as yf
from tqdm import tqdm
from collections import defaultdict

DATA_ROOT   = '/gpfs/home1/pz29075/Capstone/StockMixer/dataset'
TICKER_FILE = '.../NASDAQ_tickers_qualify_dr-0.98_min-5_smooth.csv'
OUTPUT_DIR  = f'{DATA_ROOT}/NASDAQ/relation/sector_industry'
OUTPUT_FILE = f'{OUTPUT_DIR}/NASDAQ_industry_relation.npy'

tickers = pd.read_csv(TICKER_FILE, header=None)[0].tolist()
S = len(tickers)

# Step 1: ticker → industry 문자열
industry_map = {}
for ticker in tqdm(tickers):
    try:
        info = yf.Ticker(ticker).info
        industry_map[ticker] = info.get('industry', '') or ''
    except Exception:
        industry_map[ticker] = ''
    time.sleep(0.05)

# Step 2: industry 그룹별 ticker 묶기
groups = defaultdict(list)
for i, t in enumerate(tickers):
    ind = industry_map[t]
    if ind:
        groups[ind].append(i)

# Step 3: 같은 그룹 내 모든 쌍 연결 (Binary)
relation = np.zeros((S, S, 1), dtype=np.float32)
for ind, members in groups.items():
    if len(members) < 2:
        continue
    for ii in range(len(members)):
        for jj in range(ii + 1, len(members)):
            i, j = members[ii], members[jj]
            relation[i, j, 0] = 1.0
            relation[j, i, 0] = 1.0

np.save(OUTPUT_FILE, relation)
```

---

## 주의: 이 스크립트는 실제로 0 엣지 생성

build_sector_kg.py를 그대로 실행하면 **yfinance 실패로 엣지가 0개** 나온다.

**원인**: NASDAQ 1026개 중 ~50%가 상장폐지 종목 → `info.get('industry', '')` = ''  
→ 유효한 industry 그룹이 형성되지 않음

**실제 해결책**: Wikidata KG에서 dim24 (P452=industry) 슬라이스 사용

```python
# 실제로 사용한 방법 (yfinance 대신)
wiki = np.load('NASDAQ_wiki_relation.npy')   # shape: (1026, 1026, 43)
sector_kg = wiki[:, :, 24:25]               # P452 = industry (dim index 24)
np.save('NASDAQ_industry_relation.npy', sector_kg)
```

---

## 결과 통계 (Wikidata dim24 방식)

- shape: (1026, 1026, 1)
- 엣지 수: 253개 무방향 쌍 (506개 방향성 엣지)
- 고립 노드: 952개 (92.8%) — Wikidata QID 없는 종목은 전부 고립
- 값 타입: Binary {0.0, 1.0}
- 자기 연결(self-loop): 없음

---

## 실행 (수정 버전)

```bash
# yfinance 방식은 쓰지 않는다
# Wikidata dim24 추출 방식 사용:
python - <<'EOF'
import numpy as np
wiki = np.load('/path/to/NASDAQ_wiki_relation.npy')
sector_kg = wiki[:, :, 24:25]
np.save('/path/to/NASDAQ_industry_relation.npy', sector_kg)
print(f'shape: {sector_kg.shape}')
EOF
```
