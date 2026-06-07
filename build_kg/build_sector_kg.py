# ============================================================
# 섹터/업종 KG 빌더 (yfinance - 서버에서 직접 실행 가능)
# 같은 업종(industry) 종목끼리 연결
# 출력: [S, S, 1] — 같은 industry면 1, 아니면 0
# ============================================================

#%%
import numpy as np
import pandas as pd
import time
import yfinance as yf
from tqdm import tqdm
from collections import defaultdict

#%%
DATA_ROOT   = '/gpfs/home1/pz29075/Capstone/StockMixer/dataset'
TICKER_FILE = '/gpfs/home1/pz29075/Capstone/StockMixer/Temporal_Relational_Stock_Ranking/data/NASDAQ_tickers_qualify_dr-0.98_min-5_smooth.csv'
OUTPUT_DIR  = f'{DATA_ROOT}/NASDAQ/relation/sector_industry'
OUTPUT_FILE = f'{OUTPUT_DIR}/NASDAQ_industry_relation.npy'

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

#%%
tickers = pd.read_csv(TICKER_FILE, header=None)[0].tolist()
S = len(tickers)
print(f"티커 수: {S}")

#%%
print("\n섹터/업종 데이터 수집 중 (yfinance)...")
industry_map = {}  # ticker → industry string

for ticker in tqdm(tickers):
    try:
        info = yf.Ticker(ticker).info
        industry_map[ticker] = info.get('industry', '') or ''
    except Exception:
        industry_map[ticker] = ''
    time.sleep(0.05)

# 업종별 티커 그룹
groups = defaultdict(list)
for i, t in enumerate(tickers):
    ind = industry_map[t]
    if ind:
        groups[ind].append(i)

print(f"유효 업종 수: {len(groups)}")
for ind, members in sorted(groups.items(), key=lambda x: -len(x[1]))[:10]:
    print(f"  {ind}: {len(members)}개")

#%%
print("\n관계 행렬 계산 중...")
relation = np.zeros((S, S, 1), dtype=np.float32)

for ind, members in groups.items():
    if len(members) < 2:
        continue
    for ii in range(len(members)):
        for jj in range(ii + 1, len(members)):
            i, j = members[ii], members[jj]
            relation[i, j, 0] = 1.0
            relation[j, i, 0] = 1.0

#%%
np.save(OUTPUT_FILE, relation)
print(f"\n저장 완료: {OUTPUT_FILE}")
print(f"shape: {relation.shape}")
edges = int(np.sum(np.any(relation != 0, axis=2)))
print(f"엣지 수: {edges} / {S*S} ({100*edges/(S*S):.2f}%)")
