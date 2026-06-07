# ============================================================
# 이사회 인터락 KG 빌더 (yfinance 버전 - 무료)
# 두 기업이 임원(CEO/CFO 등)을 공유할 때 연결
#
# 설치:  pip install yfinance numpy pandas tqdm
# ============================================================

#%%
import numpy as np
import pandas as pd
import time
import yfinance as yf
from tqdm import tqdm

#%%
TICKER_FILE = 'NASDAQ_tickers_qualify_dr-0.98_min-5_smooth.csv'
OUTPUT_FILE = 'NASDAQ_board_relation.npy'

#%%
tickers = pd.read_csv(TICKER_FILE, header=None)[0].tolist()
S = len(tickers)
print(f"티커 수: {S}")

#%%
print("\n임원 데이터 수집 중 (yfinance)...")
board_map = {}   # { ticker: set(officer_names) }

for ticker in tqdm(tickers):
    try:
        info = yf.Ticker(ticker).info
        # companyOfficers 리스트에서 이름 추출
        officers = info.get('companyOfficers', [])
        names = set(o['name'] for o in officers if 'name' in o)
        board_map[ticker] = names
    except Exception as e:
        board_map[ticker] = set()
    time.sleep(0.1)

#%%
print("\n이사회 인터락 행렬 계산 중...")
relation = np.zeros((S, S, 1), dtype=np.float32)

for i in tqdm(range(S)):
    board_i = board_map.get(tickers[i], set())
    if not board_i:
        continue
    for j in range(i+1, S):
        board_j = board_map.get(tickers[j], set())
        shared = board_i & board_j
        if shared:
            w = len(shared) / min(len(board_i), len(board_j))
            relation[i, j, 0] = w
            relation[j, i, 0] = w

#%%
np.save(OUTPUT_FILE, relation)
print(f"\n저장 완료: {OUTPUT_FILE}")
print(f"shape: {relation.shape}")
edges = int(np.sum(np.any(relation != 0, axis=2)))
print(f"엣지 수: {edges}")
