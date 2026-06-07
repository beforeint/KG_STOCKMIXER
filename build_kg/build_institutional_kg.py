# ============================================================
# 기관 공통 보유 KG 빌더 (yfinance 버전 - 무료)
# 실행 환경: 로컬 또는 서버 모두 가능
#
# 설치:  pip install yfinance numpy pandas tqdm
# 실행:  python build_institutional_kg.py
# 결과:  NASDAQ_institutional_relation.npy  → 서버로 업로드
# ============================================================

#%%
import numpy as np
import pandas as pd
import time
import yfinance as yf
from tqdm import tqdm

#%%
TICKER_FILE = 'NASDAQ_tickers_qualify_dr-0.98_min-5_smooth.csv'
OUTPUT_FILE = 'NASDAQ_institutional_relation.npy'
MIN_OVERLAP_RATIO = 0.01

#%%
tickers = pd.read_csv(TICKER_FILE, header=None)[0].tolist()
S = len(tickers)
print(f"티커 수: {S}")

#%%
print("\n기관 보유 데이터 수집 중 (yfinance)...")
holding_map = {}   # { ticker: set(institution_names) }

for ticker in tqdm(tickers):
    try:
        inst = yf.Ticker(ticker).institutional_holders
        if inst is not None and not inst.empty:
            holding_map[ticker] = set(inst['Holder'].dropna().tolist())
        else:
            holding_map[ticker] = set()
    except Exception as e:
        holding_map[ticker] = set()
    time.sleep(0.1)

#%%
print("\n공통 보유 행렬 계산 중...")
relation = np.zeros((S, S, 1), dtype=np.float32)

for i in tqdm(range(S)):
    inst_i = holding_map.get(tickers[i], set())
    if not inst_i:
        continue
    for j in range(i+1, S):
        inst_j = holding_map.get(tickers[j], set())
        if not inst_j:
            continue
        overlap = inst_i & inst_j
        union   = inst_i | inst_j
        ratio   = len(overlap) / len(union) if union else 0.0
        if ratio >= MIN_OVERLAP_RATIO:
            relation[i, j, 0] = ratio
            relation[j, i, 0] = ratio

#%%
np.save(OUTPUT_FILE, relation)
print(f"\n저장 완료: {OUTPUT_FILE}")
print(f"shape: {relation.shape}")
edges = int(np.sum(np.any(relation != 0, axis=2)))
print(f"엣지 수: {edges} / {S*S} ({100*edges/(S*S):.2f}%)")
