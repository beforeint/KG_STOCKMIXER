#!/usr/bin/env python3
"""
S&P 500 데이터 전처리 → StockMixer 형식 변환
입력:  dataset/SP500/SP500.npy             (474, 2526, 5) eod features
       dataset/SP500/baseline_data_sp500.npy  {ticker: DataFrame(2526,6)}
       dataset/SP500/sp500_ticker.csv         (505 tickers + Sector)
출력:  dataset/SP500/eod_data.pkl, gt_data.pkl, mask_data.pkl, price_data.pkl
       relation/sector_industry/SP500_industry_relation.npy
"""
import pickle, numpy as np, pandas as pd
from pathlib import Path

SP500_DIR = Path('/gpfs/home1/pz29075/Capstone/StockMixer/dataset/SP500')
REL_DIR   = SP500_DIR / 'relation' / 'sector_industry'
REL_DIR.mkdir(parents=True, exist_ok=True)

# ── 원본 데이터 로드 ─────────────────────────────────────────────
print('Loading SP500 data...')
sp_eod = np.load(SP500_DIR / 'SP500.npy', allow_pickle=True)           # (474, 2526, 5)
bl     = np.load(SP500_DIR / 'baseline_data_sp500.npy', allow_pickle=True).item()
tk_df  = pd.read_csv(SP500_DIR / 'sp500_ticker.csv')

tickers = list(bl.keys())   # 474 tickers, same order as SP500.npy
N, T, F = sp_eod.shape
print(f'Stocks: {N}, Days: {T}, Features: {F}')

first_df = bl[tickers[0]]
dates    = pd.to_datetime(first_df.index)
print(f'Date range: {dates[0].date()} ~ {dates[-1].date()}')

# ── gt_data, mask_data, price_data 계산 ─────────────────────────
print('Computing gt/mask/price...')
adjclose = np.stack([bl[t]['adjclose'].values.astype(np.float32) for t in tickers])  # (N, T)
close    = np.stack([bl[t]['close'].values.astype(np.float32)    for t in tickers])
volume   = np.stack([bl[t]['volume'].values.astype(np.float32)   for t in tickers])

# gt: 다음날 수익률 (t+1 - t) / t
gt_data = np.zeros((N, T), dtype=np.float32)
raw_ret = (adjclose[:, 1:] - adjclose[:, :-1]) / (np.abs(adjclose[:, :-1]) + 1e-8)
gt_data[:, :-1] = np.nan_to_num(raw_ret, nan=0.0, posinf=0.0, neginf=0.0)
gt_data[:, -1]  = 0.0

# mask: volume > 0, adjclose > 0, gt not NaN
valid = (volume > 0) & (adjclose > 0) & (~np.isnan(adjclose))
# 다음날 gt가 NaN인 날도 mask=0 (수익률 계산 불가)
gt_nan_mask = np.zeros((N, T), dtype=bool)
gt_nan_mask[:, :-1] = np.isnan(raw_ret)
gt_nan_mask[:, -1]  = True  # 마지막 날은 항상 gt 없음
mask_data = (valid & ~gt_nan_mask).astype(np.float32)

# price: SP500.npy의 마지막 feature(정규화 close) 사용 — NASDAQ과 스케일 통일
# raw close($0.7~$5959)를 쓰면 get_loss gradient 스케일이 수천배 차이남
price_data = sp_eod[:, :, -1].astype(np.float32)  # (N, T), 이미 [0,1] 정규화됨

# eod_data: SP500.npy를 그대로 사용
eod_data = sp_eod.astype(np.float32)

print(f'eod_data  shape: {eod_data.shape}')
print(f'gt_data   shape: {gt_data.shape}')
print(f'mask_data shape: {mask_data.shape}')
print(f'price_data shape:{price_data.shape}')

# ── pkl 저장 ────────────────────────────────────────────────────
print('Saving pkl files...')
for name, data in [('eod_data', eod_data), ('gt_data', gt_data),
                   ('mask_data', mask_data), ('price_data', price_data)]:
    with open(SP500_DIR / f'{name}.pkl', 'wb') as f:
        pickle.dump(data, f)
    print(f'  Saved {name}.pkl  shape={data.shape}')

# ticker 목록 저장
with open(SP500_DIR / 'tickers.txt', 'w') as f:
    f.write('\n'.join(tickers))
print(f'  Saved tickers.txt  ({len(tickers)} tickers)')

# ── sector_industry KG 구축 ──────────────────────────────────────
print('\nBuilding sector_industry KG...')
ticker_set = set(tickers)
sectors = {}
for _, row in tk_df.iterrows():
    sym = row['Symbol']
    if sym in ticker_set:
        sectors.setdefault(row['Sector'], []).append(sym)

ticker_idx = {t: i for i, t in enumerate(tickers)}
unique_sectors = sorted(sectors.keys())
R = len(unique_sectors)
print(f'Sectors: {R}')
for s, ts in sectors.items():
    print(f'  {s}: {len(ts)} stocks')

mat = np.zeros((N, N, R), dtype=np.float32)
for r_idx, sec in enumerate(unique_sectors):
    sec_tickers = [t for t in sectors.get(sec, []) if t in ticker_idx]
    for t1 in sec_tickers:
        for t2 in sec_tickers:
            i, j = ticker_idx[t1], ticker_idx[t2]
            if i != j:
                mat[i, j, r_idx] = 1.0

edge_counts = (mat > 0).sum(axis=(0, 1))
print(f'Total edges: {edge_counts.sum()}')

out_path = REL_DIR / 'SP500_industry_relation.npy'
np.save(out_path, mat)
print(f'Saved: {out_path}  shape={mat.shape}')

import json
meta = {'relations': unique_sectors, 'shape': list(mat.shape),
        'edge_counts': dict(zip(unique_sectors, edge_counts.tolist())),
        'tickers': tickers}
(REL_DIR / 'SP500_industry_relation_meta.json').write_text(json.dumps(meta, indent=2))
print('Done.')
