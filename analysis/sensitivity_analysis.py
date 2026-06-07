#!/usr/bin/env python3
"""
Sensitivity Analysis: Regime Definition Robustness
두 축으로 검정
  Axis 1. Percentile cutoff  : 10 ~ 50th percentile (Low Vol 정의)
  Axis 2. Rolling window     : 5 ~ 30 days (변동성 계산 기간)

"Our Low Vol finding holds across reasonable regime definitions"
→ fig4_sensitivity_cutoff.png
→ fig4_sensitivity_window.png
→ fig4_sensitivity_combined.png  (논문 제출용 1-panel)
"""
import os, pickle
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, 'results')
OUT     = os.path.join(BASE, 'regime_plots')
DATA    = '/gpfs/home1/pz29075/Capstone/StockMixer/dataset/NASDAQ'
DATES_CSV = ('/gpfs/home1/pz29075/Capstone/StockMixer/'
             'Temporal_Relational_Stock_Ranking/data/NASDAQ_aver_line_dates.csv')
TEST_IDX = 1008

KG_CONDS = {'gat_wikidata':      ('GAT+Wiki',   '#2196F3'),
            'gat_institutional': ('GAT+Inst',   '#FF9800'),
            'gat_supply_chain':  ('GAT+Supply', '#4CAF50')}

# ── 데이터 로드 ──────────────────────────────────────────────────
def load(key):
    return np.stack([np.load(os.path.join(RESULTS, f'{key}_seed{s}_daily_ic.npy'))
                     for s in range(5)])   # (5, 237)

base = load('none_wikidata')
kg   = {k: load(k) for k in KG_CONDS}
n    = base.shape[1]

with open(os.path.join(DATA, 'gt_data.pkl'),   'rb') as f: gt   = pickle.load(f, encoding='latin1')
with open(os.path.join(DATA, 'mask_data.pkl'), 'rb') as f: mask = pickle.load(f, encoding='latin1')
mkt = np.nanmean(np.where(mask[:, TEST_IDX:TEST_IDX+n]>0,
                            gt[:, TEST_IDX:TEST_IDX+n], np.nan), axis=0)


def compute_vol(mkt_ret, window):
    return np.array([mkt_ret[max(0,i-window+1):i+1].std() for i in range(len(mkt_ret))])


def delta_ic_low_vol(arr_kg, arr_base, mkt_ret, window, pct):
    """주어진 window와 percentile cutoff로 Low Vol 구간 ΔIC와 p값 반환."""
    vol   = compute_vol(mkt_ret, window)
    cutoff = np.percentile(vol, pct)
    low_v  = vol <= cutoff
    if low_v.sum() < 10:
        return np.nan, np.nan, 0

    dkg   = arr_kg[:,  low_v].mean(axis=0)
    dbase = arr_base[:, low_v].mean(axis=0)
    diff  = dkg - dbase
    mean_d = diff.mean()
    _, p   = stats.ttest_rel(dkg, dbase)
    ci95   = stats.t.ppf(0.975, df=low_v.sum()-1) * diff.std(ddof=1) / np.sqrt(low_v.sum())
    return mean_d * 1000, ci95 * 1000, low_v.sum()


# ══════════════════════════════════════════════════════════════════
# Axis 1: Percentile cutoff sweep  (window=20 고정)
# ══════════════════════════════════════════════════════════════════
CUTOFFS = [10, 15, 20, 25, 30, 35, 40, 45, 50]
WINDOW_FIXED = 20

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Sensitivity: "Low Vol" Regime Definition Robustness', fontsize=13, fontweight='bold')

ax = axes[0]
ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.axvline(50, color='gray', linewidth=0.8, linestyle=':', alpha=0.6)   # 기준 median split

for ck, (label, color) in KG_CONDS.items():
    deltas, cis, ns = [], [], []
    for pct in CUTOFFS:
        d, ci, ndays = delta_ic_low_vol(kg[ck], base, mkt, WINDOW_FIXED, pct)
        deltas.append(d); cis.append(ci); ns.append(ndays)
    deltas, cis = np.array(deltas), np.array(cis)
    ax.plot(CUTOFFS, deltas, 'o-', color=color, linewidth=2, markersize=6, label=label)
    ax.fill_between(CUTOFFS, deltas-cis, deltas+cis, color=color, alpha=0.15)

ax.set_xlabel('Percentile cutoff for "Low Vol" (lower = stricter)', fontsize=11)
ax.set_ylabel('ΔIC × 1000 (KG − Baseline)', fontsize=11)
ax.set_title(f'Window = {WINDOW_FIXED}d  |  vary percentile cutoff', fontsize=11)
ax.legend(fontsize=10)

# 유의미 구간 음영 (10~50, 즉 전 범위가 양수이면)
y_min = ax.get_ylim()[0]
ax.fill_between(CUTOFFS, y_min, 0, alpha=0.04, color='red')
ax.annotate('our baseline\n(median split)', xy=(50, ax.get_ylim()[0]*0.9),
            ha='center', fontsize=8, color='gray')
ax.set_xticks(CUTOFFS)


# ══════════════════════════════════════════════════════════════════
# Axis 2: Rolling window sweep  (cutoff=50th percentile 고정)
# ══════════════════════════════════════════════════════════════════
WINDOWS  = [5, 7, 10, 15, 20, 25, 30]
PCT_FIXED = 50

ax = axes[1]
ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.axvline(20, color='gray', linewidth=0.8, linestyle=':', alpha=0.6)   # 기준 window

for ck, (label, color) in KG_CONDS.items():
    deltas, cis = [], []
    for w in WINDOWS:
        d, ci, _ = delta_ic_low_vol(kg[ck], base, mkt, w, PCT_FIXED)
        deltas.append(d); cis.append(ci)
    deltas, cis = np.array(deltas), np.array(cis)
    ax.plot(WINDOWS, deltas, 's-', color=color, linewidth=2, markersize=6, label=label)
    ax.fill_between(WINDOWS, deltas-cis, deltas+cis, color=color, alpha=0.15)

ax.set_xlabel('Rolling window for volatility estimation (days)', fontsize=11)
ax.set_ylabel('ΔIC × 1000 (KG − Baseline)', fontsize=11)
ax.set_title(f'Cutoff = {PCT_FIXED}th pct (median)  |  vary window', fontsize=11)
ax.legend(fontsize=10)
ax.set_xticks(WINDOWS)
ax.annotate('our baseline\n(20d window)', xy=(20, ax.get_ylim()[0]*0.9),
            ha='center', fontsize=8, color='gray')

plt.tight_layout()
path = os.path.join(OUT, 'fig4_sensitivity.png')
plt.savefig(path, dpi=200, bbox_inches='tight')
plt.close()
print(f'[saved] {path}')


# ══════════════════════════════════════════════════════════════════
# Summary table: all (cutoff, window) combinations → p-value grid
# ══════════════════════════════════════════════════════════════════
print('\n=== GAT+Wiki: ΔIC × 1000 (p-value)  —  Low Vol regime ===')
print(f"{'':>10}", end='')
for pct in [25, 33, 40, 50]:
    print(f"  pct={pct:2d}", end='')
print()
for w in [10, 15, 20, 25, 30]:
    print(f"win={w:3d}d  ", end='')
    for pct in [25, 33, 40, 50]:
        vol    = compute_vol(mkt, w)
        low_v  = vol <= np.percentile(vol, pct)
        dkg    = kg['gat_wikidata'][:, low_v].mean(axis=0)
        dbase  = base[:, low_v].mean(axis=0)
        d_mean = (dkg - dbase).mean() * 1000
        _, p   = stats.ttest_rel(dkg, dbase)
        star   = '†' if p<0.10 else ('*' if p<0.05 else ' ')
        print(f"  {d_mean:+5.2f}{star}", end='')
    print()
print('† p<0.10   * p<0.05')
print('\nDone.')
