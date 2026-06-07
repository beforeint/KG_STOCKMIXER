#!/usr/bin/env python3
"""Fig 3 (LLM 포함) — IC Time Series with Regime Overlay"""
import os, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

BASE      = os.path.dirname(os.path.abspath(__file__))
RESULTS   = os.path.join(BASE, 'results')
OUT       = os.path.join(BASE, 'regime_plots')
DATA      = '/gpfs/home1/pz29075/Capstone/StockMixer/dataset/NASDAQ'
DATES_CSV = ('/gpfs/home1/pz29075/Capstone/StockMixer/'
             'Temporal_Relational_Stock_Ranking/data/NASDAQ_aver_line_dates.csv')
TEST_IDX  = 1008
EARNINGS  = [('2017-01-09','2017-02-10'), ('2017-04-10','2017-05-12'),
             ('2017-07-10','2017-08-11'), ('2017-10-09','2017-10-27')]

# ── 조건 (LLM 추가) ────────────────────────────────────────────
CONDITIONS = {
    'gat_wikidata':      ('GAT + Wikidata',    '#2196F3', 1.8, '-'),
    'gat_institutional': ('GAT + Institutional','#FF9800', 1.5, '--'),
    'gat_supply_chain':  ('GAT + Supply Chain', '#4CAF50', 1.5, '--'),
    'gat_llm':           ('GAT + LLM V1',       '#9C27B0', 1.5, ':'),
    'gat_llm_v2':        ('GAT + LLM V2 ★',     '#E91E63', 2.0, '-'),
}

# ── 데이터 로드 ────────────────────────────────────────────────
def load(key):
    ics = []
    for s in range(5):
        p = os.path.join(RESULTS, f'{key}_seed{s}_daily_ic.npy')
        if os.path.exists(p):
            ics.append(np.load(p))
    return np.stack(ics)   # (5, T)

base = load('none_wikidata')
kg   = {k: load(k) for k in CONDITIONS}
n    = base.shape[1]

with open(f'{DATA}/gt_data.pkl','rb')   as f: gt   = pickle.load(f, encoding='latin1')
with open(f'{DATA}/mask_data.pkl','rb') as f: mask = pickle.load(f, encoding='latin1')
mkt  = np.nanmean(np.where(mask[:,TEST_IDX:TEST_IDX+n]>0,
                            gt[:,TEST_IDX:TEST_IDX+n], np.nan), axis=0)
rvol = np.array([mkt[max(0,i-19):i+1].std() for i in range(n)])

dates_raw = pd.read_csv(DATES_CSV, header=None)[0]
td   = pd.to_datetime(dates_raw.iloc[TEST_IDX:TEST_IDX+n].values)
td_np = td.to_numpy().astype('datetime64[D]')

earn = np.zeros(n, dtype=bool)
for s,e in EARNINGS:
    earn |= (td>=pd.Timestamp(s)) & (td<=pd.Timestamp(e))
lv_mask = rvol < np.median(rvol)

# ── Figure ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 8))
gs  = gridspec.GridSpec(3, 1, height_ratios=[3.5, 1.2, 0.7], hspace=0.06)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax3 = fig.add_subplot(gs[2], sharex=ax1)

# 어닝시즌 음영 (전 패널)
for s,e in EARNINGS:
    for ax in (ax1, ax2, ax3):
        ax.axvspan(np.datetime64(s), np.datetime64(e),
                   alpha=0.12, color='gold', zorder=0)

# Low-Vol 배경 (ax1, ax2)
for ax in (ax1, ax2):
    ax.fill_between(td_np, -0.08, 0.12, where=lv_mask,
                    alpha=0.07, color='#4CAF50', zorder=0)

# ── ax1: Rolling IC ────────────────────────────────────────────
W = 20
base_roll = pd.Series(base.mean(axis=0)).rolling(W, min_periods=5).mean().values
ax1.plot(td_np, base_roll, color='#555555', linewidth=2.2,
         label='Baseline (No KG)', zorder=4, linestyle='-')

for ck, (label, color, lw, ls) in CONDITIONS.items():
    roll = pd.Series(kg[ck].mean(axis=0)).rolling(W, min_periods=5).mean().values
    ax1.plot(td_np, roll, color=color, linewidth=lw,
             label=label, zorder=3, linestyle=ls)

ax1.axhline(0, color='black', linewidth=0.7, linestyle='--', alpha=0.5)
ax1.set_ylabel('Rolling IC (20d)', fontsize=11)
ax1.set_ylim(-0.07, 0.11)

# 범례
handles, labels_ = ax1.get_legend_handles_labels()
handles += [mpatches.Patch(color='#4CAF50', alpha=0.25, label='Low-Vol Regime'),
            mpatches.Patch(color='gold',    alpha=0.35, label='Earnings Season')]
ax1.legend(handles=handles, fontsize=8.5, loc='upper right', ncol=3,
           framealpha=0.9)
ax1.set_title('IC Time Series with Market Regime Overlay  (NASDAQ, 2016-11 ~ 2017-10)',
              fontsize=12, pad=8)

# LV 구간 IC 평균 annotation (LLM V2 vs Baseline)
lv_base_mean = base.mean(axis=0)[lv_mask].mean()
lv_llm2_mean = kg['gat_llm_v2'].mean(axis=0)[lv_mask].mean()
ax1.annotate(f'LV avg: Baseline={lv_base_mean:.4f}\nLLM V2={lv_llm2_mean:.4f} (+{(lv_llm2_mean/lv_base_mean-1)*100:.0f}%)',
             xy=(td_np[int(n*0.35)], 0.075),
             fontsize=8.5, color='#E91E63',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# ── ax2: ΔIC (LLM V2 − baseline), rolling 20d ─────────────────
diff_llm2 = pd.Series((kg['gat_llm_v2'] - base).mean(axis=0)
                      ).rolling(W, min_periods=5).mean().values
diff_wiki  = pd.Series((kg['gat_wikidata'] - base).mean(axis=0)
                       ).rolling(W, min_periods=5).mean().values

ax2.fill_between(td_np, diff_llm2, 0,
                 where=diff_llm2>=0, color='#E91E63', alpha=0.65, label='LLM V2 > Base')
ax2.fill_between(td_np, diff_llm2, 0,
                 where=diff_llm2<0,  color='#E91E63', alpha=0.25, label='LLM V2 < Base')
ax2.plot(td_np, diff_wiki, color='#2196F3', linewidth=1.2,
         linestyle='--', alpha=0.7, label='Wiki ΔIC (ref)')
ax2.axhline(0, color='black', linewidth=0.5)
ax2.set_ylabel('ΔIC (20d)', fontsize=10)
ax2.legend(fontsize=8, loc='upper right', ncol=3)

# ── ax3: Volatility ───────────────────────────────────────────
ax3.fill_between(td_np, rvol, 0, color='#9C27B0', alpha=0.45)
ax3.axhline(np.median(rvol), color='black', linewidth=0.9,
            linestyle='--', label=f'median={np.median(rvol):.4f}')
ax3.set_ylabel('Roll. Vol', fontsize=9)
ax3.set_xlabel('Date', fontsize=10)
ax3.legend(fontsize=8, loc='upper right')

# x축 포맷
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax3.get_xticklabels(), rotation=0, ha='center', fontsize=9)

path = os.path.join(OUT, 'fig3_llm_timeseries.png')
plt.savefig(path, dpi=200, bbox_inches='tight')
plt.close()
print(f'[saved] {path}')
