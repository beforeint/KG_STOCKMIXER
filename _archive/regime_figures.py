#!/usr/bin/env python3
"""
Publication-quality figures for Regime Analysis
Fig 1. Effect-size Heatmap  (KG × Regime × ΔIC, cell = Cohen's d)
Fig 2. Forest Plot          (regime별 ΔIC + 95% CI + p-value)
Fig 3. IC Time Series       (rolling IC + regime overlay)
"""
import os, pickle
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

# ── 경로 ────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, 'results')
OUT     = os.path.join(BASE, 'regime_plots')
DATA    = '/gpfs/home1/pz29075/Capstone/StockMixer/dataset/NASDAQ'
DATES_CSV = ('/gpfs/home1/pz29075/Capstone/StockMixer/'
             'Temporal_Relational_Stock_Ranking/data/NASDAQ_aver_line_dates.csv')
TEST_IDX  = 1008
EARNINGS  = [('2017-01-09','2017-02-10'), ('2017-04-10','2017-05-12'),
             ('2017-07-10','2017-08-11'), ('2017-10-09','2017-10-27')]

KG_LABELS = {'gat_wikidata': 'GAT+Wiki',
             'gat_institutional': 'GAT+Inst',
             'gat_supply_chain': 'GAT+Supply'}
REGIME_ORDER = ['All', 'Bull', 'Bear', 'High Vol', 'Low Vol', 'Earnings', 'Normal']
PALETTE = {'gat_wikidata': '#2196F3', 'gat_institutional': '#FF9800',
           'gat_supply_chain': '#4CAF50'}

# ── 데이터 로드 ──────────────────────────────────────────────────
def load(key, seeds=range(5)):
    return np.stack([np.load(os.path.join(RESULTS, f'{key}_seed{s}_daily_ic.npy'))
                     for s in seeds])   # (5, 237)

base = load('none_wikidata')
kg   = {k: load(k) for k in KG_LABELS}
n    = base.shape[1]

with open(os.path.join(DATA, 'gt_data.pkl'),   'rb') as f: gt   = pickle.load(f, encoding='latin1')
with open(os.path.join(DATA, 'mask_data.pkl'), 'rb') as f: mask = pickle.load(f, encoding='latin1')
mkt  = np.nanmean(np.where(mask[:, TEST_IDX:TEST_IDX+n]>0,
                             gt[:, TEST_IDX:TEST_IDX+n], np.nan), axis=0)
rvol = np.array([mkt[max(0,i-19):i+1].std() for i in range(n)])
dates_raw = pd.read_csv(DATES_CSV, header=None)[0]
td   = pd.to_datetime(dates_raw.iloc[TEST_IDX:TEST_IDX+n].values)
earn = np.zeros(n, dtype=bool)
for s,e in EARNINGS:
    earn |= (td>=pd.Timestamp(s))&(td<=pd.Timestamp(e))

REGIMES = {'All': np.ones(n,bool), 'Bull': mkt>0, 'Bear': mkt<=0,
           'High Vol': rvol>=np.median(rvol), 'Low Vol': rvol<np.median(rvol),
           'Earnings': earn, 'Normal': ~earn}


def regime_stats(arr_kg, arr_base, rmask):
    """daily-level paired t-test → ΔIC, Cohen's d, 95% CI, p"""
    dkg  = arr_kg[:,  rmask].mean(axis=0)   # seed-mean daily IC in regime
    dbase= arr_base[:,rmask].mean(axis=0)
    diff = dkg - dbase
    n_r  = rmask.sum()
    mean_d, std_d = diff.mean(), diff.std(ddof=1)
    se   = std_d / np.sqrt(n_r)
    ci95 = stats.t.ppf(0.975, df=n_r-1) * se
    _, p = stats.ttest_rel(dkg, dbase)
    # Cohen's d (pooled std of daily IC difference)
    cohens_d = mean_d / (std_d + 1e-9)
    return dict(delta=mean_d, cohens_d=cohens_d, ci=ci95, p=p, n=n_r)


def sig_star(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    if p < 0.10:  return '†'
    return ''


# ══════════════════════════════════════════════════════════════════
# Fig 1 — Effect-size Heatmap
# ══════════════════════════════════════════════════════════════════
def fig_heatmap():
    rows = REGIME_ORDER
    cols = list(KG_LABELS.keys())
    delta = np.zeros((len(rows), len(cols)))
    pvals = np.zeros((len(rows), len(cols)))

    for i, rn in enumerate(rows):
        for j, ck in enumerate(cols):
            s = regime_stats(kg[ck], base, REGIMES[rn])
            delta[i, j] = s['delta'] * 1000    # ×1000 → milli-IC 단위
            pvals[i, j] = s['p']

    fig, ax = plt.subplots(figsize=(7, 5))
    norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=8)
    im = ax.imshow(delta, cmap='RdYlGn', norm=norm, aspect='auto')

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([KG_LABELS[c] for c in cols], fontsize=11)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=11)

    for i in range(len(rows)):
        for j in range(len(cols)):
            star = sig_star(pvals[i, j])
            val  = delta[i, j]
            txt  = f'{val:+.2f}{star}' if val != 0 else '0.00'
            color = 'white' if abs(val) > 4 else 'black'
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=10, fontweight='bold', color=color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('ΔIC × 1000 (KG − Baseline)', fontsize=9)
    ax.set_title('Effect Size Heatmap: KG vs Baseline', fontsize=13, pad=10)
    ax.set_xlabel('')
    # 하단 주석
    ax.annotate('† p<0.10   * p<0.05   ** p<0.01   *** p<0.001  (daily-level paired t-test)',
                xy=(0.5, -0.07), xycoords='axes fraction',
                ha='center', fontsize=8, color='gray')

    plt.tight_layout()
    path = os.path.join(OUT, 'fig1_heatmap.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[saved] {path}')


# ══════════════════════════════════════════════════════════════════
# Fig 2 — Forest Plot
# ══════════════════════════════════════════════════════════════════
def fig_forest():
    regimes_show = ['All', 'Bull', 'Bear', 'Low Vol', 'High Vol', 'Earnings', 'Normal']
    n_r, n_kg = len(regimes_show), len(KG_LABELS)
    offsets = np.linspace(-0.25, 0.25, n_kg)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

    for ki, (ck, label) in enumerate(KG_LABELS.items()):
        color = PALETTE[ck]
        for ri, rn in enumerate(regimes_show):
            s = regime_stats(kg[ck], base, REGIMES[rn])
            y   = (n_r - 1 - ri) + offsets[ki]
            d   = s['delta'] * 1000
            ci  = s['ci']   * 1000
            star = sig_star(s['p'])

            ax.errorbar(d, y, xerr=ci, fmt='o', color=color,
                        markersize=6, linewidth=1.5, capsize=4,
                        label=label if ri == 0 else '_')
            if star:
                ax.text(d + ci + 0.15, y, star, va='center',
                        fontsize=10, color=color, fontweight='bold')

    ax.set_yticks(range(n_r))
    ax.set_yticklabels(list(reversed(regimes_show)), fontsize=11)
    ax.set_xlabel('ΔIC × 1000 (KG − Baseline)  with 95% CI', fontsize=11)
    ax.set_title('Forest Plot: Regime-Conditional Effect Size', fontsize=13)
    ax.legend(fontsize=10, loc='lower right')

    # Regime 구분선
    for i in range(n_r - 1):
        ax.axhline(i + 0.5, color='gray', linewidth=0.4, linestyle=':')

    # Low Vol 행 강조
    lv_idx = n_r - 1 - regimes_show.index('Low Vol')
    ax.axhspan(lv_idx - 0.45, lv_idx + 0.45, alpha=0.07, color='green')
    ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] < -2 else -3,
            lv_idx, '← key', va='center', fontsize=8, color='green', style='italic')

    ax.annotate('† p<0.10   * p<0.05   ** p<0.01   *** p<0.001',
                xy=(0.98, 0.02), xycoords='axes fraction',
                ha='right', fontsize=8, color='gray')

    plt.tight_layout()
    path = os.path.join(OUT, 'fig2_forest.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[saved] {path}')


# ══════════════════════════════════════════════════════════════════
# Fig 3 — IC Time Series with Regime Overlay
# ══════════════════════════════════════════════════════════════════
def fig_timeseries():
    import matplotlib.dates as mdates
    td_np = td.to_numpy().astype('datetime64[D]')

    fig = plt.figure(figsize=(13, 7))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 0.6], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # ── 어닝시즌 음영 (전 패널) ──
    for s, e in EARNINGS:
        for ax in (ax1, ax2, ax3):
            ax.axvspan(np.datetime64(s), np.datetime64(e),
                       alpha=0.13, color='gold', zorder=0)

    # ── ax1: Rolling IC ──
    base_roll = pd.Series(base.mean(axis=0)).rolling(20, min_periods=5).mean().values
    ax1.plot(td_np, base_roll, color='#555555', linewidth=2,
             label='Baseline (No KG)', zorder=3)
    for ck, label in KG_LABELS.items():
        roll = pd.Series(kg[ck].mean(axis=0)).rolling(20, min_periods=5).mean().values
        ax1.plot(td_np, roll, color=PALETTE[ck], linewidth=1.8,
                 label=label, zorder=3)
    ax1.axhline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.6)
    # Low Vol 기간 배경 (연초록)
    lv = REGIMES['Low Vol']
    ax1.fill_between(td_np, ax1.get_ylim()[0] if ax1.get_ylim()[0] < -0.04 else -0.06,
                     0.09, where=lv, alpha=0.06, color='green', zorder=0)
    ax1.set_ylabel('Rolling IC (20d)', fontsize=11)
    ax1.set_ylim(-0.06, 0.10)
    handles, labels_ = ax1.get_legend_handles_labels()
    handles += [mpatches.Patch(color='gold', alpha=0.4, label='Earnings Season'),
                mpatches.Patch(color='green', alpha=0.2, label='Low Vol')]
    ax1.legend(handles=handles, fontsize=9, loc='upper right', ncol=2)
    ax1.set_title('IC Time Series with Market Regime Overlay  (2016-11 ~ 2017-10)',
                  fontsize=12, pad=6)

    # ── ax2: ΔIC (GAT+Wiki − baseline), rolling 20d ──
    diff_roll = pd.Series((kg['gat_wikidata'] - base).mean(axis=0)
                          ).rolling(20, min_periods=5).mean().values
    ax2.fill_between(td_np, diff_roll, 0,
                     where=diff_roll >= 0, color='#2196F3', alpha=0.7, label='KG > Base')
    ax2.fill_between(td_np, diff_roll, 0,
                     where=diff_roll < 0,  color='#F44336', alpha=0.7, label='KG < Base')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_ylabel('ΔIC (20d)', fontsize=10)
    ax2.legend(fontsize=8, loc='upper right')

    # ── ax3: Volatility regime ──
    ax3.fill_between(td_np, rvol, 0, color='#9C27B0', alpha=0.5)
    ax3.axhline(np.median(rvol), color='black', linewidth=0.8,
                linestyle='--', label='median')
    ax3.set_ylabel('Roll. Vol', fontsize=9)
    ax3.set_xlabel('Date', fontsize=10)
    ax3.legend(fontsize=8, loc='upper right')

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    fig.autofmt_xdate(rotation=0)

    path = os.path.join(OUT, 'fig3_timeseries.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[saved] {path}')


# ── 실행 ────────────────────────────────────────────────────────
fig_heatmap()
fig_forest()
fig_timeseries()
print('\nDone. Check regime_plots/')
