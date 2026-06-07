#!/usr/bin/env python3
"""
Effect Size Analysis
  - Hedges' g  : seed-level (n=5), small-sample correction 적용
  - Cohen's d  : daily-level (n=days), 비교용
  - Bootstrap 95% CI : temporal 자기상관 고려한 non-parametric CI
  - FDR correction   : Benjamini-Hochberg (21 comparisons)
  - Improved forest plot with proper CIs
"""
import os, pickle
import numpy as np
import pandas as pd
from scipy import stats
from itertools import product
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, 'results')
OUT     = os.path.join(BASE, 'regime_plots')
DATA    = '/gpfs/home1/pz29075/Capstone/StockMixer/dataset/NASDAQ'
DATES_CSV = ('/gpfs/home1/pz29075/Capstone/StockMixer/'
             'Temporal_Relational_Stock_Ranking/data/NASDAQ_aver_line_dates.csv')
TEST_IDX = 1008
EARNINGS = [('2017-01-09','2017-02-10'), ('2017-04-10','2017-05-12'),
            ('2017-07-10','2017-08-11'), ('2017-10-09','2017-10-27')]

KG_CONDS = {'gat_wikidata':      ('GAT+Wiki',   '#2196F3'),
            'gat_institutional': ('GAT+Inst',   '#FF9800'),
            'gat_supply_chain':  ('GAT+Supply', '#4CAF50')}
REGIME_ORDER = ['All', 'Bull', 'Bear', 'High Vol', 'Low Vol', 'Earnings', 'Normal']

# ── 데이터 로드 ──────────────────────────────────────────────────
def load(key):
    return np.stack([np.load(os.path.join(RESULTS, f'{key}_seed{s}_daily_ic.npy'))
                     for s in range(5)])   # (5, 237)

base = load('none_wikidata')
kg   = {k: load(k) for k in KG_CONDS}
n    = base.shape[1]

with open(os.path.join(DATA, 'gt_data.pkl'),   'rb') as f: gt   = pickle.load(f, encoding='latin1')
with open(os.path.join(DATA, 'mask_data.pkl'), 'rb') as f: mask = pickle.load(f, encoding='latin1')
mkt  = np.nanmean(np.where(mask[:,TEST_IDX:TEST_IDX+n]>0,
                             gt[:,TEST_IDX:TEST_IDX+n], np.nan), axis=0)
rvol = np.array([mkt[max(0,i-19):i+1].std() for i in range(n)])
td   = pd.to_datetime(pd.read_csv(DATES_CSV, header=None)[0]
                        .iloc[TEST_IDX:TEST_IDX+n].values)
earn = np.zeros(n, dtype=bool)
for s,e in EARNINGS:
    earn |= (td>=pd.Timestamp(s))&(td<=pd.Timestamp(e))

REGIMES = {'All': np.ones(n,bool), 'Bull': mkt>0, 'Bear': mkt<=0,
           'High Vol': rvol>=np.median(rvol), 'Low Vol': rvol<np.median(rvol),
           'Earnings': earn, 'Normal': ~earn}


# ── Effect size 함수 ─────────────────────────────────────────────
def hedges_g(a, b):
    """Paired Hedges' g with 95% CI (seed-level, n=5)."""
    diff  = a - b          # (5,)
    n_s   = len(diff)
    mean_d = diff.mean()
    sd_d   = diff.std(ddof=1)
    # Hedges' correction factor J
    df  = n_s - 1
    J   = 1 - 3 / (4 * df - 1)
    g   = J * mean_d / (sd_d + 1e-12)
    # 95% CI via non-central t approximation (Hedges & Olkin 1985)
    se_g = np.sqrt(1/n_s + g**2 / (2*n_s))
    t95  = stats.t.ppf(0.975, df=df)
    ci   = t95 * se_g
    _, p = stats.ttest_rel(a, b)
    return g, ci, p


def cohens_d_daily(arr_kg, arr_base, rmask, B=2000, seed=42):
    """
    Daily-level Cohen's d + bootstrap 95% CI.
    Bootstrap over days (not seeds) — captures temporal variability.
    """
    dkg   = arr_kg[:,  rmask].mean(axis=0)   # seed-mean daily IC
    dbase = arr_base[:, rmask].mean(axis=0)
    diff  = dkg - dbase
    n_r   = len(diff)
    mean_d = diff.mean()
    sd_d   = diff.std(ddof=1)
    d      = mean_d / (sd_d + 1e-12)

    # Block bootstrap (block size ~5d to preserve autocorrelation)
    rng    = np.random.default_rng(seed)
    block  = 5
    n_blk  = int(np.ceil(n_r / block))
    boot_means = []
    for _ in range(B):
        blk_starts = rng.integers(0, n_r - block + 1, size=n_blk)
        idx = np.concatenate([np.arange(s, min(s+block, n_r)) for s in blk_starts])[:n_r]
        boot_means.append(diff[idx].mean())
    boot_means = np.array(boot_means)
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    _, p = stats.ttest_rel(dkg, dbase)
    return d, (mean_d, ci_lo*1000, ci_hi*1000), p


def fdr_bh(pvals):
    """Benjamini-Hochberg FDR correction. Returns q-values."""
    n   = len(pvals)
    idx = np.argsort(pvals)
    q   = np.empty(n)
    q[idx] = np.array(pvals)[idx] * n / (np.arange(n) + 1)
    # monotone enforcement
    q = np.minimum.accumulate(q[::-1])[::-1]
    return np.clip(q, 0, 1)


def sig_label(p, q=None):
    ref = q if q is not None else p
    if ref < 0.001: return '***'
    if ref < 0.01:  return '**'
    if ref < 0.05:  return '*'
    if ref < 0.10:  return '†'
    return 'n.s.'


# ── 전체 계산 ────────────────────────────────────────────────────
records = []
for ck in KG_CONDS:
    for rn in REGIME_ORDER:
        rmask = REGIMES[rn]
        # Seed-level Hedges' g
        seed_kg   = kg[ck][:, rmask].mean(axis=1)   # (5,)
        seed_base = base[:, rmask].mean(axis=1)
        g, g_ci, p_seed = hedges_g(seed_kg, seed_base)
        # Daily-level Cohen's d + bootstrap CI
        d, (delta_ic, ci_lo, ci_hi), p_daily = cohens_d_daily(kg[ck], base, rmask)
        records.append(dict(
            kg=ck, regime=rn,
            delta_ic=delta_ic,       # mean ΔIC × 1000
            ci_lo=ci_lo, ci_hi=ci_hi,  # bootstrap 95% CI (× 1000)
            cohens_d=d,
            hedges_g=g, g_ci=g_ci,
            p_seed=p_seed,
            p_daily=p_daily,
            n_days=rmask.sum(),
        ))

df = pd.DataFrame(records)

# FDR correction on daily p-values
df['q_daily'] = fdr_bh(df['p_daily'].values)

# sig labels
df['sig_raw'] = df['p_daily'].apply(sig_label)
df['sig_fdr'] = df.apply(lambda r: sig_label(r['p_daily'], r['q_daily']), axis=1)


# ── 출력 테이블 ──────────────────────────────────────────────────
print("=" * 90)
print("Effect Size Report  (ΔIC×1000 with 95% Bootstrap CI  |  Cohen's d  |  Hedges' g)")
print("=" * 90)
for ck, (label, _) in KG_CONDS.items():
    print(f"\n── {label} vs Baseline ──")
    print(f"  {'Regime':<12} {'n':>4} {'ΔIC×1000':>10} {'95% CI':>18} "
          f"{'Cohen d':>8} {'Hedges g':>9} {'p(daily)':>9} {'q(FDR)':>8} {'sig':>5}")
    for rn in REGIME_ORDER:
        row = df[(df.kg==ck)&(df.regime==rn)].iloc[0]
        ci_str = f"[{row.ci_lo:+.2f}, {row.ci_hi:+.2f}]"
        print(f"  {rn:<12} {int(row.n_days):>4} {row.delta_ic:>+10.3f} {ci_str:>18} "
              f"{row.cohens_d:>+8.3f} {row.hedges_g:>+9.3f} "
              f"{row.p_daily:>9.4f} {row.q_daily:>8.4f} {row.sig_fdr:>5}")

print("\n† p<0.10  * p<0.05  ** p<0.01  *** p<0.001  (FDR-adjusted q-value)")
print("Bootstrap CI: block bootstrap (block=5d), B=2000")

# CSV 저장
df.to_csv(os.path.join(OUT, 'effect_size_table.csv'), index=False)
print(f"\n[saved] {OUT}/effect_size_table.csv")


# ── Improved Forest Plot ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

n_r  = len(REGIME_ORDER)
offsets = np.linspace(-0.28, 0.28, len(KG_CONDS))

for ki, (ck, (label, color)) in enumerate(KG_CONDS.items()):
    for ri, rn in enumerate(REGIME_ORDER):
        row = df[(df.kg==ck)&(df.regime==rn)].iloc[0]
        y   = (n_r - 1 - ri) + offsets[ki]
        d   = row.delta_ic
        lo, hi = row.ci_lo, row.ci_hi
        xerr = [[d - lo], [hi - d]]

        ax.errorbar(d, y, xerr=xerr, fmt='o', color=color,
                    markersize=6, linewidth=1.8, capsize=4,
                    label=label if ri == 0 else '_')

        # FDR-adjusted significance star
        star = row.sig_fdr
        if star not in ('n.s.', ''):
            ax.text(hi + 0.3, y, star, va='center',
                    fontsize=10, color=color, fontweight='bold')

        # Hedges' g 표시 (작게)
        ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 15 else 22,
                y, f"g={row.hedges_g:+.2f}", va='center',
                fontsize=7, color=color, alpha=0.8)

ax.set_yticks(range(n_r))
ax.set_yticklabels(list(reversed(REGIME_ORDER)), fontsize=11)
ax.set_xlabel('ΔIC × 1000  with 95% Bootstrap CI  (block bootstrap, B=2000)', fontsize=10)
ax.set_title('Effect Size: KG vs Baseline (FDR-adjusted significance)', fontsize=12)
ax.legend(fontsize=10, loc='lower right')

# Low Vol 강조
lv_i = n_r - 1 - REGIME_ORDER.index('Low Vol')
ax.axhspan(lv_i - 0.45, lv_i + 0.45, alpha=0.07, color='green')

# 구분선
for i in range(n_r - 1):
    ax.axhline(i + 0.5, color='gray', linewidth=0.4, linestyle=':')

ax.annotate('g = Hedges\' g  (seed-level, n=5, small-sample corrected)\n'
            '† p<0.10   * p<0.05   ** p<0.01   (FDR q-value)',
            xy=(0.02, 0.02), xycoords='axes fraction', fontsize=8, color='gray')

plt.tight_layout()
path = os.path.join(OUT, 'fig5_effect_size_forest.png')
plt.savefig(path, dpi=200, bbox_inches='tight')
plt.close()
print(f'[saved] {path}')


# ── Hedges' g 히트맵 ─────────────────────────────────────────────
rows_h = REGIME_ORDER
cols_h = list(KG_CONDS.keys())
g_mat  = np.zeros((len(rows_h), len(cols_h)))
q_mat  = np.zeros_like(g_mat)

for i, rn in enumerate(rows_h):
    for j, ck in enumerate(cols_h):
        row = df[(df.kg==ck)&(df.regime==rn)].iloc[0]
        g_mat[i,j] = row.hedges_g
        q_mat[i,j] = row.q_daily

from matplotlib.colors import TwoSlopeNorm
fig, ax = plt.subplots(figsize=(7, 5))
norm = TwoSlopeNorm(vmin=-0.6, vcenter=0, vmax=1.0)
im = ax.imshow(g_mat, cmap='RdYlGn', norm=norm, aspect='auto')

ax.set_xticks(range(len(cols_h)))
ax.set_xticklabels([KG_CONDS[c][0] for c in cols_h], fontsize=11)
ax.set_yticks(range(len(rows_h)))
ax.set_yticklabels(rows_h, fontsize=11)

for i in range(len(rows_h)):
    for j in range(len(cols_h)):
        g = g_mat[i,j]
        star = sig_label(q_mat[i,j])
        color = 'white' if abs(g) > 0.5 else 'black'
        ax.text(j, i, f'{g:+.2f}{star}', ha='center', va='center',
                fontsize=10, fontweight='bold', color=color)

plt.colorbar(im, ax=ax, fraction=0.03).set_label("Hedges' g  (seed-level, n=5)", fontsize=9)
ax.set_title("Hedges' g Heatmap: KG vs Baseline  (FDR-adjusted)", fontsize=12, pad=10)
ax.annotate('† q<0.10   * q<0.05   (Benjamini-Hochberg FDR)',
            xy=(0.5, -0.07), xycoords='axes fraction', ha='center', fontsize=8, color='gray')

plt.tight_layout()
path = os.path.join(OUT, 'fig6_hedges_g_heatmap.png')
plt.savefig(path, dpi=200, bbox_inches='tight')
plt.close()
print(f'[saved] {path}')
print('\nDone.')
