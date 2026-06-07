#!/usr/bin/env python3
"""
Statistical Significance Testing: KG vs Baseline
두 가지 레벨에서 검정
  1. Seed-level  : 5개 seed의 평균 IC → paired t-test / Wilcoxon (n=5)
  2. Daily-level : 237일 × seed평균 IC → paired t-test (n=237, 시계열 상관 주의)
  3. Regime별    : Low Vol / Bull / Earnings 각 구간에서 daily-level 검정

결과: regime_plots/significance_report.txt + significance_heatmap.png
"""
import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, 'results')
OUT     = os.path.join(BASE, 'regime_plots')
os.makedirs(OUT, exist_ok=True)

DATA      = '/gpfs/home1/pz29075/Capstone/StockMixer/dataset/NASDAQ'
DATES_CSV = ('/gpfs/home1/pz29075/Capstone/StockMixer/'
             'Temporal_Relational_Stock_Ranking/data/NASDAQ_aver_line_dates.csv')
TEST_IDX  = 1008
EARNINGS  = [('2017-01-09','2017-02-10'), ('2017-04-10','2017-05-12'),
             ('2017-07-10','2017-08-11'), ('2017-10-09','2017-10-27')]

CONDITIONS = {
    'none_wikidata':     'Baseline',
    'gat_wikidata':      'GAT+Wiki',
    'gat_institutional': 'GAT+Inst',
    'gat_supply_chain':  'GAT+Supply',
    'gat_news_dynamic':  'GAT+NewsDyn',
}

# ── 데이터 로드 ──────────────────────────────────────────────────
def load_arrays(cond_key, seeds=range(5)):
    return np.stack([np.load(os.path.join(RESULTS, f'{cond_key}_seed{s}_daily_ic.npy'))
                     for s in seeds], axis=0)   # (5, 237)

arrays = {k: load_arrays(k) for k in CONDITIONS}
n_test = arrays['none_wikidata'].shape[1]

# regime 레이블
with open(os.path.join(DATA, 'gt_data.pkl'), 'rb') as f:
    gt = pickle.load(f, encoding='latin1')
with open(os.path.join(DATA, 'mask_data.pkl'), 'rb') as f:
    mask = pickle.load(f, encoding='latin1')

mkt_ret  = np.nanmean(np.where(mask[:, TEST_IDX:TEST_IDX+n_test] > 0,
                                gt[:, TEST_IDX:TEST_IDX+n_test], np.nan), axis=0)
roll_vol = np.array([mkt_ret[max(0,i-19):i+1].std() for i in range(n_test)])

dates_raw  = pd.read_csv(DATES_CSV, header=None)[0]
test_dates = pd.to_datetime(dates_raw.iloc[TEST_IDX:TEST_IDX+n_test].values)
earnings   = np.zeros(n_test, dtype=bool)
for s, e in EARNINGS:
    earnings |= (test_dates >= pd.Timestamp(s)) & (test_dates <= pd.Timestamp(e))

REGIMES = {
    'All':      np.ones(n_test, dtype=bool),
    'Bull':     mkt_ret > 0,
    'Bear':     mkt_ret <= 0,
    'High Vol': roll_vol >= np.median(roll_vol),
    'Low Vol':  roll_vol < np.median(roll_vol),
    'Earnings': earnings,
    'Normal':   ~earnings,
}


# ── 검정 함수 ────────────────────────────────────────────────────
def run_tests(a, b, name_a='KG', name_b='Baseline'):
    """
    a, b: (5, n_days) arrays
    Returns dict with seed-level and daily-level test results.
    """
    # Seed-level: mean IC per seed (n=5)
    seed_a = a.mean(axis=1)   # (5,)
    seed_b = b.mean(axis=1)
    diff_seed = seed_a - seed_b

    t_stat, p_t_seed = stats.ttest_rel(seed_a, seed_b)
    w_stat, p_w_seed = stats.wilcoxon(seed_a, seed_b, alternative='greater') \
        if len(set(diff_seed)) > 1 else (np.nan, np.nan)

    # Daily-level: seed-mean daily IC (n=237)
    daily_a = a.mean(axis=0)  # (237,)
    daily_b = b.mean(axis=0)
    t_stat_d, p_t_daily = stats.ttest_rel(daily_a, daily_b)

    return {
        'mean_a':     seed_a.mean(),
        'mean_b':     seed_b.mean(),
        'diff':       diff_seed.mean(),
        'p_t_seed':   p_t_seed,       # paired t-test, n=5
        'p_w_seed':   p_w_seed,       # Wilcoxon signed-rank, n=5
        'p_t_daily':  p_t_daily,      # paired t-test on daily IC, n=237
        'daily_a':    daily_a,
        'daily_b':    daily_b,
    }


def sig_star(p):
    if np.isnan(p):   return 'n/a'
    if p < 0.001:     return '***'
    if p < 0.01:      return '**'
    if p < 0.05:      return '*'
    if p < 0.10:      return '†'
    return 'n.s.'


# ── 메인 분석 ────────────────────────────────────────────────────
baseline = arrays['none_wikidata']
kg_conds = {k: v for k, v in arrays.items() if k != 'none_wikidata'}

lines = []
lines.append("=" * 72)
lines.append("Statistical Significance: KG vs Baseline (none)")
lines.append("Test period: 2016-11-18 ~ 2017-10-27  (237 days, 5 seeds)")
lines.append("=" * 72)

# p-value 매트릭스 (regime × condition) — daily-level t-test
p_matrix  = pd.DataFrame(index=list(REGIMES.keys()),
                          columns=[CONDITIONS[k] for k in kg_conds])
ic_matrix = pd.DataFrame(index=list(REGIMES.keys()),
                          columns=['Baseline'] + [CONDITIONS[k] for k in kg_conds])

for regime_name, rmask in REGIMES.items():
    n_days = rmask.sum()
    lines.append(f"\n── {regime_name} ({n_days} days) ──")
    lines.append(f"  {'Condition':<20} {'Base IC':>8} {'KG IC':>8} {'Δ IC':>8} "
                 f"{'p(seed,t)':>10} {'p(seed,W)':>10} {'p(daily,t)':>11} {'sig':>5}")

    base_daily_r = baseline[:, rmask].mean(axis=0)
    ic_matrix.loc[regime_name, 'Baseline'] = baseline[:, rmask].mean()

    for ck, arr in kg_conds.items():
        label = CONDITIONS[ck]
        kg_daily_r = arr[:, rmask].mean(axis=0)

        seed_base = baseline[:, rmask].mean(axis=1)
        seed_kg   = arr[:, rmask].mean(axis=1)
        diff_s    = seed_kg - seed_base

        t_s, p_ts = stats.ttest_rel(seed_kg, seed_base)
        try:
            _, p_ws = stats.wilcoxon(seed_kg, seed_base, alternative='greater')
        except Exception:
            p_ws = np.nan
        t_d, p_td = stats.ttest_rel(kg_daily_r, base_daily_r)

        star = sig_star(p_td)
        lines.append(f"  {label:<20} {seed_base.mean():>8.4f} {seed_kg.mean():>8.4f} "
                     f"{diff_s.mean():>+8.4f} {p_ts:>10.4f} {p_ws:>10.4f} "
                     f"{p_td:>11.4f} {star:>5}")

        p_matrix.loc[regime_name, label]  = p_td
        ic_matrix.loc[regime_name, label] = seed_kg.mean()

lines.append("\n* p<0.05  ** p<0.01  *** p<0.001  † p<0.10  n.s. not significant")
lines.append("seed-level tests: n=5 (low power).  daily-level: n=days in regime.")
lines.append("Note: daily IC values are temporally correlated → daily p-values")
lines.append("      are anti-conservative. Use as directional evidence only.")

report = "\n".join(lines)
print(report)

report_path = os.path.join(OUT, 'significance_report.txt')
with open(report_path, 'w') as f:
    f.write(report)
print(f"\n[saved] {report_path}")


# ── 히트맵 시각화 ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Statistical Significance: KG vs Baseline\n(daily-level paired t-test p-value)',
             fontsize=13, fontweight='bold')

# p-value 히트맵
p_vals = p_matrix.astype(float)
im0 = axes[0].imshow(p_vals.values, cmap='RdYlGn_r', vmin=0, vmax=0.2, aspect='auto')
axes[0].set_xticks(range(len(p_vals.columns)))
axes[0].set_xticklabels(p_vals.columns, rotation=20, ha='right')
axes[0].set_yticks(range(len(p_vals.index)))
axes[0].set_yticklabels(p_vals.index)
axes[0].set_title('p-value (green=significant)')
for i in range(len(p_vals.index)):
    for j in range(len(p_vals.columns)):
        v = p_vals.values[i, j]
        star = sig_star(v)
        axes[0].text(j, i, f'{v:.3f}\n{star}', ha='center', va='center',
                     fontsize=8, color='black')
plt.colorbar(im0, ax=axes[0])

# IC 비교 히트맵
ic_vals = ic_matrix.astype(float)
im1 = axes[1].imshow(ic_vals.values, cmap='Blues', aspect='auto')
axes[1].set_xticks(range(len(ic_vals.columns)))
axes[1].set_xticklabels(ic_vals.columns, rotation=20, ha='right')
axes[1].set_yticks(range(len(ic_vals.index)))
axes[1].set_yticklabels(ic_vals.index)
axes[1].set_title('Mean IC by Regime')
for i in range(len(ic_vals.index)):
    for j in range(len(ic_vals.columns)):
        v = ic_vals.values[i, j]
        axes[1].text(j, i, f'{v:.4f}', ha='center', va='center', fontsize=8)
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
path = os.path.join(OUT, 'significance_heatmap.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f"[saved] {path}")
