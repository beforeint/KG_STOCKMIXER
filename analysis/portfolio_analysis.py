#!/usr/bin/env python3
"""
Portfolio Risk Metrics + KG Structure Analysis

Part 1 — Portfolio metrics from daily IC (no re-run needed):
  Sharpe ratio, Maximum Drawdown, VaR (5%), CVaR (5%), Win rate

Part 2 — KG degree stratified group IC comparison:
  High-degree (top quartile) vs Low-degree (bottom quartile) stocks
  Compute group IC per day using actual returns (gt_data)

Part 3 — KG structure validation:
  KG-adjacent pairs have higher return correlation than random pairs?
  → explains WHY KG helps
"""
import os, pickle
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, 'results')
OUT     = os.path.join(BASE, 'regime_plots')
DATA    = '/gpfs/home1/pz29075/Capstone/StockMixer/dataset/NASDAQ'
DATES_CSV = ('/gpfs/home1/pz29075/Capstone/StockMixer/'
             'Temporal_Relational_Stock_Ranking/data/NASDAQ_aver_line_dates.csv')
KG_PATH = os.path.join(DATA, 'relation/wikidata/NASDAQ_wiki_relation.npy')
TICKER_FILE = ('/gpfs/home1/pz29075/Capstone/StockMixer/'
               'Temporal_Relational_Stock_Ranking/data/'
               'NASDAQ_tickers_qualify_dr-0.98_min-5_smooth.csv')
TEST_IDX = 1008

CONDITIONS = {
    'none_wikidata':      ('Baseline',   '#9E9E9E'),
    'gat_wikidata':       ('GAT+Wiki',   '#2196F3'),
    'gat_institutional':  ('GAT+Inst',   '#FF9800'),
    'gat_supply_chain':   ('GAT+Supply', '#4CAF50'),
}
TRADING_DAYS = 252


# ── 데이터 로드 ──────────────────────────────────────────────────
def load(key):
    return np.stack([np.load(os.path.join(RESULTS, f'{key}_seed{s}_daily_ic.npy'))
                     for s in range(5)])   # (5, 237)

daily_ic = {k: load(k) for k in CONDITIONS}
n = next(iter(daily_ic.values())).shape[1]   # 237

with open(os.path.join(DATA, 'gt_data.pkl'),   'rb') as f: gt   = pickle.load(f, encoding='latin1')
with open(os.path.join(DATA, 'mask_data.pkl'), 'rb') as f: mask = pickle.load(f, encoding='latin1')

gt_test   = gt[:,   TEST_IDX:TEST_IDX+n]     # (1026, 237) actual returns
mask_test = mask[:, TEST_IDX:TEST_IDX+n]     # (1026, 237)

tickers = open(TICKER_FILE).read().strip().split('\n')

kg_raw = np.load(KG_PATH)                     # (1026, 1026, 43)
adj_raw = (kg_raw > 0).any(axis=2).astype(float)  # (1026, 1026)
# self-loop 제거 (wikidata는 전 종목에 self-loop 존재)
adj = adj_raw.copy()
np.fill_diagonal(adj, 0)
degree = adj.sum(axis=1)                     # (1026,) out-degree (self-loop 제외)


# ══════════════════════════════════════════════════════════════════
# Part 1: Portfolio Risk Metrics
# ══════════════════════════════════════════════════════════════════
def portfolio_metrics(ic_arr):
    """ic_arr: (5, 237) → dict of metrics (mean over seeds)."""
    results = []
    for s in range(ic_arr.shape[0]):
        ic = ic_arr[s]
        mu     = ic.mean()
        sigma  = ic.std(ddof=1)
        sharpe = mu / (sigma + 1e-12) * np.sqrt(TRADING_DAYS)

        cum = np.cumsum(ic)
        running_max = np.maximum.accumulate(cum)
        drawdown    = running_max - cum
        mdd         = drawdown.max()

        var_5  = np.percentile(ic, 5)
        cvar_5 = ic[ic <= var_5].mean() if (ic <= var_5).any() else var_5
        winrate = (ic > 0).mean()

        results.append(dict(sharpe=sharpe, mdd=mdd, var_5=var_5*1000,
                            cvar_5=cvar_5*1000, winrate=winrate,
                            mean_ic=mu*1000, std_ic=sigma*1000))

    df = pd.DataFrame(results)
    return df.mean().to_dict(), df.std().to_dict()

print("=" * 80)
print("Portfolio Risk Metrics  (mean ± std across 5 seeds)")
print("=" * 80)
print(f"{'Condition':<18} {'Sharpe':>8} {'MDD(×1k)':>10} {'VaR5%(×1k)':>12} "
      f"{'CVaR5%(×1k)':>13} {'WinRate':>9}")
metrics_all = {}
for ck, (label, color) in CONDITIONS.items():
    m, s = portfolio_metrics(daily_ic[ck])
    metrics_all[ck] = m
    print(f"{label:<18} {m['sharpe']:>+8.3f}±{s['sharpe']:.2f}"
          f"  {m['mdd']:>8.3f}±{s['mdd']:.3f}"
          f"  {m['var_5']:>10.3f}±{s['var_5']:.3f}"
          f"  {m['cvar_5']:>11.3f}±{s['cvar_5']:.3f}"
          f"  {m['winrate']:>8.1%}±{s['winrate']:.1%}")


# ══════════════════════════════════════════════════════════════════
# Part 2: KG Degree Stratified Group IC
# ══════════════════════════════════════════════════════════════════
# Wikidata는 extremely sparse: 87%가 degree=1, max=55
# Hub: degree ≥ 10 (top ~7%, 진짜 연결된 종목)
# Low: degree = 1  (87%, 최소 연결)
# self-loop 제외 후: 893개 고립(degree=0), 나머지 133개만 연결
HUB_THR = 10
hub_mask  = degree >= HUB_THR   # ~73 stocks
mid_mask  = (degree > 0) & (degree < HUB_THR)   # ~60 stocks
isol_mask = degree == 0          # ~893 stocks (truly isolated)

print(f"\n── KG Degree Groups (self-loop 제외) ──")
print(f"  Hub  (degree ≥ {HUB_THR}):  {hub_mask.sum():3d} stocks, "
      f"mean degree={degree[hub_mask].mean():.1f}")
print(f"  Mid  (0 < d < {HUB_THR}): {mid_mask.sum():3d} stocks, "
      f"mean degree={degree[mid_mask].mean():.1f}")
print(f"  Isolated (degree=0):    {isol_mask.sum():3d} stocks ({isol_mask.mean()*100:.0f}%)")

# 상위 10 hub stocks
top_idx = np.argsort(degree)[::-1][:10]
print(f"\n  Top-10 Hub Stocks (wikidata degree):")
for i in top_idx:
    print(f"    {tickers[i]:<8}  degree={int(degree[i])}")


def group_ic_daily(gt_arr, mask_arr, group_mask):
    """
    그룹 내 종목만으로 cross-sectional rank correlation 계산.
    gt_arr: (S, T), mask_arr: (S, T), group_mask: (S,) bool
    Returns: (T,) daily group IC
    """
    from scipy.stats import spearmanr
    ics = []
    S, T = gt_arr.shape
    gidx = np.where(group_mask)[0]
    for t in range(T):
        valid = mask_arr[gidx, t] > 0
        if valid.sum() < 5:
            ics.append(np.nan)
            continue
        ret_t = gt_arr[gidx[valid], t]
        rank  = np.argsort(np.argsort(ret_t)).astype(float)
        # 예측값 없이 degree로 그룹 IC 계산 불가 → 실제 returns의 분산만 보고
        # → 대신 "그룹 내 종목들의 평균 IC contribution 추정"은 pred 없이 불가
        # 여기선 그룹별 return std(= 예측 난이도 proxy) 계산
        ics.append(ret_t.std())
    return np.array(ics)


# per-stock degree vs return volatility (예측 난이도 indicator)
# Per-group return volatility and avg KG neighbor correlation
hub_vol, isol_vol = [], []
for t in range(n):
    vm  = mask_test[:, t] > 0
    ret = gt_test[:, t]
    hub_vol.append(ret[hub_mask & vm].std()  if (hub_mask & vm).sum()  > 2 else np.nan)
    isol_vol.append(ret[isol_mask & vm].std() if (isol_mask & vm).sum() > 2 else np.nan)

hub_vol  = np.nanmean(hub_vol)
isol_vol = np.nanmean(isol_vol)

# Hub 종목 vs Low 종목의 KG 이웃 간 return 상관관계 비교
corr_mat_full = np.corrcoef(gt_test)  # (1026, 1026)

hub_idx  = np.where(hub_mask)[0]

def avg_neighbor_corr(idx_list, adj_mat, corr_mat):
    """각 종목의 KG 이웃(self 제외)과의 평균 correlation."""
    vals = []
    for i in idx_list:
        nbrs = np.where(adj_mat[i] > 0)[0]
        nbrs = nbrs[nbrs != i]   # exclude self-loop
        if len(nbrs) == 0:
            continue
        c = corr_mat[i, nbrs]
        vals.extend(c[np.isfinite(c)].tolist())
    return np.array(vals)

hub_nbr_corr = avg_neighbor_corr(hub_idx, adj, corr_mat_full)
mid_idx      = np.where(mid_mask)[0]
mid_nbr_corr = avg_neighbor_corr(mid_idx, adj, corr_mat_full)

print(f"\n  Daily return volatility:")
print(f"    Hub stocks: {hub_vol*100:.3f}%  |  Isolated stocks: {isol_vol*100:.3f}%")
print(f"\n  KG neighbor return correlation (self-loop 제외):")
print(f"    Hub stocks' neighbors: mean={hub_nbr_corr.mean():.4f} (n={len(hub_nbr_corr)})")
if len(mid_nbr_corr) > 0:
    print(f"    Mid stocks' neighbors: mean={mid_nbr_corr.mean():.4f} (n={len(mid_nbr_corr)})")


# ══════════════════════════════════════════════════════════════════
# Part 3: KG Structure Validation
# KG 인접 종목 쌍 vs 무작위 쌍의 return correlation 비교
# ══════════════════════════════════════════════════════════════════
print("\n── KG Structure Validation ──")
# 테스트 기간 return 행렬
valid_stocks = mask_test.mean(axis=1) > 0.8   # 80% 이상 거래일에 유효
ret_mat = gt_test[valid_stocks, :]             # (S', 237)
adj_sub = adj[valid_stocks, :][:, valid_stocks]
S2      = ret_mat.shape[0]
print(f"  Valid stocks: {S2}/{len(valid_stocks)}")

# 상관관계 행렬
corr_mat = np.corrcoef(ret_mat)                # (S', S')

# KG 인접 쌍 vs 비인접 쌍
kg_pairs   = np.where(np.triu(adj_sub, k=1) > 0)
rand_pairs = np.where(np.triu(1 - adj_sub, k=1) > 0)

kg_corrs   = corr_mat[kg_pairs]
rand_corrs = corr_mat[rand_pairs]

# NaN 제거
kg_corrs   = kg_corrs[np.isfinite(kg_corrs)]
rand_corrs = rand_corrs[np.isfinite(rand_corrs)]

# 랜덤 샘플 (같은 크기로)
rng = np.random.default_rng(42)
rand_sample = rng.choice(rand_corrs, size=min(len(kg_corrs)*10, len(rand_corrs)), replace=False)

t_stat, p_val = stats.ttest_ind(kg_corrs, rand_sample, equal_var=False)
print(f"  KG-adjacent pairs   (n={len(kg_corrs):,}):  mean corr = {kg_corrs.mean():.4f}")
print(f"  Random pairs sample (n={len(rand_sample):,}): mean corr = {rand_sample.mean():.4f}")
print(f"  Difference: {kg_corrs.mean() - rand_sample.mean():+.4f}  "
      f"(t={t_stat:.2f}, p={p_val:.4f})")


# ══════════════════════════════════════════════════════════════════
# Figure: 4-panel summary
# ══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

colors = [v[1] for v in CONDITIONS.values()]
labels = [v[0] for v in CONDITIONS.values()]

# ── Panel 1: Cumulative IC curves ──
ax1 = fig.add_subplot(gs[0, :2])
for ck, (label, color) in CONDITIONS.items():
    mean_ic = daily_ic[ck].mean(axis=0)
    cum_ic  = np.cumsum(mean_ic)
    ax1.plot(cum_ic, label=label, color=color, linewidth=1.8)
ax1.axhline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.4)
ax1.set_title('Cumulative IC (seed-mean)', fontsize=12)
ax1.set_xlabel('Trading Days (Test Period)', fontsize=10)
ax1.set_ylabel('Cumulative IC', fontsize=10)
ax1.legend(fontsize=10)

# ── Panel 2: Max Drawdown bar ──
ax2 = fig.add_subplot(gs[0, 2])
mdds = [metrics_all[ck]['mdd'] * 1000 for ck in CONDITIONS]
bars = ax2.bar(labels, mdds, color=colors, alpha=0.8, edgecolor='white')
ax2.set_title('Max Drawdown of Cum-IC (×1000)', fontsize=11)
ax2.set_ylabel('MDD × 1000', fontsize=10)
ax2.tick_params(axis='x', rotation=15)
for bar, val in zip(bars, mdds):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.2f}', ha='center', fontsize=9)

# ── Panel 3: VaR / CVaR ──
ax3 = fig.add_subplot(gs[1, 0])
x = np.arange(len(CONDITIONS))
w = 0.35
vars_  = [metrics_all[ck]['var_5']  for ck in CONDITIONS]
cvars_ = [metrics_all[ck]['cvar_5'] for ck in CONDITIONS]
ax3.bar(x - w/2, vars_,  w, label='VaR (5%)',  color='#EF5350', alpha=0.8)
ax3.bar(x + w/2, cvars_, w, label='CVaR (5%)', color='#B71C1C', alpha=0.8)
ax3.set_xticks(x); ax3.set_xticklabels(labels, rotation=15, fontsize=9)
ax3.set_title('Tail Risk Metrics (×1000)', fontsize=11)
ax3.set_ylabel('IC × 1000', fontsize=10)
ax3.axhline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.4)
ax3.legend(fontsize=9)

# ── Panel 4: KG degree distribution ──
ax4 = fig.add_subplot(gs[1, 1])
deg_nonzero = degree[degree > 0]
deg_nonzero = degree[degree > 1]  # degree=1이 887개라 log scale 사용
ax4.hist(degree, bins=range(1, int(degree.max())+2), color='#2196F3', alpha=0.7,
         edgecolor='white', log=True)
ax4.axvline(HUB_THR, color='red', linestyle='--', linewidth=1.5,
            label=f'Hub threshold (≥{HUB_THR})')
ax4.set_title('Wikidata KG Degree Distribution\n(log scale)', fontsize=11)
ax4.set_xlabel('KG Out-degree', fontsize=10)
ax4.set_ylabel('Number of Stocks (log)', fontsize=10)
ax4.legend(fontsize=9)
ax4.text(0.98, 0.95, f'87% have degree=1', transform=ax4.transAxes,
         ha='right', va='top', fontsize=9, color='gray')

# ── Panel 5: KG validation (return correlation) ──
ax5 = fig.add_subplot(gs[1, 2])
bp = ax5.boxplot([kg_corrs, rand_sample],
                 labels=['KG-adjacent\npairs', 'Random\npairs'],
                 patch_artist=True, notch=True,
                 medianprops=dict(color='black', linewidth=2))
bp['boxes'][0].set_facecolor('#2196F3'); bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor('#9E9E9E'); bp['boxes'][1].set_alpha(0.5)
ax5.set_title(f'Return Correlation:\nKG-adjacent vs Random pairs\n(p={p_val:.4f})', fontsize=10)
ax5.set_ylabel('Pearson Correlation (test period)', fontsize=10)
ax5.text(0.5, 0.02, f'Δ = {kg_corrs.mean()-rand_sample.mean():+.4f}',
         transform=ax5.transAxes, ha='center', fontsize=10, fontweight='bold',
         color='#2196F3' if kg_corrs.mean() > rand_sample.mean() else 'red')

plt.suptitle('Portfolio Risk & KG Structure Analysis', fontsize=14, fontweight='bold', y=1.01)
path = os.path.join(OUT, 'fig7_portfolio_kg_analysis.png')
plt.savefig(path, dpi=200, bbox_inches='tight')
plt.close()
print(f'\n[saved] {path}')

# ── Sharpe table ──
print("\n── Annualized Sharpe Ratio Summary ──")
for ck, (label, _) in CONDITIONS.items():
    m, s = portfolio_metrics(daily_ic[ck])
    base = metrics_all['none_wikidata']['sharpe']
    ratio = m['sharpe'] / base if base != 0 else float('nan')
    print(f"  {label:<18} Sharpe={m['sharpe']:+.3f}±{s['sharpe']:.3f}"
          f"  ({ratio:.1f}x baseline)")
print('\nDone.')
