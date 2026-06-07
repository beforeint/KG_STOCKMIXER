#!/usr/bin/env python3
"""
Regime Analysis: Conditional/분리 성능 비교
- Bull/Bear market
- High/Low volatility
- Earnings season vs normal

Results dir: KG_StockMixer/results/{run_name}_daily_ic.npy
Test period:  2016-11-18 ~ 2017-10-27  (idx 1008~1244, 237일)
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 경로 설정 ──────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = '/gpfs/home1/pz29075/Capstone/StockMixer/dataset/NASDAQ'
DATES_CSV = ('/gpfs/home1/pz29075/Capstone/StockMixer/'
             'Temporal_Relational_Stock_Ranking/data/NASDAQ_aver_line_dates.csv')
RESULTS = os.path.join(BASE, 'results')
OUT     = os.path.join(BASE, 'regime_plots')
os.makedirs(OUT, exist_ok=True)

TEST_START = 1008   # config.py test_index
TEST_DAYS  = 237    # 1245 - 1008

# ── 분석 조건 정의 ─────────────────────────────────────────────────────
# key = run_name prefix (= f"{graph_type}_{kg_source}")
CONDITIONS = {
    'none_wikidata':         ('Baseline (No KG)',       '#888888'),
    'gat_wikidata':          ('GAT + Wikidata',         '#2196F3'),
    'gat_institutional':     ('GAT + Institutional',    '#FF9800'),
    'gat_supply_chain':      ('GAT + Supply Chain',     '#4CAF50'),
    'gat_news_dynamic':      ('GAT + News Dynamic',     '#9C27B0'),
}

# ── 날짜 및 gt 로드 ────────────────────────────────────────────────────
dates_raw = pd.read_csv(DATES_CSV, header=None)[0]
test_dates = pd.to_datetime(dates_raw.iloc[TEST_START:TEST_START + TEST_DAYS].values)

with open(os.path.join(DATA, 'gt_data.pkl'), 'rb') as f:
    gt = pickle.load(f, encoding='latin1')
with open(os.path.join(DATA, 'mask_data.pkl'), 'rb') as f:
    mask = pickle.load(f, encoding='latin1')

test_gt   = gt[:,   TEST_START: TEST_START + TEST_DAYS]   # (1026, 237)
test_mask = mask[:, TEST_START: TEST_START + TEST_DAYS]

# 일별 시장 평균 수익률
mkt_ret = np.nanmean(np.where(test_mask > 0, test_gt, np.nan), axis=0)  # (237,)


# ── Regime 레이블 생성 ─────────────────────────────────────────────────
def make_regimes(mkt_ret, test_dates):
    n = len(mkt_ret)

    # 1. Bull / Bear
    bull = mkt_ret > 0   # True=Bull

    # 2. High / Low Volatility (rolling 20일 std, median split)
    roll_vol = np.array([mkt_ret[max(0, i-19):i+1].std() for i in range(n)])
    high_vol = roll_vol >= np.median(roll_vol)

    # 3. Earnings Season (S&P 기준 분기별 실적발표 시즌 ~6주)
    #    Test period: 2016-11-18 ~ 2017-10-27
    earnings_windows = [
        ('2017-01-09', '2017-02-10'),   # Q4 2016 실적
        ('2017-04-10', '2017-05-12'),   # Q1 2017 실적
        ('2017-07-10', '2017-08-11'),   # Q2 2017 실적
        ('2017-10-09', '2017-10-27'),   # Q3 2017 실적 (test 종료일까지)
    ]
    earnings = np.zeros(n, dtype=bool)
    for s, e in earnings_windows:
        earnings |= (test_dates >= pd.Timestamp(s)) & (test_dates <= pd.Timestamp(e))

    return {
        'Bull':     bull,
        'Bear':     ~bull,
        'High Vol': high_vol,
        'Low Vol':  ~high_vol,
        'Earnings': earnings,
        'Normal':   ~earnings,
        'All':      np.ones(n, dtype=bool),
    }


regimes = make_regimes(mkt_ret, test_dates)
print("Regime sizes:")
for k, v in regimes.items():
    print(f"  {k:12s}: {v.sum()} days")


# ── per-day IC 로드 ────────────────────────────────────────────────────
def load_daily_ic(cond_key, seeds=range(5)):
    """seeds 별 daily_ic 배열 로드, 평균 반환."""
    arrays = []
    for s in seeds:
        fname = os.path.join(RESULTS, f'{cond_key}_seed{s}_daily_ic.npy')
        if os.path.exists(fname):
            arrays.append(np.load(fname))
        else:
            print(f"  [missing] {fname}")
    if not arrays:
        return None
    return np.stack(arrays, axis=0)   # (seeds, days)


cond_data = {}
for key in CONDITIONS:
    arr = load_daily_ic(key)
    if arr is not None:
        cond_data[key] = arr
        print(f"Loaded {key}: shape={arr.shape}")


# ── Regime별 IC 비교 ───────────────────────────────────────────────────
def regime_ic_table(cond_data, regimes):
    rows = []
    regime_names = ['All', 'Bull', 'Bear', 'High Vol', 'Low Vol', 'Earnings', 'Normal']
    for cond, arr in cond_data.items():
        label, _ = CONDITIONS[cond]
        row = {'Condition': label}
        for rname in regime_names:
            mask_r = regimes[rname]
            ic_regime = arr[:, mask_r].mean(axis=1)   # (seeds,) mean IC in this regime
            row[rname] = f"{ic_regime.mean():.4f}±{ic_regime.std():.4f}"
        rows.append(row)
    return pd.DataFrame(rows).set_index('Condition')


if cond_data:
    df = regime_ic_table(cond_data, regimes)
    print("\n=== Regime IC Table (mean±std across seeds) ===")
    print(df.to_string())
    df.to_csv(os.path.join(OUT, 'regime_ic_table.csv'))


# ── 시각화 ─────────────────────────────────────────────────────────────
def plot_regime_comparison(cond_data, regimes):
    regime_pairs = [
        ('Bull', 'Bear'),
        ('High Vol', 'Low Vol'),
        ('Earnings', 'Normal'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Regime-Conditional IC: KG vs Baseline', fontsize=14, fontweight='bold')

    cond_keys = list(cond_data.keys())
    x = np.arange(len(cond_keys))
    width = 0.35

    for ax, (r1, r2) in zip(axes, regime_pairs):
        means_r1, stds_r1 = [], []
        means_r2, stds_r2 = [], []

        for ck in cond_keys:
            arr = cond_data[ck]
            ic1 = arr[:, regimes[r1]].mean(axis=1)
            ic2 = arr[:, regimes[r2]].mean(axis=1)
            means_r1.append(ic1.mean()); stds_r1.append(ic1.std())
            means_r2.append(ic2.mean()); stds_r2.append(ic2.std())

        colors = [CONDITIONS[ck][1] for ck in cond_keys]
        bars1 = ax.bar(x - width/2, means_r1, width, yerr=stds_r1,
                       color=colors, alpha=0.9, capsize=4, label=r1)
        bars2 = ax.bar(x + width/2, means_r2, width, yerr=stds_r2,
                       color=colors, alpha=0.45, capsize=4, label=r2,
                       hatch='//')

        ax.set_title(f'{r1} vs {r2}', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([CONDITIONS[ck][0].replace(' ', '\n') for ck in cond_keys],
                           fontsize=8)
        ax.set_ylabel('Mean IC')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUT, 'regime_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[saved] {path}")


def plot_rolling_ic(cond_data, mkt_ret, test_dates, regimes):
    """20일 rolling IC + 시장 수익률 + earnings season 음영."""
    import matplotlib.dates as mdates

    td = test_dates.to_numpy().astype('datetime64[D]')  # numpy datetime64 1D

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('Rolling 20-day IC over Test Period (2016-11 ~ 2017-10)',
                 fontsize=13, fontweight='bold')

    # earnings season 음영 (두 축 공통)
    earnings_windows = [
        ('2017-01-09', '2017-02-10'),
        ('2017-04-10', '2017-05-12'),
        ('2017-07-10', '2017-08-11'),
        ('2017-10-09', '2017-10-27'),
    ]
    for s, e in earnings_windows:
        for ax in (ax1, ax2):
            ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                       alpha=0.12, color='gold', label='_nolegend_')

    # rolling IC 곡선
    for ck, (label, color) in CONDITIONS.items():
        if ck not in cond_data:
            continue
        arr = cond_data[ck]
        mean_daily = arr.mean(axis=0)
        rolling = pd.Series(mean_daily).rolling(20, min_periods=5).mean().values
        ax1.plot(td, rolling, label=label, color=color, linewidth=1.8)

    ax1.axhline(0, color='black', linewidth=0.6, linestyle='--')
    ax1.set_ylabel('Rolling IC (20d)')
    ax1.legend(fontsize=9, loc='upper left')
    # earnings legend patch
    ax1.legend(handles=[
        *ax1.get_legend_handles_labels()[0],
        mpatches.Patch(color='gold', alpha=0.4, label='Earnings Season'),
    ], fontsize=9, loc='upper left')

    # 시장 수익률 (fill_between)
    ax2.fill_between(td, np.where(mkt_ret > 0, mkt_ret, 0), 0,
                     color='#4CAF50', alpha=0.7, label='Bull')
    ax2.fill_between(td, np.where(mkt_ret < 0, mkt_ret, 0), 0,
                     color='#F44336', alpha=0.7, label='Bear')
    ax2.axhline(0, color='black', linewidth=0.4)
    ax2.set_ylabel('Mkt Return')
    ax2.set_xlabel('Date')
    ax2.legend(fontsize=8, loc='upper left')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()

    plt.tight_layout()
    path = os.path.join(OUT, 'rolling_ic.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[saved] {path}")


if cond_data:
    plot_regime_comparison(cond_data, regimes)
    plot_rolling_ic(cond_data, mkt_ret, test_dates, regimes)
else:
    print("\n[INFO] results/ 에 daily_ic 파일 없음 — submit_regime.sh 실행 후 재시도")
    print("       예상 파일: results/none_wikidata_seed0_daily_ic.npy 등")
