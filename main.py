#%%
import argparse
import random
import sys
import os
import numpy as np
import torch
import wandb
import os, sys
os.chdir('/gpfs/home1/pz29075/Capstone/KG_StockMixer')
sys.path.insert(0, '/gpfs/home1/pz29075/Capstone/KG_StockMixer')

#%%
"""argparse"""

def get_args(debug=False):
    parser = argparse.ArgumentParser(description='KG-StockMixer ablation runner')

    # ── 데이터 ──────────────────────────────────────────────
    parser.add_argument('--market', type=str, default='NASDAQ',
                        choices=['NASDAQ', 'NYSE', 'SP500'])

    # ── 비교 축 1: 그래프 방식 ───────────────────────────────
    parser.add_argument('--graph_type', type=str, default='gcn',
                        choices=['none', 'gcn', 'gat', 'hgat'],
                        help='none=NoGraphMixer, gcn=GCN, gat=GAT, hgat=Heterogeneous GAT')

    # ── 비교 축 2: KG 소스 ───────────────────────────────────
    parser.add_argument('--kg_source', type=str, default='wikidata',
                        choices=['wikidata', 'sector_industry', 'institutional', 'board',
                                 'supply_chain', 'llm', 'llm_v2', 'news_dynamic', 'llm_dynamic'],
                        help='graph_type=none 이면 무시됨. news_dynamic/llm_dynamic=연도별 동적 KG')

    # ── 모델 하이퍼파라미터 ──────────────────────────────────
    parser.add_argument('--lr',         type=float, default=0.001)
    parser.add_argument('--no_wandb',   action='store_true',
                        help='wandb 로깅 비활성화 (빠른 디버깅용)')
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--alpha',      type=float, default=0.1,
                        help='ranking loss 가중치')
    parser.add_argument('--gat_heads',  type=int,   default=4,
                        help='GAT multi-head 수 (graph_type=gat 일 때만 사용)')

    # ── 재현성 ──────────────────────────────────────────────
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--wandb_project', type=str, default='KG_StockMixer_2',
                        help='wandb 프로젝트 이름')

    if debug:
        return parser.parse_args(args=[])
    return parser.parse_args()


#%%
from modules.utility import set_seed


#%%
"""main"""

if __name__ == '__main__':
    import importlib

    args = get_args(debug=False)
    set_seed(args.seed)

    # config 병합: 데이터셋 고정값 + CLI 인자
    config_module = importlib.import_module('datasets.config')
    config = config_module.get_config(args.market)
    config.update({
        'graph_type': args.graph_type,
        'kg_source':  args.kg_source,
        'lr':         args.lr,
        'epochs':     args.epochs,
        'alpha':      args.alpha,
        'gat_heads':  args.gat_heads,
        'seed':       args.seed,
        'use_wandb':      not args.no_wandb,
        'wandb_project':  args.wandb_project,
    })

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    market_prefix = config['market'] if config['market'] != 'NASDAQ' else ''
    run_name = (f"{market_prefix}_" if market_prefix else '') + \
               f"{config['graph_type']}_{config['kg_source']}_seed{config['seed']}"
    print(f"device: {device}")
    print(f"[config] market={config['market']}  graph_type={config['graph_type']}  "
          f"kg_source={config['kg_source']}  seed={config['seed']}")

    # ── wandb 초기화 ─────────────────────────────────────────
    # group: 같은 조건(graph_type × kg_source)으로 seed 여러 개 돌릴 때
    #        wandb 프로젝트 페이지에서 자동으로 평균/표준편차 계산해줌
    # name:  개별 run 식별자 → 논문 Table의 행(row) 하나
    if config['use_wandb']:
        group_name = (f"{market_prefix}_" if market_prefix else '') + \
                     f"{config['graph_type']}_{config['kg_source']}"
        run = wandb.init(
            project=config['wandb_project'],
            group=group_name,
            name=f"{group_name}_seed{config['seed']}",
            tags=[config['market'], config['graph_type'], config['kg_source']],
            config={k: v for k, v in config.items() if k != 'kg_sources'},
        )

    # ── 데이터 로드 ──────────────────────────────────────────
    load_module = importlib.import_module('datasets.load_data')
    importlib.reload(load_module)

    eod_data, mask_data, gt_data, price_data = load_module.load_eod_data(config)

    #%% 동적 KG(news_dynamic) vs 정적 KG 분기
    kg_snapshots = None
    if config['kg_source'] in ('news_dynamic', 'llm_dynamic'):
        kg_snapshots = load_module.load_kg_snapshots(config)
        kg_data = next(iter(kg_snapshots.values())) if kg_snapshots else None
    else:
        kg_data = load_module.load_kg(config)

    print(f"EOD shape: {eod_data.shape}  |  KG: {'None' if kg_data is None else kg_data.shape}"
          + (f"  snapshots={list(kg_snapshots.keys())}" if kg_snapshots else ''))

    # ── 모델 ─────────────────────────────────────────────────
    model_module = importlib.import_module('modules.model')
    importlib.reload(model_module)
    StockMixer = model_module.StockMixer
    get_loss   = model_module.get_loss

    model = StockMixer(config, kg_data=kg_data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    # ── train/val 함수 ────────────────────────────────────────
    train_module = importlib.import_module('modules.train')
    importlib.reload(train_module)
    train_one_epoch = train_module.train_one_epoch
    validate        = train_module.validate

    # ── 학습 루프 ────────────────────────────────────────────
    valid_index = config['valid_index']
    test_index  = config['test_index']
    trade_dates = mask_data.shape[1]

    batch_offsets = np.arange(valid_index, dtype=int)
    best_valid_loss = np.inf
    best_valid_perf = best_test_perf = None
    best_test_daily_ic = None   # per-day IC at best valid epoch

    for epoch in range(config['epochs']):
        np.random.shuffle(batch_offsets)

        tra_loss, tra_reg, tra_rank = train_one_epoch(
            model, optimizer, eod_data, mask_data, gt_data, price_data,
            config, batch_offsets, device,
            kg_snapshots=kg_snapshots,
        )

        val_loss, val_reg, val_rank, val_perf = validate(
            model, eod_data, mask_data, gt_data, price_data,
            config, valid_index, test_index, device,
            kg_snapshots=kg_snapshots,
        )

        test_loss, test_reg, test_rank, test_perf = validate(
            model, eod_data, mask_data, gt_data, price_data,
            config, test_index, trade_dates, device,
            kg_snapshots=kg_snapshots,
        )

        print(f"epoch {epoch+1:03d} | "
              f"train {tra_loss:.2e}={tra_reg:.2e}+α·{tra_rank:.2e} | "
              f"valid {val_loss:.2e} | "
              f"IC={val_perf['IC']:.4f} RIC={val_perf['RIC']:.4f} "
              f"prec@10={val_perf['prec_10']:.4f} SR={val_perf['sharpe5']:.4f}")

        if config['use_wandb']:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': tra_loss,
                'train/reg_loss': tra_reg,
                'train/rank_loss': tra_rank,
                'valid/loss': val_loss,
                'valid/IC': val_perf['IC'],
                'valid/RIC': val_perf['RIC'],
                'valid/prec_10': val_perf['prec_10'],
                'valid/sharpe5': val_perf['sharpe5'],
                'test/loss': test_loss,
                'test/IC': test_perf['IC'],
                'test/RIC': test_perf['RIC'],
                'test/prec_10': test_perf['prec_10'],
                'test/sharpe5': test_perf['sharpe5'],
            })

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            best_valid_perf = val_perf
            best_test_perf  = test_perf
            best_test_daily_ic = test_perf['_daily_ic'].copy()

    # ── 최종 결과 ────────────────────────────────────────────
    print("\n====== Best Result ======")
    print(f"graph_type={config['graph_type']}  kg_source={config['kg_source']}")
    # _daily_ic는 출력에서 제외
    print("Valid: " + "  ".join(f"{k}={v:.4f}" for k, v in best_valid_perf.items()
                                 if not k.startswith('_')))
    print("Test : " + "  ".join(f"{k}={v:.4f}" for k, v in best_test_perf.items()
                                 if not k.startswith('_')))

    # per-day IC 저장 (regime 분석용)
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    daily_ic_path = os.path.join(results_dir, f'{run_name}_daily_ic.npy')
    np.save(daily_ic_path, best_test_daily_ic)
    print(f"[saved] per-day IC → {daily_ic_path}")

    # ── Regime IC 계산 ────────────────────────────────────────
    _DATES_CSV = ('/gpfs/home1/pz29075/Capstone/StockMixer/'
                  'Temporal_Relational_Stock_Ranking/data/NASDAQ_aver_line_dates.csv')
    _EARNINGS = [('2017-01-09','2017-02-10'), ('2017-04-10','2017-05-12'),
                 ('2017-07-10','2017-08-11'), ('2017-10-09','2017-10-27')]

    import pandas as pd
    t_idx = config['test_index']
    n_test = len(best_test_daily_ic)

    # 일별 시장 평균 수익률 → Bull/Bear, High/Low Vol
    test_gt_arr   = gt_data[:,   t_idx: t_idx + n_test]
    test_mask_arr = mask_data[:, t_idx: t_idx + n_test]
    mkt_ret = np.nanmean(np.where(test_mask_arr > 0, test_gt_arr, np.nan), axis=0)

    bull  = mkt_ret > 0
    roll_vol = np.array([mkt_ret[max(0,i-19):i+1].std() for i in range(n_test)])
    high_vol = roll_vol >= np.median(roll_vol)

    # Earnings season (캘린더 날짜)
    try:
        dates_raw = pd.read_csv(_DATES_CSV, header=None)[0]
        test_dates = pd.to_datetime(dates_raw.iloc[t_idx: t_idx + n_test].values)
        earnings = np.zeros(n_test, dtype=bool)
        for s, e in _EARNINGS:
            earnings |= (test_dates >= pd.Timestamp(s)) & (test_dates <= pd.Timestamp(e))
    except Exception:
        earnings = np.zeros(n_test, dtype=bool)

    regimes = {
        'bull':     bull,
        'bear':     ~bull,
        'high_vol': high_vol,
        'low_vol':  ~high_vol,
        'earnings': earnings,
        'normal':   ~earnings,
    }
    regime_ic = {f'regime/{k}_ic': best_test_daily_ic[v].mean()
                 for k, v in regimes.items() if v.sum() > 0}

    for k, v in regime_ic.items():
        print(f"  {k}: {v:.4f}")

    if config['use_wandb']:
        # summary = 논문 Table에 들어갈 최종 숫자
        # Runs 페이지에서 열 선택 후 CSV export → 논문 Table 원본
        wandb.summary.update(
            {**{'best_valid/' + k: v for k, v in best_valid_perf.items()
                if not k.startswith('_')},
             **{'best_test/'  + k: v for k, v in best_test_perf.items()
                if not k.startswith('_')},
             **regime_ic}
        )
        wandb.finish()
