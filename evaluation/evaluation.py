#%%
import numpy as np
import pandas as pd


#%%
def evaluate(prediction, ground_truth, mask):
    """
    prediction, ground_truth, mask: (S, T_eval)
    returns dict: mse, IC, RIC, prec_10, sharpe5
    """
    assert ground_truth.shape == prediction.shape
    perf = {}
    perf['mse'] = (np.linalg.norm((prediction - ground_truth) * mask) ** 2
                   / np.sum(mask))

    df_pred = pd.DataFrame(prediction * mask)
    df_gt   = pd.DataFrame(ground_truth * mask)

    ic, prec_10, sharpe_li5 = [], [], []

    for i in range(prediction.shape[1]):
        ic.append(df_pred[i].corr(df_gt[i]))

        rank_gt  = np.argsort(ground_truth[:, i])
        rank_pre = np.argsort(prediction[:, i])

        gt_top10  = _topk_valid(rank_gt,  mask[:, i], k=10)
        pre_top5  = _topk_valid(rank_pre, mask[:, i], k=5)
        pre_top10 = _topk_valid(rank_pre, mask[:, i], k=10)

        prec = sum(ground_truth[p][i] >= 0 for p in pre_top10) / 10
        prec_10.append(prec)

        ret5 = sum(ground_truth[p][i] for p in pre_top5) / 5
        sharpe_li5.append(ret5)

    perf['IC']      = np.nanmean(ic)
    perf['RIC']     = np.nanmean(ic) / (np.nanstd(ic) + 1e-8)
    s = np.array(sharpe_li5)
    perf['sharpe5'] = (np.mean(s) / (np.std(s) + 1e-8)) * 15.87
    perf['prec_10'] = np.mean(prec_10)
    perf['_daily_ic'] = np.array(ic, dtype=float)   # per-day IC (not for paper table — regime analysis용)
    return perf


def _topk_valid(rank, mask, k):
    """rank 내림차순으로 mask=1 인 상위 k개 인덱스 반환."""
    selected = []
    for j in range(1, len(rank) + 1):
        idx = rank[-j]
        if mask[idx] >= 0.5:
            selected.append(idx)
        if len(selected) == k:
            break
    return selected
