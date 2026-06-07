#%%
import random
import numpy as np
import torch

from modules.model import get_loss
from evaluation.evaluation import evaluate

#%%
def _select_kg(offset, kg_snapshots, start_year, device):
    """
    offset → 연도 추정(252 거래일 = 1년) → 가장 가까운 과거 KG 스냅샷 반환.
    kg_snapshots=None 이면 None 반환 (정적 KG 모드 유지).
    """
    if kg_snapshots is None:
        return None
    year = start_year + offset // 252
    available = [y for y in kg_snapshots if y <= year]
    if not available:
        return None
    kg_np = kg_snapshots[max(available)]
    return torch.tensor(kg_np, dtype=torch.float32).to(device)

#%%
def get_batch(eod_data, mask_data, gt_data, price_data, config, offset=None):
    T    = config['lookback_length']
    steps = config['steps']
    valid_index = config['valid_index']
    if offset is None:
        offset = random.randrange(0, valid_index)
    mask_batch = mask_data[:, offset: offset + T + steps]
    mask_batch = np.min(mask_batch, axis=1)
    return (
        eod_data[:, offset: offset + T, :],
        np.expand_dims(mask_batch, axis=1),
        np.expand_dims(price_data[:, offset + T - 1], axis=1),
        np.expand_dims(gt_data[:, offset + T + steps - 1], axis=1),
    )


#%%
def validate(model, eod_data, mask_data, gt_data, price_data, config,
             start_index, end_index, device,
             kg_snapshots=None, start_year=2013):
    """
    start_index~end_index 구간 loss + 성능 지표 반환.
    kg_snapshots: {year: np.ndarray} 동적 KG dict. None이면 정적 모드.
    """
    S     = config['stock_num']
    T     = config['lookback_length']
    steps = config['steps']
    alpha = config['alpha']

    with torch.no_grad():
        pred_arr = np.zeros([S, end_index - start_index], dtype=float)
        gt_arr   = np.zeros([S, end_index - start_index], dtype=float)
        mask_arr = np.zeros([S, end_index - start_index], dtype=float)
        total_loss = reg_loss = rank_loss = 0.0

        for cur_offset in range(start_index - T - steps + 1,
                                end_index  - T - steps + 1):
            data_b, mask_b, price_b, gt_b = map(
                lambda x: torch.tensor(x, dtype=torch.float32).to(device),
                get_batch(eod_data, mask_data, gt_data, price_data, config, cur_offset)
            )
            adj  = _select_kg(cur_offset, kg_snapshots, start_year, device)
            pred = model(data_b, adj=adj)
            l, rl, kl, rr = get_loss(pred, gt_b, price_b, mask_b, S, alpha)
            total_loss += l.item()
            reg_loss   += rl.item()
            rank_loss  += kl.item()
            col = cur_offset - (start_index - T - steps + 1)
            pred_arr[:, col] = rr[:, 0].cpu().numpy()
            gt_arr[:, col]   = gt_b[:, 0].cpu().numpy()
            mask_arr[:, col] = mask_b[:, 0].cpu().numpy()

        n = end_index - start_index
        perf = evaluate(pred_arr, gt_arr, mask_arr)
    return total_loss / n, reg_loss / n, rank_loss / n, perf


#%%
def train_one_epoch(model, optimizer, eod_data, mask_data, gt_data, price_data,
                    config, batch_offsets, device,
                    kg_snapshots=None, start_year=2013):
    """
    kg_snapshots: {year: np.ndarray} — 동적 KG. None이면 정적 모드(기존 동작).
    start_year:   offset 0에 해당하는 연도 (NASDAQ 기본 2013).
    """
    S     = config['stock_num']
    T     = config['lookback_length']
    steps = config['steps']
    valid_index = config['valid_index']
    alpha = config['alpha']

    model.train()
    total_loss = reg_loss = rank_loss = 0.0
    n_steps = valid_index - T - steps + 1

    for j in range(n_steps):
        offset = batch_offsets[j]
        data_b, mask_b, price_b, gt_b = map(
            lambda x: torch.tensor(x, dtype=torch.float32).to(device),
            get_batch(eod_data, mask_data, gt_data, price_data, config, offset)
        )
        optimizer.zero_grad()
        adj  = _select_kg(offset, kg_snapshots, start_year, device)
        pred = model(data_b, adj=adj)
        l, rl, kl, _ = get_loss(pred, gt_b, price_b, mask_b, S, alpha)
        l.backward()
        optimizer.step()
        total_loss += l.item()
        reg_loss   += rl.item()
        rank_loss  += kl.item()

    return total_loss / n_steps, reg_loss / n_steps, rank_loss / n_steps
