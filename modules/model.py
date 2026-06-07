#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.layers import (
    MultTime2dMixer,
    NoGraphMixer,
    KGMixer,
    KGGATMixer,
    KGHGATMixer,
)

#%%
"""Loss"""

def get_loss(prediction, ground_truth, base_price, mask, stock_num, alpha):
    device = prediction.device
    all_one = torch.ones(stock_num, 1, dtype=torch.float32, device=device)
    return_ratio = torch.div(torch.sub(prediction, base_price), base_price)
    reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)
    pre_pw_dif = return_ratio @ all_one.t() - all_one @ return_ratio.t()
    gt_pw_dif  = all_one @ ground_truth.t() - ground_truth @ all_one.t()
    mask_pw = mask @ mask.t()
    rank_loss = torch.mean(F.relu(pre_pw_dif * gt_pw_dif * mask_pw))
    loss = reg_loss + alpha * rank_loss
    return loss, reg_loss, rank_loss, return_ratio


#%%
"""StockMixer

graph_type 에 따라 stock_mixer 가 자동 선택됨:
  'none' → NoGraphMixer
  'gcn'  → KGMixer        (kg_data: 정규화 adjacency)
  'gat'  → KGGATMixer     (kg_data: 이진 마스크 S×S)
  'hgat' → KGHGATMixer    (kg_data: per-relation 마스크 S×S×R)

새로운 그래프 방식을 추가하려면 layers.py 에 클래스를 추가하고
여기 분기만 추가하면 됨. 데이터 로딩(load_data.py)은 건드릴 필요 없음.
"""

class StockMixer(nn.Module):
    """
    inputs shape: (S, T, F)
    S: stocks, T: lookback_length, F: fea_num
    """
    def __init__(self, config: dict, kg_data=None):
        super().__init__()
        S       = config['stock_num']
        T       = config['lookback_length']
        F       = config['fea_num']
        market  = config['market_num']
        scale   = config['scale_factor']
        gtype   = config.get('graph_type', 'none')
        n_heads = config.get('gat_heads', 4)
        scale_dim = 8
        feat_dim  = T * 2 + scale_dim   # channel_fc 이후 time 축 크기

        self.graph_type = gtype
        self.mixer  = MultTime2dMixer(T, F, scale_dim=scale_dim)
        self.conv   = nn.Conv1d(F, F, kernel_size=2, stride=2)
        self.channel_fc  = nn.Linear(F, 1)
        self.time_fc     = nn.Linear(feat_dim, 1)
        self.time_fc_    = nn.Linear(feat_dim, 1)

        if gtype == 'gcn' and kg_data is not None:
            self.register_buffer('kg_data', torch.FloatTensor(kg_data))
            self.stock_mixer = KGMixer(S, market)
        elif gtype == 'gat' and kg_data is not None:
            self.register_buffer('kg_data', torch.FloatTensor(kg_data))
            self.stock_mixer = KGGATMixer(S, feat_dim, market, num_heads=n_heads)
        elif gtype == 'hgat' and kg_data is not None:
            n_rels = kg_data.shape[2] if kg_data.ndim == 3 else 6
            self.register_buffer('kg_data', torch.FloatTensor(kg_data))
            self.stock_mixer = KGHGATMixer(S, feat_dim, market, num_heads=n_heads, num_relations=n_rels)
        else:
            self.register_buffer('kg_data', None)
            self.stock_mixer = NoGraphMixer(S, market)

    def forward(self, inputs, adj=None):
        """
        inputs: (S, T, F)
        adj:    동적 KG adjacency (S, S) or (S, S, R) — None이면 self.kg_data 사용
        returns: (S, 1)
        """
        x = inputs.permute(0, 2, 1)      # [S, F, T]
        x = self.conv(x)                 # [S, F, T//2]
        x = x.permute(0, 2, 1)          # [S, T//2, F]
        y = self.mixer(inputs, x)        # [S, feat_dim]
        y = self.channel_fc(y).squeeze(-1)  # [S, feat_dim]

        #%% KG branch: 동적 adj 우선, 없으면 학습 시 고정된 self.kg_data
        kg = adj if adj is not None else self.kg_data
        if kg is not None:
            z = self.stock_mixer(y, kg)
        else:
            z = self.stock_mixer(y)

        return self.time_fc(y) + self.time_fc_(z)  # [S, 1]
