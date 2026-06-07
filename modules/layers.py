#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

#%%
"""공통 블록"""

class MixerBlock(nn.Module):
    def __init__(self, mlp_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.dense_1 = nn.Linear(mlp_dim, hidden_dim)
        self.act = nn.GELU()
        self.dense_2 = nn.Linear(hidden_dim, mlp_dim)
        self.dropout = dropout

    def forward(self, x):
        x = self.dense_1(x)
        x = self.act(x)
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.dense_2(x)
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class TriU(nn.Module):
    def __init__(self, time_step):
        super().__init__()
        self.time_step = time_step
        self.triU = nn.ParameterList([nn.Linear(i + 1, 1) for i in range(time_step)])

    def forward(self, inputs):
        x = self.triU[0](inputs[:, :, 0].unsqueeze(-1))
        for i in range(1, self.time_step):
            x = torch.cat([x, self.triU[i](inputs[:, :, 0:i + 1])], dim=-1)
        return x


class Mixer2dTriU(nn.Module):
    def __init__(self, time_steps, channels):
        super().__init__()
        self.LN_1 = nn.LayerNorm([time_steps, channels])
        self.LN_2 = nn.LayerNorm([time_steps, channels])
        self.timeMixer = TriU(time_steps)
        self.channelMixer = MixerBlock(channels, channels)

    def forward(self, inputs):
        x = self.LN_1(inputs)
        x = x.permute(0, 2, 1)
        x = self.timeMixer(x)
        x = x.permute(0, 2, 1)
        x = self.LN_2(x + inputs)
        y = self.channelMixer(x)
        return x + y


class MultTime2dMixer(nn.Module):
    """
    inputs: (S, T, F),  y: (S, scale_dim, F)  [conv 다운샘플 결과]
    output: (S, T*2+scale_dim, F)
    """
    def __init__(self, time_step, channel, scale_dim=8):
        super().__init__()
        self.mix_layer       = Mixer2dTriU(time_step, channel)
        self.scale_mix_layer = Mixer2dTriU(scale_dim, channel)

    def forward(self, inputs, y):
        y = self.scale_mix_layer(y)               # (S, scale_dim, F)
        x = self.mix_layer(inputs)                # (S, T, F)
        return torch.cat([inputs, x, y], dim=1)   # (S, T+T+scale_dim, F)


#%%
"""graph_type == 'none': KG 없이 stock mixing"""

class NoGraphMixer(nn.Module):
    """
    inputs shape: (S, F)  S: stocks, F: feat_dim
    """
    def __init__(self, stocks, hidden_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(stocks)
        self.dense1 = nn.Linear(stocks, hidden_dim)
        self.act = nn.Hardswish()
        self.dense2 = nn.Linear(hidden_dim, stocks)

    def forward(self, inputs):
        x = inputs.permute(1, 0)      # [F, S]
        x = self.layer_norm(x)
        x = self.dense1(x)
        x = self.act(x)
        x = self.dense2(x)
        return x.permute(1, 0)        # [S, F]


#%%
"""graph_type == 'gcn': GCN 기반 KG 주입
kg_data: D^{-1/2} A D^{-1/2} 정규화 adjacency [S, S] (고정, 학습 안 함)
"""

class KGMixer(nn.Module):
    """
    inputs shape: (S, F)
    adj   shape : (S, S)  — 미리 정규화된 고정 adjacency

    GCN 방식: 모든 이웃에 degree-normalized 고정 가중치 적용.
    이웃 간 중요도 차이를 학습하지 못하는 것이 GAT와의 핵심 차이.
    """
    def __init__(self, stocks, hidden_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(stocks)
        self.kg_transform = nn.Linear(stocks, stocks, bias=False)
        self.dense1 = nn.Linear(stocks, hidden_dim)
        self.act = nn.Hardswish()
        self.dense2 = nn.Linear(hidden_dim, stocks)

    def forward(self, inputs, adj):
        x = inputs.permute(1, 0)          # [F, S]
        x = self.layer_norm(x)
        x_kg = torch.mm(x, adj)           # [F, S] — 이웃 집계 (고정 가중치)
        x_kg = self.kg_transform(x_kg)    # [F, S] — W 변환
        x_mlp = self.dense1(x)
        x_mlp = self.act(x_mlp)
        x_mlp = self.dense2(x_mlp)
        return (x_mlp + x_kg).permute(1, 0)  # [S, F]


#%%
"""graph_type == 'gat': GAT 기반 KG 주입
kg_data: 이진 마스크 [S, S], 연결=0.0 / 비연결=-1e9 (구조만 제공)

GCN과의 차이:
  - adj 가중치가 고정이 아닌 현재 주가 feature에서 동적으로 계산됨
  - 같은 KG 엣지라도 날마다 attention 가중치가 달라짐
  - alpha_ij 를 저장해두면 "오늘 어떤 종목이 어떤 종목을 참고했는지" 해석 가능
"""

class KGHGATMixer(nn.Module):
    """
    inputs  : (S, feat_dim)
    kg_mask : (S, S, R)  — per-relation mask (0: edge, -1e9: no edge)

    relation별 독립 message passing → learned weighted sum.
    기존 GAT가 (S,S,R)→sum→(S,S)로 relation 구분을 버리는 것과 달리
    각 relation type의 attention을 따로 계산하고 rel_weight으로 가중합산.
    """
    def __init__(self, stocks, feat_dim, hidden_dim, num_heads=4, num_relations=6):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads     = num_heads
        self.d_k           = hidden_dim // num_heads
        self.num_relations = num_relations

        self.layer_norm = nn.LayerNorm(feat_dim)
        self.W          = nn.Linear(feat_dim, hidden_dim, bias=False)
        self.attn_src   = nn.Linear(self.d_k, 1, bias=False)
        self.attn_dst   = nn.Linear(self.d_k, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.out_proj   = nn.Linear(hidden_dim, feat_dim, bias=False)

        self.rel_weight = nn.Parameter(torch.ones(num_relations) / num_relations)

        self.dense1   = nn.Linear(feat_dim, hidden_dim)
        self.act      = nn.Hardswish()
        self.dense2   = nn.Linear(hidden_dim, feat_dim)
        self.out_norm = nn.LayerNorm(feat_dim)

    def forward(self, inputs, kg_mask):
        """
        inputs : (S, feat_dim)
        kg_mask: (S, S, R)
        returns: (S, feat_dim)
        """
        S = inputs.size(0)
        x = self.layer_norm(inputs)

        h     = self.W(x).view(S, self.num_heads, self.d_k)              # [S, H, d_k]
        a_src = self.attn_src(h).squeeze(-1)                              # [S, H]
        a_dst = self.attn_dst(h).squeeze(-1)                              # [S, H]
        e     = self.leaky_relu(a_src.unsqueeze(1) + a_dst.unsqueeze(0)) # [S, S, H]

        rel_w   = torch.softmax(self.rel_weight, dim=0)                   # [R]
        out_agg = torch.zeros(S, self.num_heads * self.d_k, device=inputs.device)

        for r in range(self.num_relations):
            mask_r  = kg_mask[:, :, r].unsqueeze(-1)                      # [S, S, 1]
            alpha_r = torch.softmax(e + mask_r, dim=1)                    # [S, S, H]
            out_r   = torch.einsum('ijh,jhd->ihd', alpha_r, h).reshape(S, -1)  # [S, H*d_k]
            out_agg = out_agg + rel_w[r] * out_r

        x_gat = self.out_proj(out_agg)                                    # [S, feat_dim]

        x_mlp = self.dense1(x)
        x_mlp = self.act(x_mlp)
        x_mlp = self.dense2(x_mlp)

        return self.out_norm(x_gat + x_mlp)


#%%
"""graph_type == 'gat': GAT 기반 KG 주입
kg_data: 이진 마스크 [S, S], 연결=0.0 / 비연결=-1e9 (구조만 제공)

GCN과의 차이:
  - adj 가중치가 고정이 아닌 현재 주가 feature에서 동적으로 계산됨
  - 같은 KG 엣지라도 날마다 attention 가중치가 달라짐
  - alpha_ij 를 저장해두면 "오늘 어떤 종목이 어떤 종목을 참고했는지" 해석 가능
"""

class KGGATMixer(nn.Module):
    """
    inputs shape: (S, feat_dim)
    kg_mask shape: (S, S)  — 0: KG 엣지, -1e9: 비연결

    Multi-head GAT:
      h_i  = W · x_i                        [S, num_heads, d_k]
      e_ij = LeakyReLU(a_src^T h_i + a_dst^T h_j)   [S, S, num_heads]
      α_ij = softmax_j(e_ij + mask_ij)      KG 구조로 마스킹
      out_i = Σ_j α_ij · h_j                이웃 정보 집계
    """
    def __init__(self, stocks, feat_dim, hidden_dim, num_heads=4):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = hidden_dim // num_heads

        self.layer_norm = nn.LayerNorm(feat_dim)
        self.W = nn.Linear(feat_dim, hidden_dim, bias=False)       # 특징 투영
        self.attn_src = nn.Linear(self.d_k, 1, bias=False)         # source 주의 벡터
        self.attn_dst = nn.Linear(self.d_k, 1, bias=False)         # destination 주의 벡터
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.out_proj = nn.Linear(hidden_dim, feat_dim, bias=False) # 출력 투영

        # MLP branch (잔차 경로)
        self.dense1 = nn.Linear(feat_dim, hidden_dim)
        self.act = nn.Hardswish()
        self.dense2 = nn.Linear(hidden_dim, feat_dim)
        self.out_norm = nn.LayerNorm(feat_dim)

    def forward(self, inputs, kg_mask):
        """
        inputs : (S, feat_dim)
        kg_mask: (S, S)
        returns: (S, feat_dim)
        """
        S = inputs.size(0)
        x = self.layer_norm(inputs)           # [S, feat_dim]

        # --- Attention branch ---
        h = self.W(x)                         # [S, hidden_dim]
        h = h.view(S, self.num_heads, self.d_k)  # [S, H, d_k]

        a_src = self.attn_src(h).squeeze(-1)  # [S, H]
        a_dst = self.attn_dst(h).squeeze(-1)  # [S, H]

        # e_ij = a_src_i + a_dst_j → [S, S, H]
        e = self.leaky_relu(
            a_src.unsqueeze(1) + a_dst.unsqueeze(0)
        )
        e = e + kg_mask.unsqueeze(-1)          # KG 비연결 → -inf → softmax 후 0
        alpha = torch.softmax(e, dim=1)        # [S, S, H]  (j 축으로 정규화)

        # 이웃 집계: Σ_j α_ij * h_j
        out = torch.einsum('ijh,jhd->ihd', alpha, h)  # [S, H, d_k]
        out = out.reshape(S, -1)               # [S, hidden_dim]
        x_gat = self.out_proj(out)             # [S, feat_dim]

        # --- MLP branch ---
        x_mlp = self.dense1(x)
        x_mlp = self.act(x_mlp)
        x_mlp = self.dense2(x_mlp)            # [S, feat_dim]

        return self.out_norm(x_gat + x_mlp)   # [S, feat_dim]
