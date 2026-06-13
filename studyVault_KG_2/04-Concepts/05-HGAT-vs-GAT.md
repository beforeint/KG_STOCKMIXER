---
title: HGAT vs GAT — Heterogeneous Graph Attention
tags: [architecture, hgat, gat, sparsity]
created: 2026-06-06
---

# HGAT vs GAT

## 핵심 차이

| | GAT | HGAT |
|--|-----|------|
| KG 입력 | (S,S,6) → sum → (S,S) | (S,S,6) 그대로 유지 |
| Relation 처리 | 타입 구분 없음 | 타입별 독립 attention |
| 집계 방식 | softmax(e + mask) | Σ rel_weight[r] × attention_r |
| 학습 파라미터 추가 | 없음 | rel_weight (R,) — relation 중요도 |

## 구현 위치

`modules/layers.py` — `KGHGATMixer`

```python
self.rel_weight = nn.Parameter(torch.ones(num_relations) / num_relations)

for r in range(self.num_relations):
    mask_r  = kg_mask[:, :, r].unsqueeze(-1)
    alpha_r = torch.softmax(e + mask_r, dim=1)
    out_r   = torch.einsum('ijh,jhd->ihd', alpha_r, h).reshape(S, -1)
    out_agg = out_agg + rel_w[r] * out_r
```

## 실험 결과 (NewsDynamic 기준)

- GAT: IC = 0.0201
- HGAT: IC = 0.0143 — GAT 미달, baseline(0.0156) 근접

## 실패 원인 분석

전체 엣지 수는 같지만 6개 채널로 나뉘면 채널당 엣지가 ~1/6.
N=1026 기준으로 채널당 연결이 매우 sparse해져서 attention이 의미 있게 학습되지 않음.

> "relation type 구분의 이득 < sparsity 증가의 손해"

## 논문 활용

Negative finding으로 사용:
> "HGAT이 GAT보다 낮은 성능을 보인 것은 relation-type granularity가 항상 유익하지 않으며,
>  KG 밀도가 충분히 확보될 때만 이점이 있음을 시사한다."

[[03-실험결과-비교]]
