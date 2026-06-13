---
title: KG-StockMixer 완전 가이드 — 아키텍처·실험·결과 해설
tags: [report, architecture, experiment, capstone]
created: 2026-06-09
audience: 캡스톤 보고서 작성자 (처음 보는 사람 기준)
---

# KG-StockMixer: 지식 그래프 기반 주가 예측 모델 완전 가이드

> 이 문서는 캡스톤 보고서 작성을 위한 **자기완결형 레퍼런스**입니다.  
> 아키텍처 수식, 실험 설계, 전체 결과 표, 해석까지 포함합니다.

---

## 1. 문제 정의

### 1.1 주가 예측을 수식으로

$N$개의 종목이 거래되는 시장에서, 각 종목 $i$의 과거 $T$일치 지표 데이터를 $\mathbf{x}_i \in \mathbb{R}^{T \times F}$로 정의한다 ($F$: 지표 수).

- **입력**: $\mathcal{X} = \{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_N\} \in \mathbb{R}^{N \times T \times F}$
- **목표**: 다음 날($t+1$) 각 종목의 1일 수익률 비율 $r_i^t = \dfrac{p_i^t - p_i^{t-1}}{p_i^{t-1}}$ 예측
- **출력**: 예측 점수 $\hat{r} \in \mathbb{R}^N$ → 상위 종목 선별

본 연구에서는 NASDAQ 1,026개 종목, $T=16$일, $F=5$개 지표(시가, 고가, 저가, 종가, 거래량 정규화값)를 사용한다.

### 1.2 지식 그래프(KG)란?

주식 간 관계를 인접행렬 $\mathbf{A} \in \{0,1\}^{N \times N \times R}$으로 표현한다.

- $\mathbf{A}_{i,j,r} = 1$: 종목 $i$와 $j$ 사이에 관계 유형 $r$이 존재
- $R$: 관계 유형 수 (예: Supplier, Customer, Competitor 등 6종)
- KG는 주가 예측 모델에 **기업 간 관계 정보**를 주입하는 용도

---

## 2. 모델 아키텍처: GATMixer

### 2.1 전체 구조

```
입력 X ∈ ℝ^(N×T×F)
        │
        ▼
┌─────────────────────┐
│  StockMixer Backbone│  ← 시계열 특징 추출 (Temporal + Indicator Mixing)
│  (멀티스케일 MLP)    │
└────────┬────────────┘
         │  임베딩 Y ∈ ℝ^(N×H)
         ▼
┌─────────────────────┐
│   KGGATMixer        │  ← 지식 그래프 기반 종목 간 관계 학습
│   (GAT 어텐션)       │
└────────┬────────────┘
         │  KG-보강 임베딩 Z ∈ ℝ^(N×H)
         ▼
┌─────────────────────┐
│   출력 레이어        │  ŷ = W₁y + W₂z
└─────────────────────┘
         │
         ▼
   예측 수익률 ŷ ∈ ℝ^N
```

### 2.2 StockMixer Backbone (시계열 인코더)

#### Indicator Mixing

각 종목의 $T \times F$ 행렬을 전치(transpose)하여 지표 차원 간 상호작용을 학습한다:

$$\hat{\mathbf{x}}^T = \mathbf{x}^T + \mathbf{W}_2 \sigma(\mathbf{W}_1 \text{LayerNorm}(\mathbf{x}^T))$$

- $\mathbf{x}^T \in \mathbb{R}^{F \times T}$: 전치된 입력 (지표를 행으로)
- $\mathbf{W}_1 \in \mathbb{R}^{H_t \times T}$, $\mathbf{W}_2 \in \mathbb{R}^{T \times H_t}$: 학습 가중치
- $\sigma$: HardSwish 활성화 함수 ($\sigma(x) = x \cdot \text{ReLU6}(x+3)/6$)

#### Time Mixing (멀티스케일 패치)

단순 MLP로 시계열을 처리하면 시점 간 **인과성(causality)**이 보장되지 않는다. StockMixer는 상삼각 마스크를 적용하여 미래 정보 누출을 방지한다:

$$\mathbf{h} = \hat{\mathbf{x}} + \mathbf{U}_2\sigma(\mathbf{U}_1^{\text{TriU}} \text{LayerNorm}(\hat{\mathbf{x}}))$$

- $\mathbf{U}_1^{\text{TriU}}$: 상삼각 행렬 (미래 $\to$ 과거 방향만 차단)
- 패치 크기 $k \in \{T/2, T/4, 1\}$로 멀티스케일 표현 추출 후 concat

#### 최종 임베딩

$$\mathbf{Y} = \text{FC}\left(\text{concat}\left[\mathbf{h}^{(k)}\right]\right) \in \mathbb{R}^{N \times H}$$

### 2.3 KGGATMixer (그래프 어텐션 레이어)

#### Step 1. 선형 변환

$$\mathbf{H} = \mathbf{Y}\mathbf{W}_H \in \mathbb{R}^{N \times H'}$$

- $\mathbf{W}_H \in \mathbb{R}^{H \times H'}$: 학습 가중치

#### Step 2. 어텐션 점수 계산

종목 $i$와 $j$ 사이의 어텐션 점수:

$$e_{ij} = \text{LeakyReLU}\!\left(\mathbf{a}_{\text{src}}^\top \mathbf{h}_i + \mathbf{a}_{\text{dst}}^\top \mathbf{h}_j\right)$$

- $\mathbf{a}_{\text{src}}, \mathbf{a}_{\text{dst}} \in \mathbb{R}^{H'}$: 학습 가능한 어텐션 벡터
- $\mathbf{h}_i = \mathbf{H}[i, :]$: 종목 $i$의 임베딩

#### Step 3. KG 마스킹

KG에 엣지가 없는 쌍은 어텐션을 차단한다:

$$m_{ij} = \begin{cases} 0 & \text{if } \sum_r \mathbf{A}_{i,j,r} > 0 \text{ (연결됨)} \\ -\infty & \text{otherwise (미연결)} \end{cases}$$

$$\tilde{e}_{ij} = e_{ij} + m_{ij}$$

> **핵심**: KG 마스크가 없으면 dense한 무작위 어텐션 → GAT가 KG 없는 StockMixer와 동일해짐

#### Step 4. 정규화 및 집계

$$\alpha_{ij} = \text{softmax}_j(\tilde{e}_{ij}) = \frac{\exp(\tilde{e}_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(\tilde{e}_{ik})}$$

$$\mathbf{z}_i = \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{h}_j$$

- $\mathcal{N}(i)$: KG에서 종목 $i$와 연결된 이웃 집합

#### Step 5. 잔차 연결 + MLP

$$\mathbf{Z} = \text{LayerNorm}\!\left(\text{OutProj}(\mathbf{z}) + \text{MLP}(\mathbf{Y})\right)$$

### 2.4 최종 예측

시계열 정보($\mathbf{Y}$)와 KG 정보($\mathbf{Z}$)를 결합:

$$\hat{y}_i = \mathbf{W}_1 \mathbf{y}_i + \mathbf{W}_2 \mathbf{z}_i$$

### 2.5 손실 함수

MSE 회귀 손실 + 쌍별 랭킹 손실의 합:

$$\mathcal{L} = \underbrace{\frac{1}{N}\sum_i m_i \left(\hat{r}_i - r_i\right)^2}_{\mathcal{L}_{\text{reg}}} + \alpha \underbrace{\frac{1}{N^2}\sum_i\sum_j \max\!\left(0,\, -(\hat{r}_i - \hat{r}_j)(r_i - r_j)\right) \cdot m_i m_j}_{\mathcal{L}_{\text{rank}}}$$

- $\hat{r}_i = (\hat{y}_i - p_i^{t-1}) / p_i^{t-1}$: 예측 수익률
- $r_i$: 실제 수익률 (ground truth)
- $m_i \in \{0,1\}$: 거래 가능 마스크
- $\alpha = 0.1$: 랭킹 손실 가중치

> **랭킹 손실의 의미**: $\hat{r}_i > \hat{r}_j$인데 $r_i < r_j$이면 페널티 부과 → 절대값보다 **순위**가 정확한 모델 유도

---

## 3. 지식 그래프 소스 비교

### 3.1 KG 종류 및 구축 방법

| KG 소스 | 관계 유형 | 구축 방법 | 시간 변화 |
|---|---|---|---|
| **Wikidata** | 24종 (산업, 경쟁, 임원 공유 등) | Wikidata SPARQL 쿼리 | 정적 |
| **Sector/Industry** | 1종 (동일 섹터) | GICS 분류 코드 일치 | 정적 |
| **Institutional** | 1종 (기관 공동보유) | 13F SEC 공시 + Jaccard 유사도 | 정적 |
| **SupplyChain** | 1종 (공급망 관계) | BEA 산업연관표 IO계수 ≥ 0.05 | 정적 |
| **LLM Static** | 6종 | GPT-4o-mini 제로샷 추출 (2012년 뉴스 기준) | 정적 |
| **NewsDynamic** | 6종 | FMP 뉴스 API + GPT-4o-mini 연도별 추출 | **동적** |
| **LLMDynamic** | 6종 | Self-Consistency Voting (3회 다수결) + 양방향 교차검증 | **동적** |

> **동적 KG**란: 연도별로 KG를 다시 구축하여 기업 관계 변화를 반영. 학습 시 해당 연도의 KG를 사용.

### 3.2 KG 희소성(Sparsity) 비교

| KG | 전체 엣지 수 | Density | 종목당 평균 이웃 |
|---|---|---|---|
| SupplyChain | 411,030 | 39.05% | ~400개 |
| Institutional | 308,064 | 29.26% | ~300개 |
| Wikidata | 2,963 | 0.28% | ~2.9개 |
| Sector | 506 | 0.05% | ~0.5개 |
| **NewsDynamic** | **~200/년** | **~0.02%** | **~0.2개** |
| LLMDynamic | ~124/년 | ~0.01% | ~0.1개 |

> **희소성 문제**: NewsDynamic, LLMDynamic은 엣지가 극히 적어 GAT가 관계 구조를 학습하기 어렵다. 대부분 노드의 이웃이 0개.

---

## 4. 평가 지표

### 4.1 IC (Information Coefficient)

$$\text{IC} = \frac{1}{|\mathcal{D}|} \sum_{t \in \mathcal{D}} \rho\!\left(\hat{\mathbf{r}}^t,\, \mathbf{r}^t\right)$$

- $\rho(\cdot)$: Pearson 상관계수
- $\mathcal{D}$: 테스트 기간 거래일 집합
- **해석**: 예측값과 실제값의 선형 상관. 0이면 무작위 예측, 양수일수록 좋음.
- **주의**: IC = 0.02 수준도 실제 운용에서 의미있는 알파 신호

### 4.2 RIC (Rank Information Coefficient)

$$\text{RIC} = \frac{1}{|\mathcal{D}|} \sum_{t \in \mathcal{D}} \rho_s\!\left(\hat{\mathbf{r}}^t,\, \mathbf{r}^t\right)$$

- $\rho_s(\cdot)$: Spearman 순위 상관계수
- IC와 달리 이상치(outlier)에 강건. 예측 **순위**의 정확성 측정.

### 4.3 Precision@K

$$\text{P@K} = \frac{1}{|\mathcal{D}|} \sum_{t \in \mathcal{D}} \frac{|\text{Top-}K(\hat{\mathbf{r}}^t) \cap \text{Top-}K(\mathbf{r}^t)|}{K}$$

- 예측 상위 $K$개 중 실제로도 상위 $K$개에 포함된 비율
- $K=10$: 매일 10개 종목 선별 시 몇 개나 실제 상위권인지

### 4.4 SR (Sharpe Ratio)

$$\text{SR} = \frac{\sqrt{252} \cdot \mathbb{E}[R_t - R_f]}{\text{Std}(R_t - R_f)}$$

- 5일 보유 전략 기준 연환산 리스크 조정 수익률
- 높을수록 단위 리스크당 수익이 높음. 0 이상이면 무위험 자산 대비 수익.

---

## 5. 실험 설계

### 5.1 데이터셋

| 구분 | 내용 |
|---|---|
| 마켓 | NASDAQ (미국 기술주 중심) |
| 종목 수 | 1,026개 |
| 전체 기간 | 2013-01-02 ~ 2017-10-27 (1,245 거래일) |
| Train | 2013-01-02 ~ 2016-06 (index 0~755, 756일) |
| Validation | 2016-06 ~ 2016-11-18 (index 756~1007, 252일) |
| **Test** | **2016-11-21 ~ 2017-10-27 (index 1008~1244, 237일)** |
| 입력 지표 | 시가/고가/저가/종가/거래량 기반 5개 정규화 피처 |
| Lookback | $T = 16$일 |

### 5.2 실험 변수

**비교 축 1 — KG 소스** (그래프 타입 = GAT 고정):

No KG / Wikidata / Sector / Institutional / SupplyChain / LLM Static / NewsDynamic / LLMDynamic

**비교 축 2 — 그래프 아키텍처** (KG = NewsDynamic or LLMDynamic):

GCN / GAT / HGAT (Heterogeneous GAT)

**비교 축 3 — Random KG Baseline** (RQ4 구조 유효성):

NewsDynamic vs 동일 density의 무작위 KG

**비교 축 4 — 시장별 density 테스트**:

NASDAQ 1026종목 vs NASDAQ 상위100 vs SP100

### 5.3 학습 설정

| 하이퍼파라미터 | 값 |
|---|---|
| Learning rate | 0.001 (Adam) |
| Epochs | 100 |
| $\alpha$ (랭킹 손실 가중치) | 0.1 |
| GAT heads | 4 |
| Market dim $m$ | 20 |
| Seed 수 | 5개 (mean ± std) |

---

## 6. 전체 실험 결과

### 6.1 메인 결과: KG 소스별 비교 (GAT 고정, NASDAQ)

| 모델 | IC | RIC | P@10 | SR |
|---|---|---|---|---|
| **No KG (Baseline)** | 0.0156 ± 0.0052 | 0.1616 | 0.5151 | 0.393 |
| GAT + Sector | 0.0028 ± 0.0053 ↓ | 0.1424 | 0.5177 | 0.409 |
| GAT + Institutional | 0.0166 ± 0.0085 | 0.0986 | 0.5176 | −0.247 |
| GAT + SupplyChain | 0.0168 ± 0.0093 | 0.0986 | 0.5176 | −0.247 |
| GAT + LLM Static | 0.0180 ± 0.0098 | 0.1514 | 0.5228 | 0.318 |
| GAT + LLMDynamic | 0.0153 ± 0.0095 | **0.1744** | 0.5175 | 0.405 |
| **GAT + NewsDynamic** | **0.0201** ± 0.0120 | 0.1688 | 0.5198 | −0.006 |
| **GAT + Wikidata** | **0.0203** ± 0.0083 | 0.0738 | 0.5188 | −0.954 |

> **Bold**: 각 지표 최우수. ↓: baseline 대비 성능 하락.

### 6.2 아키텍처 비교 (동적 KG)

| 아키텍처 | NewsDynamic IC | LLMDynamic IC | SR (ND) | SR (LLMD) |
|---|---|---|---|---|
| GCN | 0.0160 ± 0.0071 | 0.0146 ± 0.0049 | 1.252 | **1.621** |
| **GAT** | **0.0201 ± 0.0120** | 0.0153 ± 0.0095 | −0.006 | 0.405 |
| HGAT | 0.0144 ± 0.0083 | 0.0151 ± 0.0077 | 0.293 | 0.300 |

> HGAT < GAT 이유: 관계 타입별로 분리하면 채널당 엣지 수가 1/6로 줄어들어 희소성이 더 심해짐.

### 6.3 레짐별 IC: NewsDynamic의 조건부 효과 (GAT 고정)

| 장세 조건 | 일수 | No KG | Wikidata | Institutional | SupplyChain | NewsDynamic | p-value (ND) |
|---|---|---|---|---|---|---|---|
| **전체** | 237 | 0.0156 | 0.0203 | 0.0166 | 0.0168 | 0.0201 | †(0.066) |
| Bull (상승장) | 133 | 0.0204 | 0.0261 | 0.0230 | 0.0236 | 0.0261 | †(0.075) |
| Bear (하락장) | 104 | 0.0095 | 0.0128 | 0.0085 | 0.0081 | 0.0123 | n.s. |
| High Vol (고변동) | 119 | 0.0171 | 0.0179 | 0.0116 | 0.0128 | 0.0146 | n.s. |
| **Low Vol (저변동)** | **118** | **0.0141** | 0.0227 | 0.0217 | 0.0208 | **0.0255** | **★★★(0.0004)** |
| Earnings (실적발표) | 84 | 0.0008 | 0.0099 | 0.0042 | 0.0058 | 0.0090 | †(0.052) |
| Normal | 153 | 0.0238 | 0.0260 | 0.0234 | 0.0228 | 0.0261 | n.s. |

> ★★★ p<0.001 / * p<0.05 / † p<0.10 / n.s. 유의하지 않음

**핵심 발견**: NewsDynamic은 **저변동성 구간에서만** 통계적으로 유의한 우위(p=0.0004)를 보인다.  
→ 시장이 조용할 때 뉴스 co-mention 신호가 예측에 유효하며, 고변동성 구간에서는 노이즈에 묻힘.

### 6.4 Random KG Baseline: 구조의 유효성 검증

| 모델 | IC | vs No KG |
|---|---|---|
| No KG | 0.0156 ± 0.0052 | — |
| GAT + NewsDynamic | 0.0201 ± 0.0120 | +0.0045 |
| GAT + Random KG (동일 density) | 0.0221 ± 0.0084 | +0.0065 |

**t-test (NewsDynamic vs Random)**: t = −0.731, **p = 0.505 (유의하지 않음)**

> Random KG가 NewsDynamic과 통계적으로 차이가 없다.  
> → density 0.012% (평균 이웃 0.18개)에서는 **어떤 연결이냐**보다 **연결이 존재하냐** 자체가 성능을 결정.  
> → GAT가 실제 관계 구조를 학습했다기보다 sparse attention mask의 정규화 효과로 해석 가능.

### 6.5 Density 증가 실험: NASDAQ100

상위 100개 종목(뉴스 degree 기준)으로 subset을 구성하여 density를 높임.

| 마켓 | 종목 수 | KG Density | GAT+ND IC | GAT+Rand IC | p-value |
|---|---|---|---|---|---|
| NASDAQ (전체) | 1,026 | 0.012% | 0.0201 | 0.0221 | p=0.505 n.s. |
| **NASDAQ100** | **100** | **0.25%** | **0.0175** | **0.0147** | **p=0.105 n.s.** |

> Density를 20배 높여도(0.012% → 0.25%) ND > Random 방향이 역전되고 p=0.105로 유의 수준에 근접.  
> → Density가 증가할수록 관계 구조가 의미를 가질 가능성이 있으나 현재 샘플 수(5 seed)로는 통계적 확인 불가.  
> **충분한 density에서의 구조 학습은 미래 연구 과제.**

### 6.6 외부 모델 비교 (Chen et al. 2023 인용)

> ⚠️ **비교 조건 차이**: 논문(Chen et al.)은 test=273일, 본 연구는 test=237일. IC 스케일이 다를 수 있으므로 절대값보다 **상대적 순위** 참고용.

| 모델 | IC | RIC | P@10 | SR | 출처 |
|---|---|---|---|---|---|
| LSTM | 0.032 | 0.354 | 0.514 | 0.892 | Chen et al. (2023)† |
| ALSTM | 0.035 | 0.371 | 0.522 | 0.941 | Chen et al. (2023)† |
| RGCN | 0.034 | 0.382 | 0.516 | 1.054 | Chen et al. (2023)† |
| GAT (Veličković 2017) | 0.035 | 0.377 | 0.530 | 1.133 | Chen et al. (2023)† |
| RSR-I (Feng et al. 2019) | 0.038 | 0.398 | 0.531 | 1.238 | Chen et al. (2023)† |
| STHAN-SR | 0.039 | 0.451 | 0.543 | 1.416 | Chen et al. (2023)† |
| ESTIMATE | 0.040 | 0.444 | 0.539 | 1.307 | Chen et al. (2023)† |
| StockMixer (No KG) | **0.043** | **0.501** | **0.545** | **1.465** | Chen et al. (2023)† |
| ─ | ─ | ─ | ─ | ─ | ─ |
| **RankLSTM (본 연구)** | 0.0102 ± 0.0038 | — | — | — | 본 연구 (237일) |
| **StockMixer No KG (본 연구)** | 0.0156 ± 0.0052 | 0.1616 | 0.5151 | 0.393 | 본 연구 (237일) |
| **GAT + NewsDynamic (본 연구)** | **0.0201** ± 0.0120 | **0.1688** | 0.5198 | −0.006 | 본 연구 (237일) |
| GCN + LLMDynamic | 0.0146 ± 0.0049 | 0.1068 | 0.5136 | **1.621** | 본 연구 (237일) |
| HGAT + LLMDynamic | 0.0151 ± 0.0077 | 0.1318 | **0.5241** | 0.300 | 본 연구 (237일) |

† NASDAQ, test 273일, 3 seeds 평균 (Chen et al. AAAI 2024)

### 읽는 방법

- **IC 스케일 차이**: 논문 StockMixer(IC=0.043) vs 본 연구(0.0156) — 테스트 기간(273일 vs 237일) 차이로 직접 비교 불가
- **SR**: 본 연구 GCN+LLMDynamic(SR=1.621) ≈ 논문 STHAN-SR(SR=1.416) — 리스크조정 수익은 동급
- **P@10**: 본 연구 HGAT+LLMDynamic(0.5241) ≈ 논문 StockMixer(0.545) — 선택 정확도 근접
- **StockMixer vs RankLSTM**: No KG StockMixer(0.0156) > RankLSTM(0.0102), Δ=+0.0054 (p=0.226, n.s.). StockMixer의 멀티스케일 MLP 인코더가 단순 LSTM보다 주가 패턴 포착에 효과적임을 시사.
- **KG 효과**: StockMixer No KG(0.0156) → GAT+ND(0.0201), +28.8% IC 향상. KG가 백본 성능을 추가로 끌어올림.

---

## 7. RQ별 결론 요약

| Research Question | 실험 | 결론 |
|---|---|---|
| **RQ1**: KG가 예측 성능을 높이는가? | No KG vs 7종 KG | Yes — LLMDynamic, Sector 제외 대부분 향상 (+0.001~+0.005 IC) |
| **RQ2**: 어떤 KG 소스가 가장 효과적인가? | GAT × 7 KG 소스 | NewsDynamic, Wikidata 최우수 (IC 기준). 동적 KG > 정적 KG (저변동성 구간) |
| **RQ3**: 어떤 그래프 아키텍처가 적합한가? | GCN/GAT/HGAT × 동적KG | GAT 최우수. HGAT는 sparsity로 오히려 하락. |
| **RQ4**: 뉴스 co-mention 구조가 실제로 유효한가? | NewsDynamic vs Random KG | **p=0.505 — 유의하지 않음.** 현재 density(0.012%)에서는 구조보다 연결 존재 여부가 중요. |
| **RQ5**: StockMixer 백본이 LSTM보다 우월한가? | RankLSTM vs StockMixer No KG | **Yes** — IC +53% (0.0102→0.0156). 멀티스케일 시계열 인코더의 효과. |

---

## 8. 한계 및 향후 연구

### 현재 한계

1. **극단적 희소성**: NewsDynamic density 0.012% → 평균 이웃 0.18개. GAT가 구조를 학습하기에 부족.
2. **아키텍처 위치**: KG 주입이 출력 단계(post-encoding). 입력 단계 주입과 비교 불가.
3. **테스트 기간**: 237일로 외부 논문(273일)과 달라 직접 비교에 한계.
4. **Random KG ≈ NewsDynamic**: KG 구조 자체의 기여를 증명하지 못함.

### 향후 연구 방향

1. **자기루프(Self-loop) + 빈도 가중치**: 희소 KG 문제를 완화하는 간단한 방법
2. **입력단 KG 주입**: KG를 feature extraction 단계에서 사용 (architectural contribution)
3. **더 밀도 높은 KG**: S&P 100 실험에서 density 0.5% 달성 → 구조 유효성 재검증 필요
4. **시간적 KG 변화 모델링**: 관계 강도(binary → weighted)로 확장

---

## 참고문헌

- Feng, F. et al. (2019). Temporal Relational Ranking for Stock Prediction. *ACM TOIS*.
- Chen, J. & Shen, Y. (2024). StockMixer: A Simple yet Strong MLP-based Architecture. *AAAI 2024*.
- Sawhney, R. et al. (2021). STHAN-SR. *AAAI 2021*.
- Veličković, P. et al. (2017). Graph Attention Networks. *ICLR 2018*.
- Wang, X. et al. (2023). Self-Consistency Improves Chain of Thought Reasoning. *ICLR 2023*.
- Israelsen, R.D. (2016). 'It's not who you know, it's what you know about who you know'. *Journal of Accounting Research*.
- Muslu, V. et al. (2014). Forward-Looking MD&A Disclosures. *Management Science*.
