---
title: News KG 빌더 (NewsDynamic)
tags: [kg-construction, news, dynamic-kg, gpt, fmp, focal]
created: 2026-06-04
updated: 2026-06-08
---

# 06. NewsDynamic KG — build_news_kg.py

> FMP 뉴스 기사에서 GPT로 기업 간 관계를 추출해 연도별 KG 스냅샷 생성.  
> 실험 결과 전체 KG 중 가장 강한 유의성 (Low Vol IC p=0.0004)

---

## 핵심 아이디어

- 뉴스 기사에서 **그 해에 실제로 언급된 관계만** 추출 → look-ahead 없음
- focal ticker별 수집 → 400+ 종목 × 연도별 독립 스냅샷

---

## Focal의 의미

```
focal = 지금 중심으로 보는 종목 (예: AAPL)

AAPL 기사: "Apple's chip supplier TSMC said..."
→ GPT: TSMC는 AAPL의 Supplier
→ mat[AAPL_idx, TSMC_idx, Supplier] = 1
```

노드(종목)는 고정, **엣지만** GPT가 뽑음.  
1026개 종목이 순서대로 focal이 되어 자기 기사를 수집.

---

## FMP API

```python
requests.get(FMP_NEWS_URL, params={
    'symbols': ticker,        # focal ticker
    'from': f'{year}-01-01',
    'to':   f'{year}-12-31',
    'limit': 50,
    'apikey': FMP_KEY,
})
```

Financial Modeling Prep — 종목별 뉴스 제공 유료 API.  
연도 범위를 지정하므로 해당 연도 뉴스만 수집 → look-ahead 자동 방지.

---

## 6가지 관계 타입

```python
RELATIONS = ['Supplier', 'Customer', 'Competitor', 'Partner', 'Subsidiary', 'Parent']
```

방향 있음: `Supplier(focal→other)` = focal이 other로부터 공급받음.

---

## 파이프라인 전체 흐름

```
[AAPL 2016 뉴스 50개]
    ↓ 기사마다
[GPT-4o-mini, temperature=0, json_object 강제]
"기사 본문에서만 관계 추출, 외부 지식 금지"
    ↓
mat[AAPL, TSM, Supplier] = 1
    ↓ 1026개 종목 반복 → partial 저장
[merge: 부분 행렬 누적]
    ↓ 이진화
최종 (1026, 1026, 6) 행렬
```

---

## Step 4: Merge & Binarization (수식)

**누적합**
$$\tilde{M}_{ij}^{r} = \sum_{k} M_{ij}^{(k),r} \;\in\; \{0,1,2,\ldots\}$$

**이진화**
$$A_{ij}^{r} = \mathbf{1}[\tilde{M}_{ij}^{r} > 0] \;\in\; \{0,1\}$$

```python
mat = np.zeros((N, N, R))
for p in sorted(cache.glob(f'partial_{year}_*.npy')):
    mat += np.load(p)           # 누적
mat = (mat > 0).astype(float)  # 이진화 → 빈도 정보 소실
```

> AAPL→TSM 관계가 5개 기사에서 나왔든 1개에서 나왔든 최종값은 1

---

## 두 번의 정보 손실

| 단계 | 위치 | 소실 정보 |
|------|------|-----------|
| 1차 | `merge()` | relation 내 빈도 (count → 0/1) |
| 2차 | `load_data.py` | relation 간 타입 구분 (6채널 → sum → 1채널) |

load_data.py에서:
$$C_{ij} = \mathbf{1}\!\left[\sum_{r=0}^{5} A_{ij}^{r} > 0\right]$$

GAT는 결국 "어떤 relation type으로든 연결된 적 있는가"만 봄.

---

## "KG 구조 vs 뉴스 신호" Confound

NewsDynamic이 잘 나온 이유 두 가지 해석:

**해석 A — KG 구조가 기여**  
Supplier/Customer 관계로 연결된 종목이 실제로 공동 움직임 → GAT가 포착

**해석 B — 뉴스 co-mention 자체가 신호**  
투자자들이 같은 뉴스를 보고 두 종목을 함께 거래 → 가격이 실제로 연동  
→ 관계 타입은 부수적

HGAT(관계 타입 구분)가 GAT보다 낮은 성능 → **해석 B를 지지**  
타입 구분이 오히려 손해 = 타입 정보 자체의 가치가 크지 않다는 신호

논문 표현:
> "The performance gap between GAT and HGAT suggests that the predictive signal
>  may stem from news co-mention rather than typed relational structure."

---

## Wikidata와 비교

| | Wikidata | NewsDynamic |
|--|----------|-------------|
| 구성 시점 | ~2019 단일 스냅샷 | 연도별 독립 스냅샷 |
| Look-ahead | 잠재적 있음 | 없음 (해당 연도 뉴스만) |
| 관계 유형 | 장기 안정적 구조 | 그 해 시장이 주목한 관계 |
| 평균 IC | 0.0203 | 0.0201 |

> 성능이 비슷하다는 것은 look-ahead가 주원인이 아닐 수 있음을 시사.  
> 혹은 Wikidata의 장기 안정 관계도 충분히 예측력이 있다는 뜻.

---

## FinDKG와의 차이

| | FinDKG | 본 연구 |
|--|--------|---------|
| LLM 역할 | KG 구축 (공통) | KG 구축 (공통) |
| 다음 단계 | KGTransformer → KG 링크 예측 | GAT → 주가 수익률 예측 |
| KG의 역할 | 예측 **대상** | 예측을 위한 **보조 구조** |
| 연구 문제 | "KG가 어떻게 변하는가" | "어떤 KG가 주가 예측에 유용한가" |

---

## 실행

```bash
for year in 2013 2014 2015 2016 2017; do
    python build_news_kg.py --year $year --start 0 --end 1026
    python build_news_kg.py --year $year --merge --symmetry
done
```

---

## 관련 문서
## [[03-실험결과-비교]] — IC 0.0201, Low Vol p=0.0004
## [[05-HGAT-vs-GAT]] — relation type 구분의 한계
## [[07-llm-dynamic]] — 비교: 파라메트릭 지식 기반 동적 KG
