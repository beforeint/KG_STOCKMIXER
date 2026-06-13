---
title: LLM Dynamic KG (Self-Consistency 버전)
tags: [kg-construction, llm, dynamic-kg, self-consistency, hallucination]
created: 2026-06-04
---

# LLM Dynamic KG — build_llm_kg_dynamic.py

> 기존 `llm_v2`(제로샷 정적 KG)의 한계를 보완한 **연도별 동적 LLM KG**.
> 핵심 개선: evidence 필수 + Self-Consistency Voting + 양방향 교차검증

---

## 기존 llm_v2와의 차이

| | llm_v2 (기존) | llm_dynamic (신규) |
|---|---|---|
| 방식 | 1회 질의, temperature=0 | 3회 질의, temperature=0.5 |
| 근거 | 없음 | evidence 문장 필수 |
| 필터링 | 없음 | 3회 중 ≥2회 등장만 채택 |
| 교차검증 | 없음 | 양방향 확인 (Supplier↔Customer) |
| 연도 | 2013 스냅샷만 | 2013~2017 전부 |

---

## Self-Consistency Voting

> 근거 논문: Wang et al., ICLR 2023 — "Self-Consistency Improves Chain of Thought Reasoning"

```
AAPL 3번 질의 (temperature=0.5):
  1회차: INTC(Supplier), TSLA(Customer), MSFT(Competitor)
  2회차: INTC(Supplier), MSFT(Competitor), QCOM(Customer)
  3회차: INTC(Supplier), MSFT(Competitor), GOOGL(Competitor)

vote_counts:
  (Supplier, INTC)    = 3/3 → 채택 ✅
  (Competitor, MSFT)  = 3/3 → 채택 ✅
  (Customer, TSLA)    = 1/3 → 탈락 ✗  ← hallucination 제거
```

**핵심 원리**: hallucination은 샘플마다 달리 나와 다수결에서 탈락, 실제 관계는 일관되게 등장

---

## 양방향 교차검증 (Bidirectional Cross-Validation)

```python
BIDIR_PAIRS = {
    is_Supplier_of  ↔  is_Customer_of   # A→B Supplier면 B→A Customer 확인
    is_Competitor_of ↔ is_Competitor_of  # 대칭
    is_Subsidiary_of ↔ is_Parent_of
    is_JV_partner_of ↔ is_JV_partner_of
    is_Licensor_of   →  None  # 역방향 없음
}
```

merge 시 양방향 모두 확인된 엣지는 신뢰도 높음.
`--bidir_only` 플래그로 엄격 모드 가능.

---

## 캐시 파일 구조

```
llm_dynamic_cache_2014/
  AAPL.json        ← 최종 voted 결과 (build_matrix에서 사용)
  AAPL_votes.json  ← 3회 raw 응답 (디버깅용)
  AAON.json        ← 빈 칸 = 잘 알려지지 않은 소형주, 정상
  ckpt_2014_0_1026.npy  ← 중간 체크포인트 (200 ticker마다 저장)
```

---

## SLURM 실행

```bash
# 전체 제출 (순차 실행 - API RPD 10000/day 한도 분산)
bash submit_llm_dynamic.sh

# 수동 단일 연도
sbatch run_llm_dynamic.sh --year 2013 --start 0 --end 1026
sbatch run_llm_dynamic.sh --year 2013 --merge --symmetry
```

> [!warning] partition은 cpu1 사용
> LLM 빌드는 GPU 불필요. gpu1 쓰면 GPU 자리 경쟁으로 대기 길어짐.

---

## 출력

```
relation/llm_dynamic_{year}/NASDAQ_llm_dynamic_relation.npy
shape: (1026, 1026, 8)  — 8가지 관계 타입
```

완성 연도 (2026-06-04 기준): 2013 ✅, 2014 ✅, 2015 ✅, 2016 ✅, 2017 ✅

---

## 실험 결과 및 실패 원인

**GAT+LLMDynamic: IC = 0.0153 — baseline(0.0156) 미달**

### 실패 원인 분석

**1. Temporal masking 신뢰 불가**
```python
f"Do NOT use knowledge of events after December 31, {cutoff}."
```
GPT는 실제로 특정 연도까지의 지식을 격리하지 못함.  
프롬프트 수준 제약은 soft constraint — 이후 정보가 섞일 수 있음.

**2. 양방향 필터가 sparsity 심화**  
교차검증 통과 엣지가 너무 줄어들어 GAT가 참고할 연결이 부족.

**3. 파라메트릭 지식의 한계**  
GPT 내장 지식 = 학습 데이터에서 통계적으로 자주 나온 관계.  
뉴스처럼 "그 해 시장이 실제로 주목한 관계"가 아님.  
→ 주가 공동움직임과의 연관성이 NewsDynamic보다 낮음.

### NewsDynamic과 비교

| | NewsDynamic | LLMDynamic |
|--|-------------|------------|
| 정보 소스 | FMP 뉴스 (외부 텍스트) | GPT 내장 지식 (파라메트릭) |
| Look-ahead 방지 | 연도 뉴스만 수집 (완전) | 시간 마스킹 프롬프트 (불완전) |
| IC (test) | 0.0201 | 0.0153 |
| 결론 | 유의미한 개선 | baseline 미달 |

---

## [[04-Self-Consistency-Voting]] 참고
## [[03-실험결과-비교]] 에서 llm_dynamic 실험 결과 확인
## [[06-news]] 와 비교: 뉴스 기반 vs 파라메트릭 지식
