---
title: Self-Consistency Voting
tags: [llm, hallucination, self-consistency, sampling]
source: Wang et al., ICLR 2023
created: 2026-06-04
---

# Self-Consistency Voting

> Wang et al. (2023). *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR 2023.

---

## 핵심 아이디어

> [!important]
> temperature > 0으로 **여러 번 샘플링**하고 **다수결**로 최종 답을 결정.
> hallucination(확신 없는 오답)은 매번 다르게 나와 다수결에서 탈락.
> 실제 정보는 일관되게 나와 채택됨.

```
greedy decoding (temperature=0):  1번 → 그 결과만 사용
self-consistency (temperature=0.5): 3번 샘플링 → 다수결
```

---

## 우리 프로젝트 적용

```python
N_VOTES   = 3      # 샘플 수
VOTE_TEMP = 0.5    # diverse sampling
MIN_VOTES = 2      # 채택 기준: 3회 중 2회 이상
```

**실제 효과 예시 (ADSK 2013):**

| 관계 후보 | vote1 | vote2 | vote3 | 채택 |
|-----------|-------|-------|-------|------|
| ADBE Competitor | ✓ | ✗ | ✓ | ✅ 2/3 |
| SSYS Competitor | ✓ | ✗ | ✗ | ✗ 1/3 |
| MSFT Competitor | ✗ | ✓ | ✗ | ✗ 1/3 |

**TSLA hallucination 제거 예시 (AAPL 2014):**
- vote1에서만 "AAPL이 TSLA 배터리 기술 구매" → 1/3 → 탈락

---

## 비용

- API 호출 3배 증가
- 하지만 RPD(일일 요청 한도) 문제 주의
  - 5개 연도 동시 제출 시 10,000/day 한도 초과 → 순차 실행으로 해결

---

## 관련 개념

- [[07-llm-dynamic]] — 프로젝트 적용 코드
- [[06-API-RateLimit-오류]] — RPD 한도 초과 문제
