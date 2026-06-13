---
title: Temporal Wikidata KG (아이디어 — 미구현)
tags: [wikidata, temporal-kg, look-ahead, sparql, future-work]
created: 2026-06-08
---

# 08. Temporal Wikidata KG

> 기존 정적 Wikidata의 look-ahead 문제를 해결하기 위한 연도별 동적 구성 아이디어.  
> 현재 미구현 — 논문 limitation 또는 future work로 언급 예정.

---

## 기존 Wikidata의 문제

RSR 원본 Wikidata KG는 ~2019년 단일 스냅샷.  
테스트 기간(2013~2017)에 아직 형성되지 않은 관계가 포함될 수 있음.

```
예: 2015년에 A사가 B사 인수
→ 2019년 Wikidata에 날짜 없이 자회사 관계 등록
→ 2013 KG에도 그 관계가 포함됨 → look-ahead
```

---

## 아이디어: Delta Update 방식

```
2013 KG: startDate ≤ 2013 인 관계 전부
2014 KG = 2013 KG
         + startDate = 2014 인 신규 관계
         - endDate   = 2013 인 종료 관계
2015 KG = 2014 KG + ...
```

연도별로 변한 것만 업데이트 → 진정한 동적 KG.

---

## SPARQL 구현 방법

```sparql
SELECT ?src ?rel ?tgt WHERE {
  ?src p:P127 ?stmt .
  ?stmt ps:P127 ?tgt .
  OPTIONAL { ?stmt pq:P580 ?start }
  OPTIONAL { ?stmt pq:P582 ?end }
  FILTER(!BOUND(?start) || ?start <= "2015-12-31"^^xsd:date)
  FILTER(!BOUND(?end)   || ?end   > "2015-01-01"^^xsd:date)
}
```

`P127` = owned by, `P580` = start time, `P582` = end time

---

## 핵심 한계: 시간 한정자 커버리지

Wikidata 관계의 **70~80%에 P580/P582가 없음**.

날짜 없는 관계 처리 방법:

| 방법 | 설명 | 문제 |
|------|------|------|
| **포함** (현실적) | 날짜 없으면 항상 존재했다고 가정 | look-ahead 완전 제거 불가 |
| **제외** | 날짜 있는 것만 사용 | KG가 너무 sparse |
| **덤프 파싱** | 연도별 Wikidata 덤프에서 그 시점 존재한 것만 | 수십 GB, 구현 복잡 |

---

## 왜 기존 연구가 이걸 안 했나

1. RSR(2019)의 목표가 "그래프 구조 활용 가능성 증명"이었음 → 정적으로 충분
2. 업종 분류, 모회사-자회사 같은 장기 안정 관계는 연간 변화율 < 5%
3. P580/P582 커버리지가 낮아 연도별 필터의 실익이 제한적
4. Wikidata 덤프 파싱 비용이 큼

---

## 논문 활용 방법

현재 Wikidata 결과(IC=0.0203)와 NewsDynamic(IC=0.0201)이 비슷하다는 점을 이용:

> "The comparable performance between static Wikidata and NewsDynamic suggests
>  that look-ahead bias in Wikidata may not be the primary driver of performance.
>  Nevertheless, a temporally-filtered Wikidata KG remains an important direction
>  for future work to fully isolate the effect of relational structure."

---

## [[03-실험결과-비교]] — Wikidata IC=0.0203, NewsDynamic IC=0.0201
## [[06-news]] — NewsDynamic: 뉴스 기반으로 look-ahead 자동 방지
