# 01. Jaccard 유사도 — Institutional KG의 관계 강도 계산

> 순서: 1/3 (04-Concepts)

---

## 정의

두 집합 A, B의 Jaccard 유사도:

```
J(A, B) = |A ∩ B| / |A ∪ B|
```

- 범위: [0, 1]
- 0 = 완전히 다른 집합
- 1 = 완전히 동일한 집합

---

## Institutional KG에서의 적용

```python
# A = AAPL을 보유한 기관 집합
# B = MSFT를 보유한 기관 집합

A = {"Vanguard Group", "BlackRock", "State Street"}
B = {"Vanguard Group", "BlackRock", "Fidelity"}

intersection = A & B  # {"Vanguard Group", "BlackRock"}  → 2개
union        = A | B  # {"Vanguard Group", "BlackRock", "State Street", "Fidelity"}  → 4개

Jaccard = 2 / 4 = 0.50
```

---

## Board KG에서의 다른 방식

Board KG는 Jaccard가 아니라 **min 기준 공유 비율** 사용:

```python
w = len(shared) / min(len(board_i), len(board_j))
```

- Jaccard와 달리, 더 작은 집합 기준으로 정규화
- 한쪽 이사회가 매우 작을 때 공유 비율을 더 크게 봄

---

## MIN_OVERLAP_RATIO = 0.01의 의미

Institutional KG에서 Jaccard ≥ 0.01 이면 엣지 연결:

```
100명 기관 중 1명만 공통 → J = 1/199 ≈ 0.005 → 엣지 없음
100명 기관 중 2명 공통 → J = 2/198 ≈ 0.010 → 엣지 연결 (경계)
```

임계값 0.01은 매우 관대한 기준 → density 29%의 원인

---

## 대안: Overlap Coefficient (Szymkiewicz-Simpson)

```
OC(A, B) = |A ∩ B| / min(|A|, |B|)
```

한쪽 집합이 다른 쪽의 부분집합이면 OC=1 (Jaccard=0.5 이하).  
build_institutional_kg.py 코드 주석의 `min(|A|, |B|)` 설명과 다르게  
실제 코드는 `len(union)`을 분모로 사용 → 정확히 Jaccard 방식.
