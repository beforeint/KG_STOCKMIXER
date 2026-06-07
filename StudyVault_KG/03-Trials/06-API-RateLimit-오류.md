---
title: OpenAI API Rate Limit 오류
tags: [error, api, slurm, rate-limit]
created: 2026-06-04
---

# OpenAI API Rate Limit 오류

---

## 증상

```
Error code: 429 - Rate limit reached for gpt-4o-mini
Limit 10000, Used 10000, Requested 1.
→ slurmstepd: JOB CANCELLED DUE TO TIME LIMIT
→ DependencyNeverSatisfied
```

---

## 원인

5개 연도 동시 제출 (병렬) × 3 votes = API 요청 폭발

```
5년 × ~1026 ticker × 3 votes × ~3 calls = ~46,000 calls/day
RPD 한도: 10,000/day → 초과
```

---

## 해결

**순차 실행**: 앞 연도 완료 후 다음 연도 시작

```bash
# submit_llm_dynamic.sh 수정
jid13=$(sbatch run_llm_dynamic.sh --year 2013 ...)
jid14=$(sbatch --dependency=afterok:$jid13 run_llm_dynamic.sh --year 2014 ...)
jid15=$(sbatch --dependency=afterok:$jid14 run_llm_dynamic.sh --year 2015 ...)
...
```

각 연도 ~2,700 calls → 일 한도 내 수용 가능

---

## 추가: Time Limit 문제

`--time 10:00:00` → Rate limit retry로 인해 실제 10시간 초과
→ `--time 20:00:00`으로 증가

---

## 추가: Python 3.8 호환성

```
TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'
```

`stockmixer` conda 환경이 Python 3.8 — `dict | None` 문법은 3.10+

```python
# 수정 전
def _single_call(...) -> dict | None:
# 수정 후
def _single_call(...):  # type hint 제거
```

---

## 추가: 중간 체크포인트

Time limit에 끊기면 partial .npy가 저장 안 됨 (ticker.json 캐시는 살아있음)
→ `CHECKPOINT_INTERVAL = 200` 추가 (200 ticker마다 ckpt 저장)

---

## [[07-llm-dynamic]] — 전체 빌드 스크립트
