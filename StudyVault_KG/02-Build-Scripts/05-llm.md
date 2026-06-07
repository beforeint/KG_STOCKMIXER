# 05. LLM KG 빌더 (GPT-4o-mini)

> 순서: 5/6 (02-Build-Scripts)  
> 파일: `build_kg/build_llm_kg.py`  
> 출력: `NASDAQ_llm_relation.npy`  shape=(1026,1026,R)

---

## 핵심 아이디어

> GPT-4o-mini에게 각 종목의 공급자/경쟁사/파트너 등 8가지 관계를 zero-shot으로 물어본다.  
> 시간 마스킹: "as of January 2012" → look-ahead bias 방지

---

## 8가지 관계 타입

```python
RELATIONS = [
    'is_Supplier_of',          # A가 B에 공급
    'is_Customer_of',          # A가 B의 고객 (B가 A에 공급)
    'is_Competitor_of',        # 직접 경쟁
    'is_Strategic_partner_of', # 전략적 파트너십
    'is_Subsidiary_of',        # A가 B의 자회사
    'is_Parent_of',            # A가 B의 모회사
    'is_JV_partner_of',        # 합작투자
    'is_Licensor_of',          # 라이선스 공급자
]
```

---

## V1 → V2 개선 과정

### V1 문제

GPT가 universe 밖의 ticker를 반환 → 매칭 손실 ~76.7%

```
GPT 응답: AAPL의 경쟁사 → ["GOOGL", "MSFT", "SAMSNG"]
ticker_idx에 없는 항목 → 전부 무시됨
```

### V2 해결

SYSTEM_PROMPT에 전체 NASDAQ ticker 목록 포함:

```python
TICKER_LIST_STR = ', '.join(tickers)  # ~5000 chars, ~1250 tokens

SYSTEM_PROMPT = (
    "You are a financial analyst... "
    f"Answer based ONLY on publicly available information as of January {year}. "
    "IMPORTANT: You must ONLY use ticker symbols from the following list:\n"
    f"{TICKER_LIST_STR}"
)
```

### 결과 비교

| | V1 | V2 |
|--|----|----|
| ticker 제약 | 없음 | SYSTEM_PROMPT에 전체 목록 |
| 매칭 손실 | ~76.7% | 최소화 |
| cache dir | `llm_cache/` | `llm_cache_v2/` |
| Low-Vol IC (gat) | +0.0203 (+43.9%) | **+0.0231 (+63.5%)** |

---

## 시간 마스킹 (Temporal Masking)

```python
TEMPORAL_YEAR = 2012  # 기본값

SYSTEM_PROMPT = (
    f"Answer based ONLY on publicly available information as of January {TEMPORAL_YEAR}. "
    f"Do not use any knowledge about events after December 31, {TEMPORAL_YEAR-1}."
)
```

학습 데이터 시작: 2013-01-02 → TEMPORAL_YEAR=2012는 학습 기간 전 상태 반영.

`--temporal_year` 인자로 다른 연도 지정 가능:
```bash
python build_llm_kg.py --v2 --temporal_year 2015
# → llm_cache_v2_2015/ 디렉토리에 캐시
```

---

## 실행 절차

```bash
export OPENAI_API_KEY="sk-..."

# Step 1: ticker별 LLM 호출 (배치 처리 가능)
python build_llm_kg.py --market NASDAQ --v2 --start 0 --end 256
python build_llm_kg.py --market NASDAQ --v2 --start 256 --end 512
python build_llm_kg.py --market NASDAQ --v2 --start 512 --end 769
python build_llm_kg.py --market NASDAQ --v2 --start 769 --end 1026

# Step 2: partial 파일 합치기 + pruning + 대칭화
python build_llm_kg.py --market NASDAQ --v2 --merge --kappa 1 --symmetry
```

---

## Pruning (kappa)

엣지가 kappa개 미만인 relation 채널 제거:

```python
def merge_and_prune(kappa=5):
    edge_count = (mat > 0).sum(axis=(0, 1))
    keep = edge_count >= kappa   # 채널별 유지 여부
    mat_pruned = mat[:, :, keep]
```

- `kappa=1`: 엣지 1개 이상인 relation만 유지 (느슨한 기준)
- `kappa=5`: 실용적 기준 (안정적인 relation만 남김)

---

## 캐싱 구조

```
llm_cache_v2/
├── AAPL.json        # {"is_Supplier_of": ["QCOM"], "is_Competitor_of": ["MSFT", ...], ...}
├── MSFT.json
├── partial_0_256.npy
├── partial_256_512.npy
├── ...
└── build_llm_kg.log
```

---

## SP500 결과

```bash
python build_llm_kg.py --market SP500 --v2 --start 0 --end 474
python build_llm_kg.py --market SP500 --v2 --merge --kappa 1 --symmetry
```

| 항목 | 값 |
|------|-----|
| Shape | (474, 474, 2) |
| Relations 유지 | is_Supplier_of, is_Competitor_of |
| 고립 노드 | 209/474 (44.1%) |
| Low-Vol IC | **-0.0062** (역효과) |

SP500 실패 원인: 44% 고립 → GAT 표현 불일치 → 성능 저하.
