# 03. LLM KG 개선 과정

> 순서: 3/5 (03-Trials)

---

## LLM KG 개발 타임라인

```
V1 구축 → 매칭 손실 발견 → V2 개선 → SP500 확장 → SP500 실패 분석
```

---

## Trial-13: V1 — Universe 밖 ticker 반환 문제

**현상**: GPT가 NASDAQ universe 밖의 ticker를 반환 → 76.7% 매칭 손실

```
AAPL의 경쟁사로 GPT 응답: ["GOOGL", "MSFT", "SAMSNG", "005930.KS"]
ticker_idx에 없는 항목 → 전부 무시
실제 유효 엣지: 23.3%만 사용됨
```

**V2 해결**: SYSTEM_PROMPT에 전체 NASDAQ ticker 목록 포함

```python
# V1 SYSTEM_PROMPT
SYSTEM_PROMPT = (
    "You are a financial analyst specializing in US equity markets. "
    "Answer based ONLY on publicly available information as of January 2012."
)

# V2 SYSTEM_PROMPT (ticker 목록 추가)
TICKER_LIST_STR = ', '.join(tickers)  # 1026개 ticker, ~5000 chars
SYSTEM_PROMPT = (
    "You are a financial analyst specializing in US equity markets. "
    f"Answer based ONLY on publicly available information as of January {year}.\n\n"
    "IMPORTANT: You must ONLY use ticker symbols from the following list:\n"
    f"{TICKER_LIST_STR}"
)
```

---

## Trial-14: temporal_year 파라미터 추가

**배경**: V2 기본값은 2012 (학습 시작 전) → 하지만 다른 연도 KG가 필요할 수도 있음  
**추가 기능**: `--temporal_year` 인자

```python
# 코드 변경 (build_llm_kg.py)
parser.add_argument('--temporal_year', type=int, default=2012,
    help='Knowledge cutoff year (e.g. 2013, 2015). Default=2012')

# temporal_year별 캐시 분리
if args.temporal_year == 2012:
    _ACTIVE_CACHE = CACHE_DIR_V2       # 기존 경로 유지 (하위 호환)
else:
    yr = args.temporal_year
    _ACTIVE_CACHE = Path(__file__).parent / f'llm_cache_v2_{yr}'
    OUT_DIR = _DATA_ROOT / _market_arg / 'relation' / f'llm_v2_{yr}'
```

---

## Trial-15: symmetry 플래그 추가

**배경**: GPT 응답은 방향성 있음 (A→B 명시), 하지만 GNN에서 대칭 행렬이 더 안정적  
**추가 기능**: `--symmetry` 플래그

```python
# merge 후 대칭화
if args.symmetry:
    for i in range(mat_pruned.shape[2]):
        mat_pruned[:,:,i] = np.maximum(mat_pruned[:,:,i], mat_pruned[:,:,i].T)
    out_path = OUT_DIR / f'{_market_arg}_llm_relation.npy'
    np.save(out_path, mat_pruned)
```

---

## Trial-16: kappa pruning

**배경**: 8개 relation 채널 중 일부는 엣지가 거의 없음  
**해결**: kappa 임계값 이하 채널 제거

```python
def merge_and_prune(kappa=5):
    edge_count = (mat > 0).sum(axis=(0, 1))
    keep = edge_count >= kappa
    
    # 예시 로그
    # is_Supplier_of:       1234 edges  → keep
    # is_JV_partner_of:       2 edges  → drop (kappa=5면)
    # is_Licensor_of:         1 edge   → drop
    
    mat_pruned = mat[:, :, keep]
```

**NASDAQ V2 결과** (kappa=1):
- 유지된 relation: 수 개 (실험 후 meta.json 확인)

---

## Trial-17: SP500 LLM KG 실패 분석

**결과**: Low-Vol IC = -0.0062 (역효과, baseline 대비 -273%)  
**원인 분석**:

1. **고립 노드 44.1%**: 209/474 종목이 LLM KG에서 엣지 없음
2. **GAT 표현 불일치**: 고립 노드는 GAT에서 self-attention만 → 표현 부정확
3. **SP500 유명 기업**: GPT가 잘 아는 기업들이지만 universe 밖 ticker 반환 여전히 있음

**교훈**: 고립 노드 비율이 높으면 LLM KG는 오히려 해가 된다.
