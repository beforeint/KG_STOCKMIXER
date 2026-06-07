#!/usr/bin/env python3
"""
LLM-based KG Construction (GPT-4o-mini, Zero-shot)
시간 마스킹: "as of January 2012" — look-ahead bias 방지

출력: {MARKET}_llm_relation.npy  shape=(N, N, R_final)

실행:
    export OPENAI_API_KEY="sk-..."
    python build_llm_kg.py [--market NASDAQ|SP500] [--dry_run] [--start 0] [--end N]
"""
import os, sys, json, time, argparse, logging
import numpy as np
from pathlib import Path
from openai import OpenAI

# ── market 결정 (argparse 전에 sys.argv로 미리 파악) ────────────
_market_arg = 'NASDAQ'
for i, a in enumerate(sys.argv):
    if a == '--market' and i + 1 < len(sys.argv):
        _market_arg = sys.argv[i + 1]

# ── 경로 (market별 분기) ────────────────────────────────────────
_DATA_ROOT = Path('/gpfs/home1/pz29075/Capstone/StockMixer/dataset')

if _market_arg == 'SP500':
    TICKER_FILE = str(_DATA_ROOT / 'SP500' / 'tickers.txt')
    OUT_DIR     = _DATA_ROOT / 'SP500' / 'relation' / 'llm'
    CACHE_DIR   = Path(__file__).parent / 'llm_cache_sp500'
    CACHE_DIR_V2 = Path(__file__).parent / 'llm_cache_sp500_v2'
else:  # NASDAQ
    TICKER_FILE = ('/gpfs/home1/pz29075/Capstone/StockMixer/'
                   'Temporal_Relational_Stock_Ranking/data/'
                   'NASDAQ_tickers_qualify_dr-0.98_min-5_smooth.csv')
    OUT_DIR     = _DATA_ROOT / 'NASDAQ' / 'relation' / 'llm'
    CACHE_DIR   = Path(__file__).parent / 'llm_cache'
    CACHE_DIR_V2 = Path(__file__).parent / 'llm_cache_v2'

OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CACHE_DIR_V2.mkdir(exist_ok=True)

# ── Relation 정의 (S&P Global taxonomy 기반, 논문 참고) ──────────
RELATIONS = [
    'is_Supplier_of',       # A가 B에 공급
    'is_Customer_of',       # A가 B의 고객
    'is_Competitor_of',     # 직접 경쟁
    'is_Strategic_partner_of',  # 전략적 파트너십
    'is_Subsidiary_of',     # A가 B의 자회사
    'is_Parent_of',         # A가 B의 모회사
    'is_JV_partner_of',     # 합작투자
    'is_Licensor_of',       # 라이선스 공급자
]
R = len(RELATIONS)
REL_IDX = {r: i for i, r in enumerate(RELATIONS)}

# ── NASDAQ ticker 로드 ───────────────────────────────────────────
tickers = open(TICKER_FILE).read().strip().split('\n')
N = len(tickers)
ticker_idx = {t: i for i, t in enumerate(tickers)}

# ── V2 프롬프트: NASDAQ-1026 ticker 목록을 system prompt에 포함 ──
# V1 문제: GPT가 NYSE 종목 등 universe 밖 ticker 반환 → 76.7% 매칭 손실
# V2 해결: valid ticker set을 명시 → 매칭 손실 최소화
TICKER_LIST_STR = ', '.join(tickers)  # 약 5000 chars, ~1250 tokens

#%% ── 동적 프롬프트 빌더 (temporal_year에 따라 변경) ─────────────────
# TEMPORAL_YEAR / SYSTEM_PROMPT 는 argparse 이후 __main__ 블록에서 설정됨
TEMPORAL_YEAR = 2012   # 기본값; --temporal_year 인자로 덮어씌워짐
SYSTEM_PROMPT = None   # 아래 build_prompts()로 초기화


def build_prompts(year: int):
    """TEMPORAL_YEAR에 맞게 모듈 전역 프롬프트 변수를 갱신한다."""
    global TEMPORAL_YEAR, SYSTEM_PROMPT
    TEMPORAL_YEAR = year
    t_str = f"January {year}"
    c_str = f"December 31, {year - 1}"
    SYSTEM_PROMPT = (
        "You are a financial analyst specializing in US equity markets. "
        f"Answer based ONLY on publicly available information as of {t_str}. "
        "Do not use any knowledge about events, mergers, acquisitions, or "
        f"business changes that occurred after {c_str}.\n\n"
        "IMPORTANT: You must ONLY use ticker symbols from the following list of "
        f"NASDAQ-listed stocks (as of {t_str}). Do not return any ticker not in this list:\n"
        f"{TICKER_LIST_STR}"
    )


build_prompts(TEMPORAL_YEAR)   # 기본값(2012)으로 초기화


def make_user_prompt(ticker: str) -> str:
    t_str = f"January {TEMPORAL_YEAR}"
    return f"""For the NASDAQ-listed company with ticker symbol "{ticker}" (as of {t_str}):

Identify up to 5 companies for EACH of the following relationship types.
Only include relationships that existed as of {t_str}.
ONLY use ticker symbols from the valid list provided in the system prompt.

Relationships to identify:
- is_Supplier_of: companies that "{ticker}" supplied products/services to
- is_Customer_of: companies that "{ticker}" purchased from (its suppliers)
- is_Competitor_of: direct competitors of "{ticker}"
- is_Strategic_partner_of: strategic alliance/partnership partners
- is_Subsidiary_of: parent companies of "{ticker}" (if any)
- is_Parent_of: subsidiaries that "{ticker}" owned
- is_JV_partner_of: joint venture partners
- is_Licensor_of: companies "{ticker}" licensed technology/IP to

Return ONLY valid JSON in this exact format (no explanation):
{{
  "is_Supplier_of": ["TICK1", "TICK2"],
  "is_Customer_of": ["TICK3"],
  "is_Competitor_of": ["TICK4", "TICK5"],
  "is_Strategic_partner_of": [],
  "is_Subsidiary_of": [],
  "is_Parent_of": [],
  "is_JV_partner_of": [],
  "is_Licensor_of": []
}}

If "{ticker}" was not a well-known NASDAQ company or you are uncertain, return empty lists."""


# ── API 호출 ────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[logging.FileHandler(CACHE_DIR/'build_llm_kg.log'),
                               logging.StreamHandler()])
log = logging.getLogger(__name__)

client = OpenAI()   # OPENAI_API_KEY 환경변수 자동 사용

# 런타임에 --v2 플래그로 캐시 디렉토리 전환
_ACTIVE_CACHE = CACHE_DIR   # build_matrix/query_llm에서 참조


def query_llm(ticker: str, dry_run=False) -> dict:
    """GPT-4o-mini 호출. 캐시 있으면 재사용."""
    cache_path = _ACTIVE_CACHE / f'{ticker}.json'
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    if dry_run:
        dummy = {r: [] for r in RELATIONS}
        dummy['is_Competitor_of'] = ['AAPL']   # 테스트용
        return dummy

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user',   'content': make_user_prompt(ticker)},
                ],
                temperature=0.0,
                max_tokens=512,
                response_format={'type': 'json_object'},
            )
            text = resp.choices[0].message.content
            result = json.loads(text)
            # key 정규화: 응답이 다른 key를 쓸 수 있음
            parsed = {r: [] for r in RELATIONS}
            for key, val in result.items():
                for rel in RELATIONS:
                    if rel.lower() in key.lower() or key.lower() in rel.lower():
                        parsed[rel] = [t.strip().upper() for t in (val or [])
                                       if isinstance(t, str)]
                        break
            cache_path.write_text(json.dumps(parsed, indent=2))
            return parsed

        except Exception as e:
            log.warning(f'{ticker} attempt {attempt+1} failed: {e}')
            time.sleep(2 ** attempt)

    log.error(f'{ticker}: all attempts failed, using empty')
    empty = {r: [] for r in RELATIONS}
    cache_path.write_text(json.dumps(empty, indent=2))
    return empty


# ── 행렬 구축 ────────────────────────────────────────────────────
def build_matrix(start: int, end: int, dry_run=False):
    """start~end 범위 ticker 처리 → partial npy 저장."""
    partial_path = _ACTIVE_CACHE / f'partial_{start}_{end}.npy'
    if partial_path.exists():
        log.info(f'Partial {start}-{end} already exists, skipping.')
        return

    mat = np.zeros((N, N, R), dtype=np.float32)

    for i in range(start, end):
        ticker = tickers[i]
        result = query_llm(ticker, dry_run=dry_run)
        log.info(f'[{i+1}/{end}] {ticker}: '
                 + ', '.join(f'{r}:{len(v)}' for r, v in result.items() if v))

        for rel, targets in result.items():
            if rel not in REL_IDX:
                continue
            r_idx = REL_IDX[rel]
            for t in targets:
                if t in ticker_idx:
                    j = ticker_idx[t]
                    if i != j:
                        mat[i, j, r_idx] = 1.0

        # Rate limit 방지: 20 req/min 기준
        if not dry_run:
            time.sleep(3.5)

    np.save(partial_path, mat)
    log.info(f'Saved partial {start}-{end}')


def merge_and_prune(kappa=5):
    """partial 파일들 합쳐서 pruning 후 최종 저장."""
    mat = np.zeros((N, N, R), dtype=np.float32)
    for p in sorted(_ACTIVE_CACHE.glob('partial_*.npy')):
        mat += np.load(p)

    # 엣지가 kappa개 미만인 relation 제거
    edge_count = (mat > 0).sum(axis=(0, 1))
    keep = edge_count >= kappa
    log.info(f'Relation edge counts: {dict(zip(RELATIONS, edge_count))}')
    log.info(f'Keeping {keep.sum()}/{R} relations (kappa={kappa})')

    mat_pruned = mat[:, :, keep]
    rel_kept   = [RELATIONS[i] for i in range(R) if keep[i]]

    out_path = OUT_DIR / f'{_market_arg}_llm_relation.npy'
    np.save(out_path, mat_pruned)

    meta = {'relations': rel_kept, 'shape': list(mat_pruned.shape),
            'kappa': kappa, 'edge_counts': dict(zip(RELATIONS, edge_count.tolist()))}
    mkt = _market_arg
    (OUT_DIR / f'{mkt}_llm_relation_meta.json').write_text(json.dumps(meta, indent=2))

    log.info(f'Saved: {out_path}  shape={mat_pruned.shape}')
    log.info(f'Relations kept: {rel_kept}')
    print(f"\n[config.py에 추가]\n'llm': 'relation/llm/{mkt}_llm_relation.npy',")
    return mat_pruned, rel_kept


# ── CLI ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--market',  type=str, default='NASDAQ',
                        choices=['NASDAQ', 'SP500'])
    parser.add_argument('--dry_run',  action='store_true')
    parser.add_argument('--start',   type=int, default=0)
    parser.add_argument('--end',     type=int, default=N)
    parser.add_argument('--merge',   action='store_true')
    parser.add_argument('--kappa',   type=int, default=1)
    parser.add_argument('--v2',           action='store_true')
    parser.add_argument('--symmetry',     action='store_true')
    parser.add_argument('--temporal_year', type=int, default=2012,
                        help='Knowledge cutoff year for LLM prompt (e.g. 2013, 2015, 2017). '
                             'Default 2012 keeps existing V2 behaviour.')
    args = parser.parse_args()

    # ── temporal 프롬프트 설정 ───────────────────────────────────────
    build_prompts(args.temporal_year)
    log.info(f'=== temporal_year={args.temporal_year} (cutoff: Dec {args.temporal_year-1}) ===')

    if args.v2:
        # temporal_year=2012(기본)이면 기존 경로 유지 → 하위 호환
        if args.temporal_year == 2012:
            _ACTIVE_CACHE = CACHE_DIR_V2
            OUT_DIR       = Path(str(OUT_DIR).replace('/llm', '/llm_v2'))
        else:
            yr = args.temporal_year
            _ACTIVE_CACHE = Path(__file__).parent / f'llm_cache_v2_{yr}'
            OUT_DIR       = _DATA_ROOT / _market_arg / 'relation' / f'llm_v2_{yr}'
        _ACTIVE_CACHE.mkdir(exist_ok=True)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        log.info(f'=== V2 mode: ticker list in prompt, cache={_ACTIVE_CACHE} ===')

    if args.merge:
        mat_pruned, rel_kept = merge_and_prune(kappa=args.kappa)
        if args.symmetry:
            for i in range(mat_pruned.shape[2]):
                mat_pruned[:,:,i] = np.maximum(mat_pruned[:,:,i], mat_pruned[:,:,i].T)
            out_path = OUT_DIR / f'{_market_arg}_llm_relation.npy'
            np.save(out_path, mat_pruned)
            log.info(f'Symmetrized and saved: {out_path}')
    else:
        log.info(f'Building LLM KG [{_market_arg}]: tickers {args.start}~{args.end}, dry_run={args.dry_run}')
        build_matrix(args.start, args.end, dry_run=args.dry_run)
        log.info('Done. Run with --merge to finalize.')
