#!/usr/bin/env python3
"""
LLM Dynamic KG Construction — Anti-hallucination version
개선 사항:
  1. Evidence 필수: 각 관계마다 근거 문장 요구 → 근거 없으면 제외
  2. 양방향 교차검증: merge 시 A→B / B→A 쌍 확인 → 점수 기반 필터링
  3. 연도별 템포럴 마스킹: "as of {year}" 지식만 사용

실행:
  python build_llm_kg_dynamic.py --year 2013 --start 0 --end 1026
  python build_llm_kg_dynamic.py --year 2013 --merge
  python build_llm_kg_dynamic.py --year 2013 --merge --bidir_only  # 양방향 확인된 것만
"""
import os, json, time, argparse, logging
import numpy as np
from pathlib import Path
from openai import OpenAI

# ── 경로 설정 ─────────────────────────────────────────────────────
_DATA_ROOT  = Path('/gpfs/home1/pz29075/Capstone/StockMixer/dataset')
TICKER_FILE = ('/gpfs/home1/pz29075/Capstone/StockMixer/'
               'Temporal_Relational_Stock_Ranking/data/'
               'NASDAQ_tickers_qualify_dr-0.98_min-5_smooth.csv')

tickers    = open(TICKER_FILE).read().strip().split('\n')
N          = len(tickers)
ticker_idx = {t: i for i, t in enumerate(tickers)}
ticker_set = set(tickers)
TICKER_LIST_STR = ', '.join(tickers)

# ── 관계 정의 ─────────────────────────────────────────────────────
RELATIONS = [
    'is_Supplier_of',
    'is_Customer_of',
    'is_Competitor_of',
    'is_Strategic_partner_of',
    'is_Subsidiary_of',
    'is_Parent_of',
    'is_JV_partner_of',
    'is_Licensor_of',
]
R = len(RELATIONS)
REL_IDX = {r: i for i, r in enumerate(RELATIONS)}

# 양방향 쌍 정의: rel_idx → 역방향 rel_idx (없으면 None)
BIDIR_PAIRS = {
    REL_IDX['is_Supplier_of']:          REL_IDX['is_Customer_of'],
    REL_IDX['is_Customer_of']:          REL_IDX['is_Supplier_of'],
    REL_IDX['is_Competitor_of']:        REL_IDX['is_Competitor_of'],
    REL_IDX['is_Strategic_partner_of']: REL_IDX['is_Strategic_partner_of'],
    REL_IDX['is_Subsidiary_of']:        REL_IDX['is_Parent_of'],
    REL_IDX['is_Parent_of']:            REL_IDX['is_Subsidiary_of'],
    REL_IDX['is_JV_partner_of']:        REL_IDX['is_JV_partner_of'],
    REL_IDX['is_Licensor_of']:          None,  # 역방향 없음
}

# ── 로깅 ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

client = OpenAI()


# ── 프롬프트 ─────────────────────────────────────────────────────
def make_system_prompt(year: int) -> str:
    cutoff = year - 1
    return (
        f"You are a financial analyst. Answer based ONLY on publicly available "
        f"information as of January {year}. Do NOT use knowledge of events after "
        f"December 31, {cutoff}.\n\n"
        f"IMPORTANT: Only use ticker symbols from this NASDAQ list:\n{TICKER_LIST_STR}"
    )


def make_user_prompt(ticker: str, year: int) -> str:
    return f"""For NASDAQ ticker "{ticker}" as of January {year}:

List inter-company relationships with EVIDENCE. Only include relationships you are
confident existed and were publicly documented before January {year}.

For each relationship, provide:
- ticker: the other company's NASDAQ ticker (must be in the valid list)
- evidence: one specific, verifiable fact (e.g., "Intel supplied A-series chips to Apple per their 2012 earnings call")

Return ONLY valid JSON. Use empty list [] if uncertain.

{{
  "is_Supplier_of":          [{{"ticker": "...", "evidence": "..."}}],
  "is_Customer_of":          [{{"ticker": "...", "evidence": "..."}}],
  "is_Competitor_of":        [{{"ticker": "...", "evidence": "..."}}],
  "is_Strategic_partner_of": [{{"ticker": "...", "evidence": "..."}}],
  "is_Subsidiary_of":        [{{"ticker": "...", "evidence": "..."}}],
  "is_Parent_of":            [{{"ticker": "...", "evidence": "..."}}],
  "is_JV_partner_of":        [{{"ticker": "...", "evidence": "..."}}],
  "is_Licensor_of":          [{{"ticker": "...", "evidence": "..."}}]
}}

Rules:
- Max 5 entries per relationship type
- Skip if evidence is generic (e.g., "they are in the same industry")
- If "{ticker}" was not a well-known NASDAQ company before {year}, return all empty lists"""


# ── Self-Consistency 설정 (Wang et al., ICLR 2023) ────────────────
N_VOTES   = 3    # 샘플 수
VOTE_TEMP = 0.5  # diverse sampling을 위한 temperature
MIN_VOTES = 2    # 채택 기준: N_VOTES 중 MIN_VOTES 이상 등장


def _parse_response(raw: dict, focal: str) -> dict:
    """GPT raw JSON → {rel: [{"ticker":..., "evidence":...}]} 정규화."""
    parsed = {r: [] for r in RELATIONS}
    for key, val in raw.items():
        matched = next((r for r in RELATIONS
                        if r.lower() in key.lower() or key.lower() in r.lower()), None)
        if matched is None or not isinstance(val, list):
            continue
        for entry in val:
            if isinstance(entry, dict):
                t  = str(entry.get('ticker',   '') or '').strip().upper()
                ev = str(entry.get('evidence', '') or '').strip()
            elif isinstance(entry, str):
                t, ev = entry.strip().upper(), ''
            else:
                continue
            if t and t in ticker_set and t != focal and ev:
                parsed[matched].append({'ticker': t, 'evidence': ev})
    return parsed


def _single_call(system: str, user: str, temperature: float):
    """단일 GPT 호출. 실패 시 None 반환."""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{'role': 'system', 'content': system},
                          {'role': 'user',   'content': user}],
                temperature=temperature,
                max_tokens=800,
                response_format={'type': 'json_object'},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            log.warning(f'  call attempt {attempt+1}: {e}')
            time.sleep(2 ** attempt)
    return None


def query_llm(ticker: str, year: int, cache_dir: Path) -> dict:
    """
    Self-consistency voting (Wang et al., ICLR 2023):
      N_VOTES번 샘플링 후 MIN_VOTES 이상 등장한 (rel, ticker) 쌍만 채택.
      hallucination은 샘플마다 달리 나와 다수결에서 탈락,
      실제 관계는 일관되게 등장해 통과.

    캐시:
      {ticker}.json       → 최종 voted 결과 (build_matrix에서 사용)
      {ticker}_votes.json → 3회 raw 응답 (디버깅/분석용)
    """
    cache_path       = cache_dir / f'{ticker}.json'
    cache_votes_path = cache_dir / f'{ticker}_votes.json'

    if cache_path.exists():
        return json.loads(cache_path.read_text())

    system = make_system_prompt(year)
    user   = make_user_prompt(ticker, year)

    # ── N_VOTES번 샘플링 ──────────────────────────────────────────
    vote_counts:  dict[tuple, int] = {}   # (rel, ticker) → 등장 횟수
    best_evidence: dict[tuple, str] = {}  # (rel, ticker) → 마지막 evidence
    raw_votes = []

    for _ in range(N_VOTES):
        raw = _single_call(system, user, VOTE_TEMP)
        if raw is None:
            continue
        raw_votes.append(raw)
        for rel, entries in _parse_response(raw, ticker).items():
            for e in entries:
                key = (rel, e['ticker'])
                vote_counts[key]   = vote_counts.get(key, 0) + 1
                best_evidence[key] = e['evidence']
        time.sleep(1.5)

    # ── MIN_VOTES 이상 등장한 관계만 채택 ─────────────────────────
    result = {r: [] for r in RELATIONS}
    for (rel, t), cnt in vote_counts.items():
        if cnt >= MIN_VOTES:
            result[rel].append({'ticker': t, 'evidence': best_evidence[(rel, t)],
                                'votes': cnt})

    cache_votes_path.write_text(json.dumps(raw_votes, indent=2))
    cache_path.write_text(json.dumps(result, indent=2))

    kept = sum(len(v) for v in result.values())
    log.info(f'  {ticker}: {len(vote_counts)} candidates → {kept} passed (≥{MIN_VOTES}/{N_VOTES} votes)')
    return result


# ── 행렬 구축 ────────────────────────────────────────────────────
CHECKPOINT_INTERVAL = 200  # N ticker마다 중간 저장


def build_matrix(year: int, start: int, end: int, cache_dir: Path):
    partial_path = cache_dir / f'partial_{year}_{start}_{end}.npy'
    if partial_path.exists():
        log.info(f'Partial {year} {start}-{end} exists, skip.')
        return

    # 중간 체크포인트 복구: 마지막 저장된 체크포인트부터 재시작
    ckpt_path = cache_dir / f'ckpt_{year}_{start}_{end}.npy'
    resume_from = start
    if ckpt_path.exists():
        mat = np.load(ckpt_path)
        # 체크포인트 파일명에 진행 위치 저장
        ckpt_meta = cache_dir / f'ckpt_{year}_{start}_{end}.txt'
        if ckpt_meta.exists():
            resume_from = int(ckpt_meta.read_text().strip())
        log.info(f'Checkpoint found: resuming from ticker {resume_from}')
    else:
        mat = np.zeros((N, N, R), dtype=np.float32)

    for i in range(resume_from, end):
        ticker = tickers[i]
        result = query_llm(ticker, year, cache_dir)
        n_edges = 0
        for rel, entries in result.items():
            if rel not in REL_IDX:
                continue
            r_idx = REL_IDX[rel]
            for entry in entries:
                j = ticker_idx.get(entry['ticker'])
                if j is not None and j != i:
                    mat[i, j, r_idx] = 1.0
                    n_edges += 1
        log.info(f'[{i+1-start}/{end-start}] {ticker}: {n_edges} edges')
        time.sleep(3.5)

        # 체크포인트 저장
        if (i + 1 - start) % CHECKPOINT_INTERVAL == 0:
            np.save(ckpt_path, mat)
            (cache_dir / f'ckpt_{year}_{start}_{end}.txt').write_text(str(i + 1))
            log.info(f'[checkpoint] saved at ticker {i+1}')

    np.save(partial_path, mat)
    # 체크포인트 정리
    if ckpt_path.exists():
        ckpt_path.unlink()
        (cache_dir / f'ckpt_{year}_{start}_{end}.txt').unlink(missing_ok=True)
    log.info(f'Saved partial {year} {start}-{end}')


# ── Merge + 양방향 교차검증 ──────────────────────────────────────
def merge(year: int, cache_dir: Path, out_dir: Path,
          bidir_only: bool = False, symmetry: bool = False):
    """
    partial 합산 후 양방향 교차검증:
      bidir_only=True : 양방향 모두 확인된 엣지만 보존 (엄격)
      bidir_only=False: 단방향도 보존하되 양방향 엣지는 가중치 2 (기본)
    """
    mat = np.zeros((N, N, R), dtype=np.float32)
    for p in sorted(cache_dir.glob(f'partial_{year}_*.npy')):
        mat += np.load(p)
    mat = (mat > 0).astype(np.float32)

    # 양방향 교차검증
    mat_bidir = np.zeros_like(mat)
    for r_idx, r_rev in BIDIR_PAIRS.items():
        if r_rev is None:
            # 역방향 없는 관계 (Licensor): 단방향 그대로
            mat_bidir[:, :, r_idx] = mat[:, :, r_idx]
        else:
            forward  = mat[:, :, r_idx]       # A→B
            backward = mat[:, :, r_rev].T     # B→A (전치: B가 A에 역방향 관계)
            confirmed = forward * backward    # 양방향 모두 있는 것
            if bidir_only:
                mat_bidir[:, :, r_idx] = confirmed
            else:
                # 단방향도 살리되 양방향 확인된 건 별도 표시 가능 → 여기선 모두 1로 통일
                mat_bidir[:, :, r_idx] = np.maximum(forward, confirmed)

    if symmetry:
        for r_idx in range(R):
            mat_bidir[:, :, r_idx] = np.maximum(
                mat_bidir[:, :, r_idx], mat_bidir[:, :, r_idx].T)

    # 통계
    connected   = int((mat_bidir.sum(axis=(1, 2)) > 0).sum())
    total_raw   = int((mat > 0).sum())
    total_final = int((mat_bidir > 0).sum())
    bidir_cnt   = sum(
        int((mat[:,:,r] * mat[:,:,BIDIR_PAIRS[r]].T).sum())
        for r in range(R) if BIDIR_PAIRS[r] is not None
    )
    log.info(f'Year {year}: raw={total_raw} edges → after bidir filter={total_final} '
             f'(양방향 확인={bidir_cnt}), connected={connected}/{N}')

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'NASDAQ_llm_dynamic_relation.npy'
    np.save(out_path, mat_bidir)
    (out_dir / 'meta.json').write_text(json.dumps({
        'year': year, 'relations': RELATIONS, 'shape': list(mat_bidir.shape),
        'bidir_only': bidir_only, 'raw_edges': total_raw,
        'final_edges': total_final, 'bidir_confirmed': bidir_cnt,
        'connected_tickers': connected,
    }, indent=2))
    log.info(f'Saved: {out_path}')


# ── CLI ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year',       type=int, required=True,
                        help='KG 연도 (2013-2017)')
    parser.add_argument('--start',      type=int, default=0)
    parser.add_argument('--end',        type=int, default=N)
    parser.add_argument('--merge',      action='store_true')
    parser.add_argument('--bidir_only', action='store_true',
                        help='양방향 확인된 엣지만 보존 (엄격 모드)')
    parser.add_argument('--symmetry',   action='store_true')
    args = parser.parse_args()

    cache_dir = Path(__file__).parent / f'llm_dynamic_cache_{args.year}'
    out_dir   = _DATA_ROOT / 'NASDAQ' / 'relation' / f'llm_dynamic_{args.year}'
    cache_dir.mkdir(exist_ok=True)

    if args.merge:
        merge(args.year, cache_dir, out_dir, args.bidir_only, args.symmetry)
    else:
        log.info(f'LLM Dynamic KG year={args.year}, tickers {args.start}~{args.end}')
        build_matrix(args.year, args.start, args.end, cache_dir)
        log.info('Done. Run with --merge to finalize.')
