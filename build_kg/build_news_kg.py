#!/usr/bin/env python3
"""
News-based Dynamic KG Construction (FMP + GPT)
FMP 뉴스 → GPT 기사 근거 관계 추출 → 연도별 KG 행렬

실행:
    export OPENAI_API_KEY="sk-..."
    python build_news_kg.py --year 2013 --start 0 --end 100
    python build_news_kg.py --year 2013 --merge --symmetry
"""
#%%
import os, sys, json, time, argparse, logging
import numpy as np
import requests
from pathlib import Path
from openai import OpenAI

#%%
_DATA_ROOT  = Path('/gpfs/home1/pz29075/Capstone/StockMixer/dataset')
TICKER_FILE = ('/gpfs/home1/pz29075/Capstone/StockMixer/'
               'Temporal_Relational_Stock_Ranking/data/'
               'NASDAQ_tickers_qualify_dr-0.98_min-5_smooth.csv')

tickers    = open(TICKER_FILE).read().strip().split('\n')
N          = len(tickers)
ticker_idx = {t: i for i, t in enumerate(tickers)}
ticker_set = set(tickers)

FMP_KEY      = os.environ.get('FMP_API_KEY', '')
FMP_NEWS_URL = 'https://financialmodelingprep.com/stable/news/stock'

#%%
RELATIONS = ['Supplier', 'Customer', 'Competitor', 'Partner', 'Subsidiary', 'Parent']
R       = len(RELATIONS)
REL_IDX = {r: i for i, r in enumerate(RELATIONS)}

#%%
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[logging.StreamHandler()])
log = logging.getLogger(__name__)
client = OpenAI()

#%%
SYSTEM_PROMPT = (
    'Extract inter-company relationships ONLY from the provided article text. '
    'Do NOT use outside knowledge. Return valid JSON only.'
)

def make_prompt(focal: str, date: str, text: str) -> str:
    return f"""Article ({date}), focal ticker: {focal}

{text[:2000]}

From the article ONLY, identify other companies and their relation to {focal}.
Relation types: Supplier(focal→other) / Customer(other→focal) / Competitor / Partner / Subsidiary / Parent

JSON:
{{"relationships":[{{"company_name":"...","ticker_guess":"...","relation":"...","evidence":"exact quote"}}]}}

If none found: {{"relationships":[]}}"""

#%%
def fetch_news(ticker: str, year: int) -> list:
    cache = _NEWS_CACHE / f'{ticker}_{year}.json'
    if cache.exists():
        return json.loads(cache.read_text())
    try:
        r = requests.get(FMP_NEWS_URL, params={
            'symbols': ticker, 'from': f'{year}-01-01',
            'to': f'{year}-12-31', 'limit': 50, 'apikey': FMP_KEY,
        }, timeout=15)
        data = r.json() if r.status_code == 200 else []
        data = data if isinstance(data, list) else []
        cache.write_text(json.dumps(data))
        return data
    except Exception as e:
        log.warning(f'FMP {ticker} {year}: {e}')
        return []


def extract_relations(focal: str, art: dict) -> list:
    art_id = ''.join(c if c.isalnum() else '_' for c in art.get('url', art.get('title',''))[:60])
    cache  = _GPT_CACHE / f'{focal}_{art_id}.json'
    if cache.exists():
        return json.loads(cache.read_text())
    text = art.get('text', '')
    if len(text) < 80:
        return []
    try:
        resp = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{'role':'system','content':SYSTEM_PROMPT},
                      {'role':'user','content':make_prompt(focal, art.get('publishedDate','')[:10], text)}],
            temperature=0.0, max_tokens=500,
            response_format={'type':'json_object'},
        )
        rels = json.loads(resp.choices[0].message.content).get('relationships', [])
        cache.write_text(json.dumps(rels))
        return rels
    except Exception as e:
        log.warning(f'GPT {focal}: {e}')
        return []

#%%
def build_matrix(year: int, start: int, end: int):
    partial_path = _GPT_CACHE / f'partial_{year}_{start}_{end}.npy'
    if partial_path.exists():
        log.info(f'Partial {year} {start}-{end} exists, skipping.')
        return

    mat = np.zeros((N, N, R), dtype=np.float32)

    for i in range(start, end):
        focal    = tickers[i]
        articles = fetch_news(focal, year)
        time.sleep(0.3)

        n_edges = 0
        for art in articles:
            for rel in extract_relations(focal, art):
                r_type  = rel.get('relation', '') or ''
                t_raw   = rel.get('ticker_guess') or ''
                t_guess = t_raw.strip().upper()
                # 'N/A', '...', '?' 등 무효 응답 필터
                if not t_guess or t_guess in {'N/A', 'NA', '...', '?', 'NONE'}:
                    continue
                if r_type not in REL_IDX or t_guess not in ticker_set or t_guess == focal:
                    continue
                mat[i, ticker_idx[t_guess], REL_IDX[r_type]] = 1.0
                n_edges += 1
            time.sleep(0.5)

        log.info(f'[{i+1-start}/{end-start}] {focal}: {len(articles)} articles → {n_edges} edges')

    np.save(partial_path, mat)
    log.info(f'Saved partial {year} {start}-{end}')


def merge(year: int, symmetry: bool = False):
    mat = np.zeros((N, N, R), dtype=np.float32)
    for p in sorted(_GPT_CACHE.glob(f'partial_{year}_*.npy')):
        mat += np.load(p)
    mat = (mat > 0).astype(np.float32)

    if symmetry:
        for r_idx in range(R):
            mat[:, :, r_idx] = np.maximum(mat[:, :, r_idx], mat[:, :, r_idx].T)

    connected   = int((mat.sum(axis=(1,2)) > 0).sum())
    total_edges = int((mat > 0).sum())
    log.info(f'Year {year}: {connected}/{N} connected, {total_edges} edges')

    out_dir = _DATA_ROOT / 'NASDAQ' / 'relation' / f'news_{year}'
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'NASDAQ_news_relation.npy', mat)
    (out_dir / 'meta.json').write_text(json.dumps(
        {'year': year, 'relations': RELATIONS, 'shape': list(mat.shape),
         'connected': connected, 'total_edges': total_edges}, indent=2))
    log.info(f'Saved: {out_dir}/NASDAQ_news_relation.npy')

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year',     type=int, required=True)
    parser.add_argument('--start',    type=int, default=0)
    parser.add_argument('--end',      type=int, default=N)
    parser.add_argument('--merge',    action='store_true')
    parser.add_argument('--symmetry', action='store_true')
    args = parser.parse_args()

    _NEWS_CACHE = Path(__file__).parent / f'news_cache_{args.year}'
    _GPT_CACHE  = Path(__file__).parent / f'news_gpt_cache_{args.year}'
    _NEWS_CACHE.mkdir(exist_ok=True)
    _GPT_CACHE.mkdir(exist_ok=True)

    if args.merge:
        merge(args.year, args.symmetry)
    else:
        log.info(f'News KG year={args.year}, tickers {args.start}~{args.end}')
        build_matrix(args.year, args.start, args.end)
        log.info('Done. Run with --merge to finalize.')
