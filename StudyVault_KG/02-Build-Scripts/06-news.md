# 06. News KG 빌더 (동적 KG)

> 순서: 6/6 (02-Build-Scripts)  
> 파일: `build_kg/build_news_kg.py`  
> 출력: `NASDAQ_news_relation.npy`  shape=(1026,1026,6)  (연도별)

---

## 핵심 아이디어

> FMP(Financial Modeling Prep)에서 뉴스 기사를 수집하고, GPT로 기사 본문에서만 관계를 추출한다.  
> 연도별로 동적 KG 생성 → 시계열 KG 가능

---

## 6가지 관계 타입

```python
RELATIONS = ['Supplier', 'Customer', 'Competitor', 'Partner', 'Subsidiary', 'Parent']
```

LLM KG(8종)보다 단순화된 6종.

---

## 파이프라인

```
FMP API → 뉴스 기사 수집 (연도/ticker별)
    ↓
GPT-4o-mini → 기사 본문에서 관계 추출 (외부 지식 금지)
    ↓
[N, N, 6] 행렬 → partial 저장
    ↓
merge → (선택) 대칭화 → 최종 저장
```

---

## 핵심 코드

```python
# FMP 뉴스 수집
FMP_KEY = 'your-key'
FMP_NEWS_URL = 'https://financialmodelingprep.com/stable/news/stock'

def fetch_news(ticker, year):
    r = requests.get(FMP_NEWS_URL, params={
        'symbols': ticker, 'from': f'{year}-01-01',
        'to': f'{year}-12-31', 'limit': 50, 'apikey': FMP_KEY,
    })
    return r.json()

# GPT 관계 추출 (기사 본문에서만)
SYSTEM_PROMPT = (
    'Extract inter-company relationships ONLY from the provided article text. '
    'Do NOT use outside knowledge. Return valid JSON only.'
)

def make_prompt(focal, date, text):
    return f"""Article ({date}), focal ticker: {focal}

{text[:2000]}

From the article ONLY, identify other companies and their relation to {focal}.
Relation types: Supplier / Customer / Competitor / Partner / Subsidiary / Parent

JSON:
{{"relationships":[{{"company_name":"...","ticker_guess":"...","relation":"...","evidence":"exact quote"}}]}}
"""
```

---

## LLM KG와의 차이

| | LLM KG | News KG |
|--|--------|---------|
| 데이터 소스 | GPT 사전지식 | FMP 뉴스 기사 |
| 시간 범위 | 고정 (2012) | 연도별 동적 |
| 관계 타입 | 8종 | 6종 |
| 비용 | GPT API | FMP API + GPT API |
| Look-ahead bias | 시간 마스킹 | 기사 날짜 기반 자동 방지 |
| KG 유형 | 정적 | **동적 (연도별)** |

---

## 실행

```bash
export OPENAI_API_KEY="sk-..."

# Step 1: 연도별 구축
python build_news_kg.py --year 2013 --start 0 --end 500
python build_news_kg.py --year 2013 --start 500 --end 1026

# Step 2: merge + 대칭화
python build_news_kg.py --year 2013 --merge --symmetry

# 여러 연도
for year in 2013 2014 2015 2016 2017; do
    python build_news_kg.py --year $year
    python build_news_kg.py --year $year --merge --symmetry
done
```

---

## 저장 구조

```
dataset/NASDAQ/relation/
├── news_2013/
│   ├── NASDAQ_news_relation.npy   # (1026, 1026, 6)
│   └── meta.json
├── news_2014/
│   └── ...
└── ...
```

---

## 무효 ticker 필터링

```python
INVALID = {'N/A', 'NA', '...', '?', 'NONE'}
if not t_guess or t_guess in INVALID:
    continue
if t_guess not in ticker_set:   # universe 밖 ticker 제거
    continue
```
