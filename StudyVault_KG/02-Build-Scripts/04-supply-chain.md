# 04. Supply Chain KG 빌더

> 순서: 4/6 (02-Build-Scripts)  
> 파일: `build_kg/build_supply_chain_kg.py`  
> 출력: `NASDAQ_supply_chain_relation.npy`  shape=(1026,1026,1)

---

## 핵심 아이디어

> BEA(미국 경제분석국) 산업연관(IO) 표를 이용해 섹터 간 공급 관계를 수치화한다.  
> relation[i,j] = "종목 i(섹터)가 종목 j(섹터)의 중간재 공급자" 강도

유일하게 **비대칭(방향성 있는)** KG.

---

## 구축 절차 (순서대로)

### 1단계: 티커 → NASDAQ 업종명

```python
# TRSR 원본 NASDAQ_industry_ticker.json 활용
with open(INDUSTRY_JSON) as f:
    industry_tickers = json.load(f)
# { "Semiconductors": ["AAPL", "INTC", ...], ... }

ticker_industry = {}
for ind, tick_list in industry_tickers.items():
    for t in tick_list:
        ticker_industry[t] = ind
```

### 2단계: NASDAQ 업종명 → BEA 14개 섹터 코드

113개 NASDAQ 업종을 수동으로 BEA 섹터로 매핑:

| BEA 코드 | 섹터 이름 | 해당 NASDAQ 업종 예시 |
|---------|----------|-------------------|
| 11 | 농업 | Farming/Seeds/Milling |
| 21 | 광업/에너지 | Oil & Gas Production, Coal Mining |
| 22 | 유틸리티 | Electric Utilities, Natural Gas |
| 23 | 건설 | Engineering & Construction |
| 31G | 제조 | Semiconductors, Computer Manufacturing, Auto |
| 42 | 도매 | Food Distributors |
| 44RT | 소매 | Department Stores, Food Chains |
| 48TW | 운송 | Air Freight, Railroads |
| 51 | 정보/소프트웨어 | Computer Software, Broadcasting |
| FIRE | 금융/부동산 | Banks, Insurance, REITs |
| PROF | 전문서비스 | Advertising, Business Services |
| 6 | 의료/교육 | Hospital, Medical Services |
| 7 | 여가/외식 | Restaurants, Hotels, Movies |
| 81 | 기타서비스 | — |

```python
INDUSTRY_TO_BEA = {
    'Semiconductors': '31G',
    'Computer Software: Prepackaged Software': '51',
    'Banks': 'FIRE',
    'Major Pharmaceuticals': '31G',
    # ... (113개 전체 목록은 build_supply_chain_kg.py 참조)
}
```

### 3단계: BEA IO Use Table → 기술계수 계산

```python
# BEA 2017 Sector IO Use Table (xlsx)
wb = openpyxl.load_workbook(BEA_IO_FILE, read_only=True, data_only=True)
ws = wb['2017']

# 열 합계로 정규화 → 기술계수 (technical coefficient)
# coeff_df[i, j] = "섹터 i가 섹터 j 중간재 구입의 몇 %를 차지하는가"
col_totals = io_df.sum(axis=0).replace(0, np.nan)
coeff_df = io_df.div(col_totals, axis=1).fillna(0)
```

### 4단계: 종목 간 관계 행렬 구성

```python
THRESHOLD = 0.05   # IO 계수 ≥ 5%만 연결

for i, ti in enumerate(tickers):
    si = ticker_bea[ti]   # 종목 i의 BEA 섹터
    for j, tj in enumerate(tickers):
        if i == j:
            continue
        sj = ticker_bea[tj]
        val = coeff_df.loc[si, sj]  # si가 sj의 공급자 강도
        if val >= THRESHOLD:
            relation[i, j, 0] = float(val)  # 비대칭: i→j만
```

---

## 결과 통계

- shape: (1026, 1026, 1)
- 비영 엣지: 279,448개
- Density: 53.1%
- 티커 커버리지: 870/1026 (84.8%)
- 고립 노드: 156개 (업종 매핑 없는 종목)
- 값 범위: [0.051, 0.769]
- 대칭: **False** (방향성 있음)

---

## 시행착오: supply_chain.npy가 institutional.npy와 동일했던 사건

```python
# 확인 방법
inst  = np.load('institutional/NASDAQ_institutional_relation.npy')
supp  = np.load('supply_chain/NASDAQ_supply_chain_relation.npy')
print(np.array_equal(inst, supp))  # True → 버그!
```

**원인**: `build_supply_chain_kg.py`가 상대 경로로 저장하여 다른 위치에 출력됨.  
제출 시 institutional 파일을 supply_chain 경로로 복사한 것.

**해결**: 절대 경로 강제 사용
```python
OUTPUT_DIR  = '/gpfs/home1/pz29075/Capstone/StockMixer/dataset/NASDAQ/relation/supply_chain'
OUTPUT_FILE = f'{OUTPUT_DIR}/NASDAQ_supply_chain_relation.npy'
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

---

## 필요한 파일

```bash
# TRSR 원본 industry_ticker.json 추출
tar xzf relation.tar.gz relation/sector_industry/NASDAQ_industry_ticker.json

# BEA IO Use Table 다운로드 (bea.gov, 무료)
# https://www.bea.gov/industry/input-output-accounts-data
# → IOUse_After_Redefinitions_PUR_2017_Sector.xlsx
```

---

## 실행

```bash
conda run -n stockmixer python build_supply_chain_kg.py
# → (1026, 1026, 1), 279,448 undirected edges, asymmetric
```
