# 03. BEA IO Use Table — 공급망 KG의 원리

> 순서: 3/3 (04-Concepts)

---

## BEA IO Use Table이란?

**BEA(Bureau of Economic Analysis)** 미국 경제분석국이 제공하는 산업연관표.

- 산업 간 중간재 거래 관계를 정량화
- 5년 주기 발행 (2017년 버전 사용)
- 파일: `IOUse_After_Redefinitions_PUR_2017_Sector.xlsx`
- 무료 다운로드: [bea.gov](https://www.bea.gov/industry/input-output-accounts-data)

---

## 기술계수(Technical Coefficient)란?

```
coeff[i, j] = "섹터 i가 섹터 j의 중간재 구입 총액에서 차지하는 비중"
```

예시:
```
coeff['31G', '51'] = 0.15
→ "정보/소프트웨어 업종이 중간재를 살 때 15%를 제조업에서 구입"
→ 제조업이 소프트웨어업의 주요 공급자
```

계산:
```python
io_df   = 원본 IO Use Table (섹터 × 섹터, 단위: 백만 달러)
col_sum = io_df.sum(axis=0)          # 각 섹터의 중간재 구입 총액
coeff   = io_df / col_sum            # 열(column) 기준 정규화
```

---

## Supply Chain KG에서의 적용

```python
THRESHOLD = 0.05   # 5% 이상 차지하는 공급 관계만 연결

relation[i, j, 0] = coeff[sector_i, sector_j]
# → "종목 i(섹터)가 종목 j(섹터)의 중간재를 5% 이상 공급"
```

**비대칭**: `coeff[i, j] ≠ coeff[j, i]`
- "철강이 자동차에 공급" ≠ "자동차가 철강에 공급"

---

## 14개 BEA 섹터

| 코드 | 섹터 | NASDAQ 예시 업종 |
|------|------|----------------|
| 11 | 농업 | Farming/Seeds/Milling |
| 21 | 광업/에너지 | Oil & Gas, Coal Mining |
| 22 | 유틸리티 | Electric Utilities |
| 23 | 건설 | Engineering & Construction |
| 31G | 제조 | Semiconductors, Pharma, Auto |
| 42 | 도매 | Food Distributors |
| 44RT | 소매 | Department Stores |
| 48TW | 운송 | Air Freight, Railroads |
| 51 | 정보/SW | Computer Software |
| FIRE | 금융/부동산 | Banks, Insurance, REITs |
| PROF | 전문서비스 | Advertising, Consulting |
| 6 | 의료/교육 | Hospital, Medical Services |
| 7 | 여가/외식 | Restaurants, Hotels |
| 81 | 기타 | — |

---

## 장점과 한계

| 장점 | 한계 |
|------|------|
| 무료 (정부 데이터) | 5년 주기 → 시의성 떨어짐 |
| 섹터 단위 → 구 ticker 커버 가능 | 섹터 내 종목 간 동질성 가정 |
| 정량적 공급 강도 | 기업 수준 공급망 아님 |
| 비대칭 (방향성) 표현 가능 | 113개 업종→14개 섹터 수동 매핑 필요 |

---

## 핵심 교훈

BEA IO는 개별 기업 수준의 공급망이 아니라 **섹터 수준 근사치**다.  
그럼에도 무료 + 오프라인 가능 + 비대칭 표현 + 84.8% 커버리지로  
yfinance 기반 방법들보다 품질이 높다.
