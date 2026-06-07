# ============================================================
# 공급망 KG 빌더
# 방법론:
#   1. TRSR 원본 NASDAQ_industry_ticker.json → ticker → 업종 레이블
#   2. 업종 레이블 → BEA IO 섹터 코드 (14개)
#   3. BEA 2017 Sector IO Use Table → 섹터간 구매 비중(기술계수)
#   4. relation[i,j] = IO coeff(sector_i → sector_j): 비대칭, 방향성 있음
#      → sector_i 가 sector_j 의 중간재 공급자일수록 높은 값
# 출력: [S, S, 1] float32
# ============================================================
import numpy as np
import pandas as pd
import json
import os
import openpyxl

# ── 경로 ────────────────────────────────────────────────────
TICKER_FILE   = '/gpfs/home1/pz29075/Capstone/StockMixer/Temporal_Relational_Stock_Ranking/data/NASDAQ_tickers_qualify_dr-0.98_min-5_smooth.csv'
INDUSTRY_JSON = '/tmp/relation/sector_industry/NASDAQ_industry_ticker.json'
BEA_IO_FILE   = '/tmp/bea_io/IOUse_After_Redefinitions_PUR_2017_Sector.xlsx'
OUTPUT_DIR    = '/gpfs/home1/pz29075/Capstone/StockMixer/dataset/NASDAQ/relation/supply_chain'
OUTPUT_FILE   = f'{OUTPUT_DIR}/NASDAQ_supply_chain_relation.npy'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. 티커 로드 ────────────────────────────────────────────
tickers = pd.read_csv(TICKER_FILE, header=None)[0].tolist()
S = len(tickers)
print(f"티커 수: {S}")

# ── 2. 업종 → ticker 역방향 매핑 ───────────────────────────
with open(INDUSTRY_JSON) as f:
    industry_tickers = json.load(f)

ticker_industry = {}
for ind, tick_list in industry_tickers.items():
    for t in tick_list:
        ticker_industry[t] = ind

covered = sum(1 for t in tickers if ticker_industry.get(t, 'n/a') != 'n/a')
print(f"업종 정보 보유 티커: {covered} / {S} ({100*covered/S:.1f}%)")

# ── 3. NASDAQ 업종명 → BEA 섹터 코드 매핑 ──────────────────
# BEA 섹터: 11=농업, 21=광업, 22=유틸리티, 23=건설, 31G=제조,
#           42=도매, 44RT=소매, 48TW=운송, 51=정보, FIRE=금융부동산,
#           PROF=전문서비스, 6=의료교육, 7=여가외식, 81=기타서비스
INDUSTRY_TO_BEA = {
    'Advertising':                                           'PROF',
    'Aerospace':                                             '31G',
    'Air Freight/Delivery Services':                         '48TW',
    'Apparel':                                               '31G',
    'Auto Manufacturing':                                    '31G',
    'Auto Parts:O.E.M.':                                     '31G',
    'Automotive Aftermarket':                                '44RT',
    'Banks':                                                 'FIRE',
    'Beverages (Production/Distribution)':                   '31G',
    'Biotechnology: Biological Products (No Diagnostic Substances)': '31G',
    'Biotechnology: Commercial Physical & Biological Resarch': 'PROF',
    'Biotechnology: Electromedical & Electrotherapeutic Apparatus': '31G',
    'Biotechnology: In Vitro & In Vivo Diagnostic Substances': '31G',
    'Biotechnology: Laboratory Analytical Instruments':      '31G',
    'Books':                                                 '31G',
    'Broadcasting':                                         '51',
    'Building Materials':                                    '31G',
    'Building Products':                                     '31G',
    'Business Services':                                     'PROF',
    'Catalog/Specialty Distribution':                        '42',
    'Clothing/Shoe/Accessory Stores':                        '44RT',
    'Coal Mining':                                           '21',
    'Commercial Banks':                                      'FIRE',
    'Computer Communications Equipment':                     '31G',
    'Computer Manufacturing':                                '31G',
    'Computer Software: Prepackaged Software':               '51',
    'Computer Software: Programming, Data Processing':       '51',
    'Computer peripheral equipment':                         '31G',
    'Construction/Ag Equipment/Trucks':                      '31G',
    'Consumer Electronics/Appliances':                       '31G',
    'Consumer Electronics/Video Chains':                     '44RT',
    'Consumer Specialties':                                  '31G',
    'Containers/Packaging':                                  '31G',
    'Department/Specialty Retail Stores':                    '44RT',
    'Diversified Commercial Services':                       'PROF',
    'EDP Services':                                          '51',
    'Electric Utilities: Central':                           '22',
    'Electrical Products':                                   '31G',
    'Electronic Components':                                 '31G',
    'Engineering & Construction':                            '23',
    'Environmental Services':                                'PROF',
    'Farming/Seeds/Milling':                                 '11',
    'Finance Companies':                                     'FIRE',
    'Finance: Consumer Services':                            'FIRE',
    'Food Chains':                                           '44RT',
    'Food Distributors':                                     '42',
    'Forest Products':                                       '31G',
    'Home Furnishings':                                      '31G',
    'Homebuilding':                                          '23',
    'Hospital/Nursing Management':                           '6',
    'Hotels/Resorts':                                        '7',
    'Industrial Machinery/Components':                       '31G',
    'Industrial Specialties':                                '31G',
    'Investment Bankers/Brokers/Service':                    'FIRE',
    'Investment Managers':                                   'FIRE',
    'Life Insurance':                                        'FIRE',
    'Major Banks':                                           'FIRE',
    'Major Chemicals':                                       '31G',
    'Major Pharmaceuticals':                                 '31G',
    'Marine Transportation':                                 '48TW',
    'Meat/Poultry/Fish':                                     '11',
    'Medical Electronics':                                   '31G',
    'Medical Specialities':                                  '6',
    'Medical/Dental Instruments':                            '31G',
    'Medical/Nursing Services':                              '6',
    'Metal Fabrications':                                    '31G',
    'Military/Government/Technical':                         'PROF',
    'Mining & Quarrying of Nonmetallic Minerals (No Fuels)': '21',
    'Miscellaneous':                                         'PROF',
    'Miscellaneous manufacturing industries':                '31G',
    'Motor Vehicles':                                        '31G',
    'Movies/Entertainment':                                  '7',
    'Multi-Sector Companies':                                'PROF',
    'Natural Gas Distribution':                              '22',
    'Office Equipment/Supplies/Services':                    '31G',
    'Oil & Gas Production':                                  '21',
    'Oil Refining/Marketing':                                '21',
    'Ophthalmic Goods':                                      '31G',
    'Ordnance And Accessories':                              '31G',
    'Other Consumer Services':                               'PROF',
    'Other Pharmaceuticals':                                 '31G',
    'Other Specialty Stores':                                '44RT',
    'Package Goods/Cosmetics':                               '31G',
    'Packaged Foods':                                        '31G',
    'Paper':                                                 '31G',
    'Plastic Products':                                      '31G',
    'Precious Metals':                                       '21',
    'Professional Services':                                 'PROF',
    'Property-Casualty Insurers':                            'FIRE',
    'Publishing':                                            '51',
    'RETAIL: Building Materials':                            '44RT',
    'Radio And Television Broadcasting And Communications Equipment': '31G',
    'Railroads':                                             '48TW',
    'Real Estate':                                           'FIRE',
    'Real Estate Investment Trusts':                         'FIRE',
    'Recreational Products/Toys':                            '31G',
    'Rental/Leasing Companies':                              'FIRE',
    'Restaurants':                                           '7',
    'Retail: Computer Software & Peripheral Equipment':      '44RT',
    'Savings Institutions':                                  'FIRE',
    'Semiconductors':                                        '31G',
    'Services-Misc. Amusement & Recreation':                 '7',
    'Shoe Manufacturing':                                    '31G',
    'Specialty Chemicals':                                   '31G',
    'Specialty Foods':                                       '31G',
    'Specialty Insurers':                                    'FIRE',
    'Steel/Iron Ore':                                        '21',
    'Telecommunications Equipment':                          '31G',
    'Television Services':                                   '51',
    'Transportation Services':                               '48TW',
    'Trucking Freight/Courier Services':                     '48TW',
    'Water Supply':                                          '22',
    'n/a':                                                   None,
}

ticker_bea = {t: INDUSTRY_TO_BEA.get(ticker_industry.get(t, 'n/a'), None)
              for t in tickers}

sector_dist = {}
for t in tickers:
    s = ticker_bea[t]
    sector_dist[s] = sector_dist.get(s, 0) + 1
print("\n섹터별 티커 분포:")
for s, cnt in sorted(sector_dist.items(), key=lambda x: -(x[1] or 0)):
    print(f"  {str(s):6s}: {cnt}개")

# ── 4. BEA IO Use Table → 기술계수 행렬 ──────────────────────
print("\nBEA IO Use Table 로드 중...")
wb = openpyxl.load_workbook(BEA_IO_FILE, read_only=True, data_only=True)
ws = wb['2017']
rows = list(ws.iter_rows(values_only=True))

VALID_SECTORS = ['11','21','22','23','31G','42','44RT','48TW','51','FIRE','PROF','6','7','81']
header_row = rows[5]
col_idx = {str(v).strip(): j for j, v in enumerate(header_row)
           if v and str(v).strip() in VALID_SECTORS}

io_data = {}
for row in rows[7:]:
    code = str(row[0]).strip() if row[0] else ''
    if code not in VALID_SECTORS:
        continue
    row_vals = {}
    for s, j in col_idx.items():
        val = row[j]
        if val in ('...', None) or val == '':
            row_vals[s] = 0.0
        else:
            try:
                row_vals[s] = float(val)
            except (ValueError, TypeError):
                row_vals[s] = 0.0
    io_data[code] = row_vals

io_df = pd.DataFrame(io_data).T.reindex(index=VALID_SECTORS, columns=VALID_SECTORS).fillna(0)
col_totals = io_df.sum(axis=0).replace(0, np.nan)
coeff_df = io_df.div(col_totals, axis=1).fillna(0)
print(f"IO 기술계수 행렬 shape: {coeff_df.shape}")
print(f"최대값: {coeff_df.values.max():.4f}")

# ── 5. (S, S, 1) 공급망 행렬 구성 ───────────────────────────
# IO 계수 > THRESHOLD인 (공급자, 구매자) 섹터 쌍만 연결
# 0.05 = "sector i가 sector j 중간재 구입의 5% 이상 차지" → 유의미한 공급관계
THRESHOLD = 0.05
print(f"\n공급망 행렬 구성 중... (임계값 = {THRESHOLD})")
relation = np.zeros((S, S, 1), dtype=np.float32)

for i, ti in enumerate(tickers):
    si = ticker_bea[ti]
    if si is None:
        continue
    for j, tj in enumerate(tickers):
        if i == j:
            continue
        sj = ticker_bea[tj]
        if sj is None:
            continue
        # relation[i,j] = "stock i(sector si)가 stock j(sector sj)의 공급자" 강도
        val = coeff_df.loc[si, sj] if (si in coeff_df.index and sj in coeff_df.columns) else 0.0
        if val >= THRESHOLD:
            relation[i, j, 0] = float(val)

# ── 6. 저장 및 통계 ─────────────────────────────────────────
np.save(OUTPUT_FILE, relation)
print(f"\n저장 완료: {OUTPUT_FILE}")
print(f"shape: {relation.shape}, dtype: {relation.dtype}")

nonzero = int(np.sum(relation[:, :, 0] > 0))
print(f"비영 엣지 쌍: {nonzero} / {S*S} ({100*nonzero/(S*S):.2f}%)")
if relation[relation > 0].size > 0:
    print(f"값 범위: [{float(relation[relation>0].min()):.6f}, {float(relation.max()):.6f}]")

is_sym = np.allclose(relation[:, :, 0], relation[:, :, 0].T)
print(f"대칭 여부: {is_sym}  (False = 비대칭 = 방향성 있는 공급망)")
