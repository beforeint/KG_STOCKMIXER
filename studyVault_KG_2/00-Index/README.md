# KG StockMixer — 빌드 기록 & 시행착오 StudyVault

> 생성일: 2026-06-01  
> 대상 프로젝트: `/gpfs/home1/pz29075/Capstone/KG_StockMixer/`

---

## 폴더 구조 (순서대로 읽을 것)

```
StudyVault_KG/
├── 00-Index/
│   └── README.md              ← 지금 여기
├── 01-KG-Structure/
│   ├── 01-KG-텐서형식.md       ← KG [S,S,R] 구조 설명
│   ├── 02-KG-6종-비교.md      ← 6개 KG 통계 및 비교
│   ├── 03-실험결과-비교.md     ← 전체 실험 지표 (IC/RIC/P@10/SR/레짐)
│   └── 04-모델아키텍처-평가지표.md ← GATMixer 수식 완전 가이드 ★
├── 02-Build-Scripts/
│   ├── 01-institutional.md    ← Jaccard 기반 기관투자자 KG
│   ├── 02-sector.md           ← 섹터/업종 이진 KG
│   ├── 03-board.md            ← 임원 공유 인터락 KG
│   ├── 04-supply-chain.md     ← BEA IO 테이블 기반 공급망 KG
│   ├── 05-llm.md              ← GPT-4o-mini 제로샷 멀티관계 KG
│   ├── 06-news.md             ← FMP뉴스 + GPT 동적 KG ★ (merge수식, confound, FinDKG비교)
│   ├── 07-llm-dynamic.md     ← Self-Consistency Voting + 양방향 검증 KG (실패원인 분석)
│   └── 08-wikidata-temporal.md ← Temporal Wikidata 아이디어 (미구현, future work)
├── 03-Trials/
│   ├── 01-데이터수집-오류.md   ← LSEG/yfinance/EDGAR 실패기
│   ├── 02-Python-환경-오류.md  ← 버전 충돌, numpy, conda 문제
│   ├── 03-LLM-KG-오류.md      ← API키/캐시/V1→V2 전환 과정
│   ├── 04-학습-오류.md         ← argparse/price_data/IC=NaN
│   ├── 05-SP500-오류.md       ← SP500 확장 시 발생한 문제들
│   └── 06-API-RateLimit-오류.md ← RPD 한도/Time Limit/Python 3.8 문제 ★
└── 04-Concepts/
    ├── 01-Jaccard유사도.md     ← 기관 동시보유 판단 원리
    ├── 02-SEC-13F-공시.md      ← yfinance 데이터 출처
    ├── 03-BEA-IO-테이블.md     ← 공급망 계수 원리
    ├── 04-Self-Consistency-Voting.md ← Wang et al. ICLR 2023, hallucination 제거 ★
    └── 05-HGAT-vs-GAT.md      ← HGAT 구현 및 실패 원인 분석 (sparsity)
```

---

## 한 줄 요약

| 파일 | 내용 |
|------|------|
| `01-KG-텐서형식` | 모든 KG는 `[S, S, R]` float32, S=1026, R은 종류별 상이 |
| `02-KG-6종-비교` | Wikidata/Sector/Institutional/Board/SupplyChain/LLM 통계 |
| `01-institutional` | Jaccard ≥ 0.01, yfinance 13F, density 29% |
| `02-sector` | 같은 industry 문자열이면 1.0, Wikidata dim24에서 추출 |
| `03-board` | 공유 임원 비율, 21쌍 (너무 sparse) |
| `04-supply-chain` | BEA IO 계수 ≥ 0.05, 비대칭, 84.8% 커버리지 |
| `05-llm` | GPT-4o-mini, 8관계, V1→V2 개선, 시간마스킹 2012 |
| `06-news` | FMP 뉴스 + GPT 관계 추출, 연도별 동적 KG |
| `07-llm-dynamic` | Self-Consistency 3-vote + 양방향 교차검증, cpu1 파티션 |
| `03-실험결과-비교` | GAT/HGAT/GCN 조건별 IC, 아키텍처 비교 포함 (2026-06-06 업데이트) |
| `04-Self-Consistency` | Wang et al. ICLR 2023, temperature=0.5 × 3회 다수결 |
| `05-HGAT-vs-GAT` | HGAT 구현, GAT 미달 원인 = sparsity, negative finding |
| `06-API-RateLimit` | RPD 10000/day 초과 → 순차실행, Python 3.8 type hint 오류 |
| `06-news` (업데이트) | merge 수식, 2단계 정보손실, confound, FinDKG 비교 추가 |
| `07-llm-dynamic` (업데이트) | 실패 원인 3가지, NewsDynamic 비교 추가 |
| `08-wikidata-temporal` | Temporal Wikidata 아이디어, delta update, future work |
