# KG 빌드 가이드

## 실행 환경
- **로컬 Windows PC** (LSEG Workspace 앱이 실행 중이어야 함)
- 서버(SLURM)에서는 실행 불가

## 사전 준비

1. LSEG Workspace 앱 실행
2. 로컬에 패키지 설치
   ```
   pip install lseg-data numpy pandas tqdm
   ```
3. 서버에서 티커 파일 다운로드
   ```
   scp 서버:/gpfs/home1/pz29075/Capstone/StockMixer/Temporal_Relational_Stock_Ranking/data/NASDAQ_tickers_qualify_dr-0.98_min-5_smooth.csv .
   ```

## 실행 순서

```bash
python build_institutional_kg.py   # ~30분, 기관 공통 보유
python build_board_kg.py           # ~20분, 이사회 인터락
python build_supply_chain_kg.py    # ~40분, 공급망
```

## 결과 업로드

```bash
# 서버에 디렉토리 생성 (서버에서)
mkdir -p /gpfs/home1/pz29075/Capstone/StockMixer/dataset/NASDAQ/relation/institutional
mkdir -p /gpfs/home1/pz29075/Capstone/StockMixer/dataset/NASDAQ/relation/board
mkdir -p /gpfs/home1/pz29075/Capstone/StockMixer/dataset/NASDAQ/relation/supply_chain

# 로컬에서 업로드
scp NASDAQ_institutional_relation.npy 서버:/gpfs/home1/pz29075/Capstone/StockMixer/dataset/NASDAQ/relation/institutional/
scp NASDAQ_board_relation.npy         서버:/gpfs/home1/pz29075/Capstone/StockMixer/dataset/NASDAQ/relation/board/
scp NASDAQ_supply_chain_relation.npy  서버:/gpfs/home1/pz29075/Capstone/StockMixer/dataset/NASDAQ/relation/supply_chain/
```

## 업로드 후 config.py에 경로 추가

`datasets/config.py`의 `kg_sources`에 추가:
```python
'institutional':  'relation/institutional/NASDAQ_institutional_relation.npy',
'board':          'relation/board/NASDAQ_board_relation.npy',
'supply_chain':   'relation/supply_chain/NASDAQ_supply_chain_relation.npy',
```

## 실험 실행

```bash
sbatch --job-name=gcn_inst run.sh --graph_type gcn --kg_source institutional
sbatch --job-name=gat_inst run.sh --graph_type gat --kg_source institutional
sbatch --job-name=gcn_board run.sh --graph_type gcn --kg_source board
sbatch --job-name=gat_board run.sh --graph_type gat --kg_source board
```
