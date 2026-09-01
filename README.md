# KG-StockMixer

Knowledge Graph-augmented stock return prediction built on [StockMixer (Chen et al., AAAI 2024)](https://github.com/SJTU-DMTai/StockMixer).  
Compares 7 KG sources (Wikidata, Institutional, Supply Chain, LLM-static, NewsDynamic, LLMDynamic, FreqWeighted) across market regimes on NASDAQ 1,026 stocks.

---

## Requirements

```
Python 3.8+
torch==2.4.1
numpy==1.22.1
pandas==2.0.3
wandb==0.24.2
openai>=2.0       # KG 구축 시에만 필요
requests          # KG 구축 시에만 필요
```

```bash
conda create -n stockmixer python=3.9
conda activate stockmixer
pip install torch==2.4.1 numpy==1.22.1 pandas==2.0.3 wandb openai requests
```

---

## Data

StockMixer 원본 데이터셋을 `dataset/NASDAQ/` 아래에 배치:

```
dataset/
└── NASDAQ/
    ├── price_long_50.csv       # OHLCV (1026 종목 × 2526일)
    └── relation/
        ├── wikidata/           NASDAQ_wiki_relation.npy        (1026,1026,43)
        ├── sector_industry/    NASDAQ_industry_relation.npy    (1026,1026,1)
        ├── institutional/      NASDAQ_institutional_relation.npy
        ├── supply_chain/       NASDAQ_supply_chain_relation.npy
        ├── llm_v2/             NASDAQ_llm_relation.npy
        ├── news_{year}/        NASDAQ_news_relation.npy        × 2013~2017
        ├── llm_dynamic_{year}/ NASDAQ_llm_dynamic_relation.npy × 2013~2017
        └── news_freq_{year}/   NASDAQ_news_freq_relation.npy   × 2013~2017
```

`datasets/config.py`의 `DATA_ROOT`를 실제 경로로 수정하세요.

---

## Run

### 기본 실험

```bash
# Baseline (KG 없음)
python main.py --market NASDAQ --graph_type none

# Static KG
python main.py --market NASDAQ --graph_type gat --kg_source wikidata
python main.py --market NASDAQ --graph_type gat --kg_source llm_v2

# Dynamic KG
python main.py --market NASDAQ --graph_type gat --kg_source news_dynamic
python main.py --market NASDAQ --graph_type gat --kg_source llm_dynamic
```

### Ablation (5 seeds)

```bash
for seed in 0 1 2 3 4; do
    python main.py --market NASDAQ --graph_type gat \
                   --kg_source news_dynamic --seed ${seed}
done
```

### SLURM (서버 환경)

```bash
# 전체 ablation 한 번에 제출
bash submit_kg_ablation.sh

# 개별 job
sbatch --job-name=gat_nd --partition=gpu1 --gres=gpu:1 \
       --wrap="conda run -n stockmixer python main.py \
               --market NASDAQ --graph_type gat --kg_source news_dynamic"
```

---

## KG 직접 구축 (선택)

```bash
# NewsDynamic (FMP API + OpenAI API 키 필요)
export OPENAI_API_KEY="sk-..."
for year in 2013 2014 2015 2016 2017; do
    python build_kg/build_news_kg.py --year ${year}
    python build_kg/build_news_kg.py --year ${year} --merge --symmetry
done

# LLMDynamic (OpenAI API 키 필요)
for year in 2013 2014 2015 2016 2017; do
    python build_kg/build_llm_kg_dynamic.py --year ${year}
    python build_kg/build_llm_kg_dynamic.py --year ${year} --merge
done
```

---

## Results


| Model | IC | RIC | P@10 | SR |
|---|---|---|---|---|
| No KG (Baseline) | 0.0156 ± 0.0052 | 0.1616 | 0.5151 | 0.393 |
| GAT + Wikidata | **0.0203** ± 0.0083 | 0.0738 | 0.5188 | −0.954 |
| GAT + Sector | 0.0028 ± 0.0053 | 0.1424 | 0.5177 | 0.409 |
| GAT + Institutional | 0.0166 ± 0.0085 | 0.0986 | 0.5176 | −0.247 |
| GAT + SupplyChain | 0.0168 ± 0.0093 | 0.0986 | 0.5176 | −0.247 |
| GAT + LLM Static | 0.0174 ± 0.0094 | 0.1514 | 0.5228 | 0.318 |
| **GAT + NewsDynamic** | **0.0201** ± 0.0120 | **0.1688** | 0.5198 | −0.006 |
| GAT + LLMDynamic | 0.0153 ± 0.0095 | **0.1744** | 0.5175 | 0.405 |
| GCN + NewsDynamic | 0.0160 ± 0.0071 | 0.0726 | 0.5226 | 1.252 |
| GCN + LLMDynamic | 0.0146 ± 0.0049 | 0.1068 | 0.5136 | **1.621** |
| HGAT + NewsDynamic | 0.0144 ± 0.0083 | 0.1234 | **0.5237** | 0.293 |
| HGAT + LLMDynamic | 0.0151 ± 0.0077 | 0.1318 | **0.5241** | 0.300 |

> Test set: NASDAQ, 2016-11-21 ~ 2017-10-27 (237일), 5 seeds mean

---

## Citation

```bibtex
@inproceedings{chen2024stockmixer,
  title={StockMixer: A Simple Yet Strong MLP Mixer for Stock Price Forecasting},
  author={Chen, Ting and others},
  booktitle={AAAI},
  year={2024}
}
```

## Acknowledgement
This repository was developed with support from the 서울시립대학교 데이터 사이언스 플러스 차세대 융합인재 양성사업단 - http://dsplus.uos.ac.kr/
