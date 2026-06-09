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

| Model | IC | Low-Vol IC | p-value |
|---|---|---|---|
| Baseline (No KG) | 0.0156 | 0.0141 | — |
| GAT + Wikidata | 0.0203 | 0.0227 | p=0.053 |
| GAT + NewsDynamic | **0.0201** | **0.0255** | **p=0.0004** |
| GAT + LLMDynamic | 0.0153 | — | n.s. |

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
