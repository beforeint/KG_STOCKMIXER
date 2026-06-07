#!/bin/bash
# gat + llm_dynamic x5 seeds → KG_StockMixer_2
# 실행: bash submit_llm_dynamic_exp.sh

cd /gpfs/home1/pz29075/Capstone/KG_StockMixer

echo "=== gat_llm_dynamic x5 seeds (KG_StockMixer_2) ==="

for seed in 0 1 2 3 4; do
    jid=$(sbatch \
        --job-name=gat_ldyn_s${seed} \
        run.sh \
        --graph_type gat \
        --kg_source llm_dynamic \
        --seed ${seed} \
        --wandb_project KG_StockMixer_2 \
        | awk '{print $4}')
    echo "  seed${seed}: job ${jid}"
done
