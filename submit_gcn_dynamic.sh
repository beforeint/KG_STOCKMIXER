#!/bin/bash
# gcn + news_dynamic / llm_dynamic x5 seeds → KG_StockMixer_2

cd /gpfs/home1/pz29075/Capstone/KG_StockMixer

echo "=== gcn_news_dynamic x5 seeds ==="
for seed in 0 1 2 3 4; do
    jid=$(sbatch \
        --job-name=gcn_ndyn_s${seed} \
        run.sh \
        --graph_type gcn \
        --kg_source news_dynamic \
        --seed ${seed} \
        --wandb_project KG_StockMixer_2 \
        | awk '{print $4}')
    echo "  seed${seed}: job ${jid}"
done

echo ""
echo "=== gcn_llm_dynamic x5 seeds ==="
for seed in 0 1 2 3 4; do
    jid=$(sbatch \
        --job-name=gcn_ldyn_s${seed} \
        run.sh \
        --graph_type gcn \
        --kg_source llm_dynamic \
        --seed ${seed} \
        --wandb_project KG_StockMixer_2 \
        | awk '{print $4}')
    echo "  seed${seed}: job ${jid}"
done

echo ""
echo "squeue -u pz29075 | grep gcn_"
