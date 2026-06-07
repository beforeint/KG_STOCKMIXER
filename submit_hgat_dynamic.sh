#!/bin/bash
# hgat + news_dynamic x5 seeds → KG_StockMixer_2

cd /gpfs/home1/pz29075/Capstone/KG_StockMixer

echo "=== hgat_news_dynamic x5 seeds (KG_StockMixer_2) ==="

for seed in 0 1 2 3 4; do
    jid=$(sbatch \
        --job-name=hgat_ndyn_s${seed} \
        run.sh \
        --graph_type hgat \
        --kg_source news_dynamic \
        --seed ${seed} \
        --wandb_project KG_StockMixer_2 \
        | awk '{print $4}')
    echo "  seed${seed}: job ${jid}"
done

echo ""
echo "squeue -u pz29075 | grep hgat_ndyn"
