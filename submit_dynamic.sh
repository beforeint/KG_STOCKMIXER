#!/bin/bash
# gat + news_dynamic x5 seeds → KG_StockMixer_2
# 모든 연도 KG merge 완료 후 실행: bash submit_dynamic.sh
#
# merge job 의존성 걸어서 자동 실행:
#   bash submit_dynamic.sh --after 660612:660613:660614:660607:660608

cd /gpfs/home1/pz29075/Capstone/KG_StockMixer

DEPENDENCY=""
if [ "$1" = "--after" ] && [ -n "$2" ]; then
    DEPENDENCY="--dependency=afterok:$2"
    echo "의존성: job $2 완료 후 시작"
fi

echo "=== gat_news_dynamic x5 seeds (KG_StockMixer_2) ==="

for seed in 0 1 2 3 4; do
    jid=$(sbatch $DEPENDENCY \
        --job-name=gat_ndyn_s${seed} \
        run.sh \
        --graph_type gat \
        --kg_source news_dynamic \
        --seed ${seed} \
        --wandb_project KG_StockMixer_2 \
        | awk '{print $4}')
    echo "  seed${seed}: job ${jid}"
done

echo ""
echo "squeue -u pz29075 | grep gat_ndyn"
