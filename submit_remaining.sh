#!/bin/bash
# 한도 초과로 못 넣은 나머지 실험들
# 기존 job들 완료 후 실행: bash submit_remaining.sh

cd /gpfs/home1/pz29075/Capstone/KG_StockMixer

# board (gcn seed2-4, gat seed0-4)
for seed in 2 3 4; do
    sbatch --job-name=gcn_boar_s${seed} run.sh --graph_type gcn --kg_source board --seed ${seed}
done
for seed in 0 1 2 3 4; do
    sbatch --job-name=gat_boar_s${seed} run.sh --graph_type gat --kg_source board --seed ${seed}
done

# supply_chain (gcn/gat seed0-4)
for gt in gcn gat; do
    for seed in 0 1 2 3 4; do
        sbatch --job-name=${gt}_supp_s${seed} run.sh --graph_type ${gt} --kg_source supply_chain --seed ${seed}
    done
done

echo "제출 완료"
squeue -u pz29075 --noheader | wc -l
