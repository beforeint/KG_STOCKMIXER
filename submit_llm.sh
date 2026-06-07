#!/bin/bash
# gat_llm × seeds 0~4, wandb 포함

BASE=/gpfs/home1/pz29075/Capstone/KG_StockMixer

submit() {
    local GRAPH=$1 KG=$2 SEED=$3
    sbatch --job-name="${GRAPH:0:3}_${KG:0:4}_s${SEED}" \
           --partition=gpu1 \
           --gres=gpu:1 \
           --cpus-per-task=4 \
           --mem=16G \
           --time=02:00:00 \
           --output="${BASE}/logs/${GRAPH}_${KG}_s${SEED}_%j.out" \
           --error="${BASE}/logs/${GRAPH}_${KG}_s${SEED}_%j.err" \
           --wrap="source /gpfs/home1/pz29075/.bashrc && \
                   conda activate stockmixer && \
                   cd ${BASE} && \
                   python main.py --graph_type ${GRAPH} --kg_source ${KG} --seed ${SEED}"
}

for SEED in 0 1 2 3 4; do
    submit gat llm $SEED
done

echo "Submitted 5 gat_llm jobs."
squeue -u pz29075 --noheader | wc -l
