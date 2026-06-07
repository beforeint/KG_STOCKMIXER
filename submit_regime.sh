#!/bin/bash
# Regime analysis용 재실행: none + gat_wiki + gat_inst + gat_supply x seeds 0-4
# per-day IC → results/{run_name}_daily_ic.npy 저장

LOGDIR=/gpfs/home1/pz29075/Capstone/KG_StockMixer/logs
mkdir -p "$LOGDIR"

submit() {
    local GTYPE=$1 KSRC=$2 SEED=$3
    local NAME="${GTYPE}_${KSRC}_s${SEED}"
    sbatch --partition=gpu1 \
           --gres=gpu:1 \
           --time=02:00:00 \
           --job-name="$NAME" \
           --output="${LOGDIR}/regime_${NAME}_%j.out" \
           --error="${LOGDIR}/regime_${NAME}_%j.err" \
           --wrap="source ~/.bashrc && conda activate stockmixer && \
                   cd /gpfs/home1/pz29075/Capstone/KG_StockMixer && \
                   python main.py --graph_type ${GTYPE} --kg_source ${KSRC} --seed ${SEED} --no_wandb"
    echo "Submitted: $NAME"
}

for SEED in 0 1 2 3 4; do
    submit none     wikidata     $SEED
    submit gat      wikidata     $SEED
    submit gat      institutional $SEED
    submit gat      supply_chain $SEED
done
