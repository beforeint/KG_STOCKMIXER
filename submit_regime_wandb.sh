#!/bin/bash
# Regime 분석 재실행 — wandb 활성화 버전
# wandb summary에 regime/bull_ic, regime/low_vol_ic 등 추가 기록

LOGDIR=/gpfs/home1/pz29075/Capstone/KG_StockMixer/logs
mkdir -p "$LOGDIR"

submit() {
    local GTYPE=$1 KSRC=$2 SEED=$3
    local NAME="${GTYPE}_${KSRC}_s${SEED}"
    sbatch --partition=gpu1 \
           --gres=gpu:1 \
           --time=02:00:00 \
           --job-name="$NAME" \
           --output="${LOGDIR}/regime_wb_${NAME}_%j.out" \
           --error="${LOGDIR}/regime_wb_${NAME}_%j.err" \
           --wrap="source ~/.bashrc && conda activate stockmixer && \
                   cd /gpfs/home1/pz29075/Capstone/KG_StockMixer && \
                   python main.py --graph_type ${GTYPE} --kg_source ${KSRC} --seed ${SEED}"
    echo "Submitted: $NAME"
}

for SEED in 0 1 2 3 4; do
    submit none wikidata      $SEED
    submit gat  wikidata      $SEED
    submit gat  institutional $SEED
    submit gat  supply_chain  $SEED
done
