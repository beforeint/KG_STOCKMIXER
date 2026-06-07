#!/bin/bash
#SBATCH --job-name=kg_stockmixer
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/home1/pz29075/Capstone/KG_StockMixer/logs/%x_%j.out
#SBATCH --error=/gpfs/home1/pz29075/Capstone/KG_StockMixer/logs/%x_%j.err

# 사용 예:
#   단일 실험:
#     sbatch run.sh --graph_type gcn  --kg_source wikidata
#     sbatch run.sh --graph_type gat  --kg_source wikidata
#     sbatch run.sh --graph_type none
#
#   다중 실험 (ablation 전체):
#     for gt in none gcn gat; do
#       for ks in wikidata sector_industry; do
#         [ "$gt" = "none" ] && ks_flag="" || ks_flag="--kg_source $ks"
#         sbatch --job-name=${gt}_${ks} run.sh --graph_type $gt $ks_flag
#         [ "$gt" = "none" ] && break
#       done
#     done

mkdir -p /gpfs/home1/pz29075/Capstone/KG_StockMixer/logs

source /home1/pz29075/miniconda3/etc/profile.d/conda.sh
conda activate stockmixer

export WANDB_API_KEY="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"

cd /gpfs/home1/pz29075/Capstone/KG_StockMixer
python main.py "$@"
