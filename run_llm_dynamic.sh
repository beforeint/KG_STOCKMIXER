#!/bin/bash
#SBATCH --job-name=llm_dyn
#SBATCH --partition=cpu1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=/gpfs/home1/pz29075/Capstone/KG_StockMixer/logs/%x_%j.out
#SBATCH --error=/gpfs/home1/pz29075/Capstone/KG_StockMixer/logs/%x_%j.err

mkdir -p /gpfs/home1/pz29075/Capstone/KG_StockMixer/logs

source /home1/pz29075/miniconda3/etc/profile.d/conda.sh
conda activate stockmixer

export OPENAI_API_KEY="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"

cd /gpfs/home1/pz29075/Capstone/KG_StockMixer/build_kg
python build_llm_kg_dynamic.py "$@"
