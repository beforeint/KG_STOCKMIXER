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

# export OPENAI_API_KEY="your_openai_api_key_here"

cd /gpfs/home1/pz29075/Capstone/KG_StockMixer/build_kg
python build_llm_kg_dynamic.py "$@"
