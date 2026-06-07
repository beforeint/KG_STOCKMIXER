#!/bin/bash
#SBATCH --job-name=news_kg
#SBATCH --partition=gpu1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/home1/pz29075/Capstone/KG_StockMixer/logs/%x_%j.out
#SBATCH --error=/gpfs/home1/pz29075/Capstone/KG_StockMixer/logs/%x_%j.err

# 사용 예:
#   sbatch run_news_kg.sh --year 2013 --start 0 --end 1026
#   sbatch run_news_kg.sh --year 2013 --merge --symmetry

mkdir -p /gpfs/home1/pz29075/Capstone/KG_StockMixer/logs

source /home1/pz29075/miniconda3/etc/profile.d/conda.sh
conda activate stockmixer

# export OPENAI_API_KEY="your_openai_api_key_here"

cd /gpfs/home1/pz29075/Capstone/KG_StockMixer/build_kg
python build_news_kg.py "$@"
