#!/bin/bash
#SBATCH --job-name=llm_kg
#SBATCH --partition=cpu1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/home1/pz29075/Capstone/KG_StockMixer/build_kg/llm_kg_%j.out
#SBATCH --error=/gpfs/home1/pz29075/Capstone/KG_StockMixer/build_kg/llm_kg_%j.err

source /gpfs/home1/pz29075/.bashrc
conda activate stockmixer

# export OPENAI_API_KEY="your-key-here"
# Alternatively, set in environment before running: export OPENAI_API_KEY="sk-..."

cd /gpfs/home1/pz29075/Capstone/KG_StockMixer/build_kg

echo "=== Start: $(date) ==="
python build_llm_kg.py --start 5 --end 1026
echo "=== Done building: $(date) ==="

python build_llm_kg.py --merge --kappa 5
echo "=== Merge done: $(date) ==="
