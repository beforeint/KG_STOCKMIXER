#!/bin/bash
#SBATCH --job-name=llm_kg_v2
#SBATCH --partition=cpu1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/home1/pz29075/Capstone/KG_StockMixer/build_kg/llm_kg_v2_%j.out
#SBATCH --error=/gpfs/home1/pz29075/Capstone/KG_StockMixer/build_kg/llm_kg_v2_%j.err

source /gpfs/home1/pz29075/.bashrc
conda activate stockmixer

# export OPENAI_API_KEY="your-key-here"

cd /gpfs/home1/pz29075/Capstone/KG_StockMixer/build_kg

echo "=== V2 Start: $(date) ==="
python build_llm_kg.py --v2 --start 0 --end 1026
echo "=== V2 Build done: $(date) ==="

python build_llm_kg.py --v2 --merge --kappa 5 --symmetry
echo "=== V2 Merge+Symmetry done: $(date) ==="
