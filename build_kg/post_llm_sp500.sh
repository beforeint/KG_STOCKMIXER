#!/bin/bash
# Run after LLM KG job 648338 completes
# Step 1: merge + symmetrize
cd /gpfs/home1/pz29075/Capstone/KG_StockMixer/build_kg
source /gpfs/home1/pz29075/miniconda3/etc/profile.d/conda.sh
conda activate stockmixer

# export OPENAI_API_KEY="..."  # pass via environment

python build_llm_kg.py --market SP500 --v2 --merge --kappa 1 --symmetry
echo "=== Merge done ==="
ls -lh /gpfs/home1/pz29075/Capstone/StockMixer/dataset/SP500/relation/llm/
