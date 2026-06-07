#!/bin/bash
# SP500 LLM KG 실험 (gat × 5 seeds)
# Prerequisites:
#   1. Job 648338 complete: SP500_llm_relation.npy must exist
#   2. config.py SP500 must have 'llm' key

cd /gpfs/home1/pz29075/Capstone/KG_StockMixer

LLM_NPY="/gpfs/home1/pz29075/Capstone/StockMixer/dataset/SP500/relation/llm_v2/SP500_llm_relation.npy"
if [ ! -f "$LLM_NPY" ]; then
    echo "ERROR: LLM KG not found: $LLM_NPY"
    echo "Wait for job 648338 to complete and run merge first."
    exit 1
fi

echo "=== Submitting SP500 gat_llm × 5 seeds ==="
for seed in 0 1 2 3 4; do
    sbatch --job-name=sp5_gat_llm_s${seed} run.sh --market SP500 --graph_type gat --kg_source llm --seed ${seed}
done

echo "Submitted 5 jobs. Check: squeue -u pz29075"
squeue -u pz29075 --noheader | wc -l
