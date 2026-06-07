#!/bin/bash
# News KG 3개 연도 병렬 제출
# 실행: bash submit_news_kg.sh

cd /gpfs/home1/pz29075/Capstone/KG_StockMixer

echo "=== Submitting News KG jobs (2013 / 2015 / 2017) ==="

for year in 2013 2015 2017; do
    jid=$(sbatch --job-name=news_kg_${year} run_news_kg.sh \
        --year ${year} --start 0 --end 1026 | awk '{print $4}')
    echo "Year ${year}: job ${jid}"
done

echo ""
echo "Check status: squeue -u pz29075"
echo ""
echo "After ALL jobs complete, run merge:"
echo "  for year in 2013 2015 2017; do"
echo "    sbatch --job-name=news_merge_\${year} run_news_kg.sh --year \${year} --merge --symmetry"
echo "  done"
