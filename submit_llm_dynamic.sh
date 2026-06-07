#!/bin/bash
# LLM Dynamic KG 전 연도 빌드 제출 — 순차 실행 버전
# API RPD(10000/day) 한도 초과 방지: 앞 연도 완료 후 다음 연도 시작
# 실행: bash submit_llm_dynamic.sh

cd /gpfs/home1/pz29075/Capstone/KG_StockMixer
mkdir -p logs

echo "=== LLM Dynamic KG build (2013→2014→2015→2016→2017, 순차) ==="

# 2013 먼저 시작
jid13=$(sbatch --job-name=llm_dyn_2013 \
    run_llm_dynamic.sh --year 2013 --start 0 --end 1026 | awk '{print $4}')
echo "  build 2013: job $jid13"

# 이후 연도는 앞 연도 완료 후 시작 (API 한도 분산)
jid14=$(sbatch --dependency=afterok:$jid13 --job-name=llm_dyn_2014 \
    run_llm_dynamic.sh --year 2014 --start 0 --end 1026 | awk '{print $4}')
echo "  build 2014: job $jid14 (after $jid13)"

jid15=$(sbatch --dependency=afterok:$jid14 --job-name=llm_dyn_2015 \
    run_llm_dynamic.sh --year 2015 --start 0 --end 1026 | awk '{print $4}')
echo "  build 2015: job $jid15 (after $jid14)"

jid16=$(sbatch --dependency=afterok:$jid15 --job-name=llm_dyn_2016 \
    run_llm_dynamic.sh --year 2016 --start 0 --end 1026 | awk '{print $4}')
echo "  build 2016: job $jid16 (after $jid15)"

jid17=$(sbatch --dependency=afterok:$jid16 --job-name=llm_dyn_2017 \
    run_llm_dynamic.sh --year 2017 --start 0 --end 1026 | awk '{print $4}')
echo "  build 2017: job $jid17 (after $jid16)"

echo ""
echo "=== Merge jobs (각 build 완료 후 즉시 실행) ==="

for year in 2013 2014 2015 2016 2017; do
    bjid_var="jid${year: -2}"  # jid13, jid14, ...
    bjid=$(eval echo \$jid${year: -2})
    mjid=$(sbatch --dependency=afterok:$bjid \
        --job-name=llm_dyn_merge_$year \
        run_llm_dynamic.sh --year $year --merge --symmetry | awk '{print $4}')
    echo "  merge $year: job $mjid (after $bjid)"
done
