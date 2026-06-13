# 02. Python 환경 / 버전 오류

> 순서: 2/5 (03-Trials)

---

## Trial-07: yfinance 버전 충돌 (Python 3.8 서버)

**현상**: `TypeError: 'type' object is not subscriptable` (multitasking 패키지)  
**원인**: 최신 yfinance가 Python 3.9+ 문법(`list[str]`, `dict[str, int]` 등) 사용  
**해결**:

```bash
pip install "yfinance==0.2.18"
pip install "multitasking==0.0.9"
```

---

## Trial-08: Python 3.8 dict union 연산자 오류

**현상**: `TypeError: unsupported operand type(s) for |: 'dict' and 'dict'`  
**위치**: `main.py`의 `wandb.summary.update(...)` 부분  
**원인**: `dict1 | dict2` 문법은 Python 3.9+

```python
# 오류 코드 (Python 3.9+)
wandb.summary.update(
    {'best_valid/' + k: v for k,v in best_valid_perf.items()}
    | {'best_test/' + k: v for k,v in best_test_perf.items()}
)

# 수정 코드 (Python 3.8 호환)
wandb.summary.update({
    **{'best_valid/'+k: v for k,v in best_valid_perf.items()},
    **{'best_test/'+k: v for k,v in best_test_perf.items()}
})
```

---

## Trial-09: numpy 버전 불일치 (pickle 호환 오류)

**현상**: SP500 pkl 파일 저장 후 `import pickle; pickle.load(...)` 실행 시  
`ModuleNotFoundError: No module named 'numpy._core'`  
**원인**: 전처리 스크립트를 시스템 Python(numpy 2.3.4)으로 실행  
→ stockmixer conda env(numpy 1.22.1)와 호환 불가

```bash
# 틀린 방법 (시스템 Python)
python3 build_sp500_data.py

# 올바른 방법 (stockmixer conda env)
conda run -n stockmixer python build_sp500_data.py
```

**규칙**: 전처리, 학습, 평가 스크립트 모두 반드시 동일한 conda env로 실행

---

## Trial-10: argparse invalid choice 오류

**현상**: institutional/board/supply_chain 실험 전부 fail  
**오류**: `invalid choice: 'institutional' (choose from 'wikidata', 'sector_industry')`  
**원인**: 새 KG 소스를 코드에는 추가했는데 argparse choices 리스트를 업데이트 안 함

```python
# 버그 (기존)
parser.add_argument('--kg_source', type=str, default='wikidata',
    choices=['wikidata', 'sector_industry'])

# 수정
parser.add_argument('--kg_source', type=str, default='wikidata',
    choices=['wikidata', 'sector_industry', 'institutional', 'board', 'supply_chain', 'llm', 'llm_v2'])
```

**교훈**: 새 KG를 추가할 때 체크리스트:
1. npy 파일 생성
2. `config.py`에 경로 등록
3. `main.py` argparse choices에 추가

---

## Trial-11: SLURM job 한도 초과

**현상**: `sbatch` 시 `AssocMaxSubmitJobLimit` 오류  
**원인**: 동시 제출 가능 job 수 초과  

```bash
# 현재 큐 확인
squeue -u pz29075 --noheader | wc -l

# 완료된 job 후 나머지 제출
bash submit_remaining.sh
```

---

## Trial-12: LLM API 키 git 노출 위험

**현상**: `run_llm_kg.sh`에 `export OPENAI_API_KEY="sk-..."` 하드코딩 → git push 전 위험  
**해결**: push 전 키를 주석 처리로 교체

```bash
# run_llm_kg.sh — 절대 커밋하지 말 것
# export OPENAI_API_KEY="sk-proj-..."

# 올바른 방법: 환경변수로 전달
export OPENAI_API_KEY="sk-..."
sbatch run_llm_kg.sh
```
