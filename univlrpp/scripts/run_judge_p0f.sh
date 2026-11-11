#!/bin/bash
# dsw3 CPU: LLM-judge pass (VLMEvalKit build_prompt, Anthropic /v1/messages, gpt-6-astra) on HRBench4K / HRBench8K /
# MME-RealWorld-Lite predictions of p0f_u, then the same for s1_official and qwen_instruct (fair comparison).
# compare_judge.py columns: exact_match (can_infer only), vlmeval+llm (judge only where can_infer fails = official
# pipeline with a judge), llm_all (judge on every row). Per-row accuracy (HRBench not averaged over cyclic orders).
source /mnt/cpfs/jieduan/univlr/experiments/p0p3_20261105/code/runtime_env.sh
R=/mnt/cpfs/jieduan/univlr; E=$R/experiments/p0_20261111/eval; L=$E/logs_judge; mkdir -p $L
export JUDGE_BACKEND=anthropic JUDGE_MODEL=gpt-6-astra LMUData=$R/Datasets/LMUData CUDA_VISIBLE_DEVICES=
export PYTHONPATH=$R/UniVLRplus/VLMEvalKit:$R/tools006:${PYTHONPATH:-}
cd $R/UniVLRplus/VLMEvalKit
log() { echo "$(date '+%F %T') $*" >> $L/status; }
for w in "p0f_u $E/outputs/p0f_u" "s1_official $R/experiments/p0_20261111/step0/outputs/s1_official" "qwen_instruct $R/experiments/p0_20261111/step0/outputs/qwen_instruct"; do
  set -- $w; log "JUDGE $1"
  $P $R/tools006/compare_judge.py $2 HRBench4K HRBench8K MME-RealWorld-Lite > $L/$1.log 2>&1 && log "DONE $1" || log "FAIL $1"
done
log ALL_DONE
