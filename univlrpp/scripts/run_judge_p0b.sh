#!/bin/bash
# dsw3 CPU: gpt-6-astra judge pass on P0b ckpt_last (update 1000) HRBench4K predictions (same as run_judge_p0f.sh).
source /mnt/cpfs/jieduan/univlr/experiments/p0p3_20261105/code/runtime_env.sh
R=/mnt/cpfs/jieduan/univlr; E=$R/experiments/p0_20261111/eval; L=$E/logs_judge
export JUDGE_BACKEND=anthropic JUDGE_MODEL=gpt-6-astra LMUData=$R/Datasets/LMUData CUDA_VISIBLE_DEVICES=
export PYTHONPATH=$R/UniVLRplus/VLMEvalKit:$R/tools006:${PYTHONPATH:-}
cd $R/UniVLRplus/VLMEvalKit
$P $R/tools006/compare_judge.py $E/outputs/p0b1000_u HRBench4K > $L/p0b1000_u.log 2>&1 \
  && echo "$(date '+%F %T') DONE p0b1000_u" >> $L/status || echo "$(date '+%F %T') FAIL p0b1000_u" >> $L/status
