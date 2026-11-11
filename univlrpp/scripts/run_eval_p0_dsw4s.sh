#!/bin/bash
# dsw4, shared mode (11-11 user: the WM Safety project uses PPU 0-5 8 9; P0 eval shares them, no holders, no re-hold).
# Waits for dsw3 P0 training (full/logs/status ALL_DONE + TRAIN_DONE + full/ckpt). Each job waits until its card has
# >= NEED_MB free, never touches other processes. Class smoke (4 V* items, both modes) -> 8 single-card run.py jobs
# (p0_m / p0_u x 4 datasets) -> early score after both V* jobs (VSTAR_SCORED) -> final score. PPU 10-13 never used.
source /mnt/cpfs/jieduan/univlr/experiments/p0p3_20261105/code/runtime_env.sh
R=/mnt/cpfs/jieduan/univlr; X=$R/experiments/p0_20261111; C=$X/code; E=$X/eval; L=$E/logs4s; CK=$X/full/ckpt
S=$R/experiments/p1_20261110/code/score_official.py; NEED_MB=32000
mkdir -p $L $E/cfg $E/outputs
log() { echo "$(date '+%F %T') $*" >> $L/status; }
free_mb() { ppu-smi | grep MiB | sed -n "$(($1 + 1))p" | awk -F'|' '{split($3,a,"/"); u=a[1]; t=a[2]; gsub(/[ MiB]/,"",u); gsub(/[ MiB]/,"",t); print t-u}'; }
wait_free() { local n=0; while [ "$(free_mb $1)" -lt $NEED_MB ]; do [ $((n % 20)) = 0 ] && log "WAIT_MEM $1 free $(free_mb $1) MiB"; n=$((n + 1)); sleep 30; done; }
dsclass() { case $1 in VStarBench) echo ImageMCQDataset;; HRBench*) echo HRBenchDataset;; *) echo MMERealWorld;; esac; }
log WAIT_TRAIN
while ! grep -q ALL_DONE $X/full/logs/status 2>/dev/null; do sleep 60; done
{ grep -q TRAIN_DONE $X/full/logs/status && [ -f $CK/config.json ]; } || { log ABORT_NO_CKPT; exit 1; }
export LMUData=$R/Datasets/LMUData OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false VLMEVAL_EVAL_ID=E20261111
wait_free 0; log "CLASS_SMOKE on 0 (free $(free_mb 0) MiB)"
CUDA_VISIBLE_DEVICES=0 timeout 1800 $P $C/p0vlm.py --smoke $CK > $L/class_smoke.log 2>&1
grep -q SMOKE_OK $L/class_smoke.log || { log "CLASS_SMOKE_FAIL $(grep -a -v -E 'hggcrt|ALINPU' $L/class_smoke.log | tail -n 3 | tr '\n' ' ' | cut -c1-300)"; exit 1; }
log "CLASS_SMOKE_OK $(grep -a '^SMOKE masked' $L/class_smoke.log | tr '\n' ' ' | cut -c1-900)"
JOBS="0 p0_m true VStarBench
1 p0_u false VStarBench
2 p0_m true MME-RealWorld-Lite
3 p0_u false MME-RealWorld-Lite
4 p0_m true HRBench8K
5 p0_u false HRBench8K
8 p0_m true HRBench4K
9 p0_u false HRBench4K"
while read g c mk d; do
  j=${c}__${d}
  echo "{\"model\": {\"$c\": {\"class\": \"P0VLM\", \"model_path\": \"$CK\", \"masked\": $mk}}, \"data\": {\"$d\": {\"class\": \"$(dsclass $d)\", \"dataset\": \"$d\"}}}" > $E/cfg/$j.json
  ( wait_free $g; log "RUN $j on $g (free $(free_mb $g) MiB)"; t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=$g $P $C/run_vlm_p0.py --config $E/cfg/$j.json --work-dir $E/outputs --mode all --judge exact_matching --reuse > $L/$j.log 2>&1 < /dev/null
    if [ -f $E/outputs/$c/E20261111/${c}_${d}.pkl ]; then log "DONE $j on $g $(( $(date +%s) - t0 ))s"
    else log "FAIL $j on $g $(grep -a -v -E 'hggcrt|ALINPU' $L/$j.log | tail -n 2 | tr '\n' ' ' | cut -c1-200)"; fi ) < /dev/null &
done <<< "$JOBS"
( while [ "$(grep -c -E '(DONE|FAIL) p0_[mu]__VStarBench' $L/status)" -lt 2 ]; do sleep 20; done
  $P $S > $L/score_vstar.log 2>&1; log "VSTAR_SCORED $(grep -a -E '^\| p0_[mu] \|' $R/experiments/eval_20261110/report.md | head -n 2 | tr '\n' ' ' | cut -c1-400)" ) < /dev/null &
wait
$P $S > $L/score.log 2>&1
log "SCORED $(grep -c '^SCORED' $L/score.log) new jobs, $(grep -c '^SCORE_FAIL' $L/score.log) failed"
log ALL_DONE
