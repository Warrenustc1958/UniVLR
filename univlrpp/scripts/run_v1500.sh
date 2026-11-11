#!/bin/bash
# dsw4, shared with WM Safety (no holders): V* of P0 ckpt_last (update 1500) on ONE card among PPU 0-5 8 9 (10-13 never
# used) with the most free memory (>= 32 GB): p0s1500_m (answer sees the image only through the slots) then p0s1500_u
# (image visible), then score_official.py. Never kills or releases anything.
source /mnt/cpfs/jieduan/univlr/experiments/p0p3_20261105/code/runtime_env.sh
R=/mnt/cpfs/jieduan/univlr; X=$R/experiments/p0_20261111; C=$X/code; E=$X/eval; L=$E/logs_v1500; CK=$X/full/ckpt_last
mkdir -p $L $E/cfg $E/outputs
log() { echo "$(date '+%F %T') $*" >> $L/status; }
free_mb() { ppu-smi | grep MiB | sed -n "$(($1 + 1))p" | awk -F'|' '{split($3,a,"/"); u=a[1]; t=a[2]; gsub(/[ MiB]/,"",u); gsub(/[ MiB]/,"",t); print t-u}'; }
[ -f $CK/config.json ] || { log NO_CKPT; exit 1; }
g=; best=0
for d in 0 1 2 3 4 5 8 9; do f=$(free_mb $d); [ "$f" -gt "$best" ] && { best=$f; g=$d; }; done
[ "$best" -ge 32000 ] || { log "NO_FREE_CARD best $best MiB"; exit 1; }
log "CARD $g free $best MiB"
export LMUData=$R/Datasets/LMUData OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false VLMEVAL_EVAL_ID=E20261111
for cm in "p0s1500_m true" "p0s1500_u false"; do set -- $cm; c=$1; mk=$2; j=${c}__VStarBench
  echo "{\"model\": {\"$c\": {\"class\": \"P0VLM\", \"model_path\": \"$CK\", \"masked\": $mk}}, \"data\": {\"VStarBench\": {\"class\": \"ImageMCQDataset\", \"dataset\": \"VStarBench\"}}}" > $E/cfg/$j.json
  log "RUN $j on $g"; t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$g timeout 3600 $P $C/run_vlm_p0.py --config $E/cfg/$j.json --work-dir $E/outputs --mode all --judge exact_matching --reuse > $L/$j.log 2>&1 < /dev/null
  if [ -f $E/outputs/$c/E20261111/${c}_VStarBench.pkl ]; then log "DONE $j $(( $(date +%s) - t0 ))s"
  else log "FAIL $j $(grep -a -v -E 'hggcrt|ALINPU' $L/$j.log | tail -n 3 | tr '\n' ' ' | cut -c1-300)"; fi
done
$P $R/experiments/p1_20261110/code/score_official.py > $L/score.log 2>&1
log "SCORED $(grep -a -E '^\| p0s1500_[mu] \|' $R/experiments/eval_20261110/report.md | head -n 2 | tr '\n' ' ' | cut -c1-300)"
log ALL_DONE
