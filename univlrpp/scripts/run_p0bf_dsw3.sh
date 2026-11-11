#!/bin/bash
# dsw3: official-protocol V*, HRBench4K, HRBench8K, MME-RealWorld-Lite of the FINAL P0b ckpt (full_b/ckpt, update 1844),
# both configs: p0bf_u (image visible to the answer) and p0bf_m (answer sees the image only through the slots).
# One torchrun run.py per (config, dataset) on PPU 0-7, sequential, V* first (early score after both V* runs).
# Releases dsw3 holders 0-7, re-holds them at exit. Launch via `bash -lc`.
source /mnt/cpfs/jieduan/univlr/experiments/p0p3_20261105/code/runtime_env.sh
R=/mnt/cpfs/jieduan/univlr; X=$R/experiments/p0_20261111; C=$X/code; E=$X/eval; L=$E/logs_p0bf; CK=$X/full_b/ckpt
CTL=$R/tmp/ppu_hold_dsw3; RP=$R/experiments/eval_20261110/report.md; S=$R/experiments/p1_20261110/code/score_official.py
mkdir -p $L $E/cfg $E/outputs
log() { echo "$(date '+%F %T') $*" >> $L/status; }
used() { ppu-smi | grep MiB | sed -n "$(($1 + 1))p" | awk -F'|' '{split($3,a,"MiB"); gsub(/ /,"",a[1]); print a[1]}'; }
alive() { [ -f $CTL/hold_$1.pid ] && [ -d /proc/$(cat $CTL/hold_$1.pid) ]; }
hold() { alive $1 && return 0; rm -f $CTL/release_$1
  CUDA_VISIBLE_DEVICES=$1 setsid nohup $P $R/univlr006/ppu_hold.py --dev-label $1 --ctl $CTL > $CTL/hold_$1.log 2>&1 < /dev/null & }
dsclass() { case $1 in VStarBench) echo ImageMCQDataset;; HRBench*) echo HRBenchDataset;; *) echo MMERealWorld;; esac; }
[ -f $CK/config.json ] || { log NO_CKPT; exit 1; }
trap 'for d in 0 1 2 3 4 5 6 7; do hold $d; done; sleep 60; log "REHELD $(for d in $(seq 0 15); do alive $d && echo -n "$d "; done)"' EXIT
for d in 0 1 2 3 4 5 6 7; do alive $d && touch $CTL/release_$d; done
for t in $(seq 1 60); do busy=0; for d in 0 1 2 3 4 5 6 7; do [ "$(used $d)" -lt 4000 ] || busy=1; done; [ $busy = 0 ] && break; sleep 5; done
[ $busy = 0 ] || { log RELEASE_TIMEOUT; exit 1; }
log RELEASED
export LMUData=$R/Datasets/LMUData OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false VLMEVAL_EVAL_ID=E20261111
port=29641
for d in VStarBench HRBench4K HRBench8K MME-RealWorld-Lite; do
  for cm in "p0bf_u false" "p0bf_m true"; do set -- $cm; c=$1; mk=$2; j=${c}__${d}
    echo "{\"model\": {\"$c\": {\"class\": \"P0VLM\", \"model_path\": \"$CK\", \"masked\": $mk}}, \"data\": {\"$d\": {\"class\": \"$(dsclass $d)\", \"dataset\": \"$d\"}}}" > $E/cfg/$j.json
    log "RUN $j on 0-7"; t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 7200 $P -m torch.distributed.run --nproc-per-node 8 --master-port $port \
      $C/run_vlm_p0.py --config $E/cfg/$j.json --work-dir $E/outputs --mode all --judge exact_matching --reuse > $L/$j.log 2>&1 < /dev/null
    if [ -f $E/outputs/$c/E20261111/${c}_${d}.pkl ]; then log "DONE $j $(( $(date +%s) - t0 ))s"
    else log "FAIL $j $(grep -a -v -E 'hggcrt|ALINPU|NCCL INFO' $L/$j.log | grep -a -E 'Error|error' | tail -n 3 | tr '\n' ' ' | cut -c1-300)"; fi
    port=$((port + 1))
  done
  if [ $d = VStarBench ]; then $P $S > $L/score_vstar.log 2>&1
    log "VSTAR_SCORED $(awk -F'|' '$2 ~ / p0bf_[mu] /' $RP | cut -c1-200 | tr '\n' ' ')"; fi
done
$P $S > $L/score.log 2>&1
log "SCORED $(awk -F'|' '$2 ~ / p0bf_[mu] /' $RP | cut -c1-260 | tr '\n' ' ')"
log ALL_DONE
