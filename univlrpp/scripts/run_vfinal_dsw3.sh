#!/bin/bash
# dsw3: official-protocol V* of the FINAL P0 ckpt (full/ckpt, update 1844), ONE config only = the stronger mode at
# update 1500 (p0s1500_u vs p0s1500_m V* in report.md; ties -> image visible). One torchrun run.py on PPU 0-7
# (run.py shards the 191 items over 8 ranks). Releases dsw3 holders 0-7, re-holds them at exit. Launch via `bash -lc`.
source /mnt/cpfs/jieduan/univlr/experiments/p0p3_20261105/code/runtime_env.sh
R=/mnt/cpfs/jieduan/univlr; X=$R/experiments/p0_20261111; C=$X/code; E=$X/eval; L=$E/logs_vfinal; CK=$X/full/ckpt
CTL=$R/tmp/ppu_hold_dsw3; RP=$R/experiments/eval_20261110/report.md
mkdir -p $L $E/cfg $E/outputs
log() { echo "$(date '+%F %T') $*" >> $L/status; }
used() { ppu-smi | grep MiB | sed -n "$(($1 + 1))p" | awk -F'|' '{split($3,a,"MiB"); gsub(/ /,"",a[1]); print a[1]}'; }
alive() { [ -f $CTL/hold_$1.pid ] && [ -d /proc/$(cat $CTL/hold_$1.pid) ]; }
hold() { alive $1 && return 0; rm -f $CTL/release_$1
  CUDA_VISIBLE_DEVICES=$1 setsid nohup $P $R/univlr006/ppu_hold.py --dev-label $1 --ctl $CTL > $CTL/hold_$1.log 2>&1 < /dev/null & }
vstar() { awk -F'|' -v r=" $1 " '$2 == r {gsub(/ /, "", $4); print $4}' $RP | head -n 1; }
[ -f $CK/config.json ] || { log NO_CKPT; exit 1; }
log WAIT_V1500
for t in $(seq 1 120); do grep -q ALL_DONE $E/logs_v1500/status 2>/dev/null && break; sleep 30; done
m=$(vstar p0s1500_m); u=$(vstar p0s1500_u)
if [ -n "$u" ] && [ "$u" != "-" ] && awk -v a="$u" -v b="${m:-0}" 'BEGIN{exit !(a >= b)}'; then c=p0f_u; mk=false; else c=p0f_m; mk=true; fi
log "PICK $c (update 1500 V*: masked ${m:-?} / image visible ${u:-?})"
trap 'for d in 0 1 2 3 4 5 6 7; do hold $d; done; sleep 60; log "REHELD $(for d in $(seq 0 15); do alive $d && echo -n "$d "; done)"' EXIT
for d in 0 1 2 3 4 5 6 7; do alive $d && touch $CTL/release_$d; done
for t in $(seq 1 60); do busy=0; for d in 0 1 2 3 4 5 6 7; do [ "$(used $d)" -lt 4000 ] || busy=1; done; [ $busy = 0 ] && break; sleep 5; done
[ $busy = 0 ] || { log RELEASE_TIMEOUT; exit 1; }
log RELEASED
export LMUData=$R/Datasets/LMUData OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false VLMEVAL_EVAL_ID=E20261111
j=${c}__VStarBench
echo "{\"model\": {\"$c\": {\"class\": \"P0VLM\", \"model_path\": \"$CK\", \"masked\": $mk}}, \"data\": {\"VStarBench\": {\"class\": \"ImageMCQDataset\", \"dataset\": \"VStarBench\"}}}" > $E/cfg/$j.json
log "RUN $j on 0-7"; t0=$(date +%s)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 3600 $P -m torch.distributed.run --nproc-per-node 8 --master-port 29621 \
  $C/run_vlm_p0.py --config $E/cfg/$j.json --work-dir $E/outputs --mode all --judge exact_matching --reuse > $L/$j.log 2>&1 < /dev/null
if [ -f $E/outputs/$c/E20261111/${c}_VStarBench.pkl ]; then log "DONE $j $(( $(date +%s) - t0 ))s"
else log "FAIL $j $(grep -a -v -E 'hggcrt|ALINPU' $L/$j.log | grep -a -E 'Error|error' | tail -n 3 | tr '\n' ' ' | cut -c1-300)"; exit 1; fi
$P $R/experiments/p1_20261110/code/score_official.py > $L/score.log 2>&1
log "SCORED $(awk -F'|' -v r=" $c " '$2 == r' $RP | head -n 1 | cut -c1-300)"
log ALL_DONE
