#!/bin/bash
# dsw4 PPU 0-5 8 9 (user-authorised; 10-13 = WM Safety, never touched). Release holders -> smoke (tbox2 2 items, g0p s2 1 item)
# -> phase 1: fair text-bbox baseline (tbox2.py: V* 2 shards on 0 1, HR4K 6 shards on 2 3 4 5 8 9) -> phase 2: G0' on Stage2
# (g0p.py --model s2: HR4K 3 shards on 0 1 2, MME-RW-Lite 5 shards on 3 4 5 8 9) -> report -> re-hold. Non-login shell.
source /mnt/cpfs/jieduan/univlr/experiments/p0p3_20261105/code/runtime_env.sh
R=/mnt/cpfs/jieduan/univlr; C=$R/experiments/p0_20261111/code; L=$R/experiments/p0_20261111/gates/dsw4_tb_g0_logs; HOLD=$R/tmp/ppu_hold
K=$R/UniVLRplus/VLMEvalKit; CARDS="0 1 2 3 4 5 8 9"
mkdir -p $L
log() { echo "$(date '+%F %T') $*" >> $L/status; }
used() { ppu-smi | grep MiB | sed -n "$(($1 + 1))p" | awk -F'|' '{split($3,a,"MiB"); gsub(/ /,"",a[1]); print a[1]}'; }
alive() { [ -f $HOLD/hold_$1.pid ] && [ -d /proc/$(cat $HOLD/hold_$1.pid) ]; }
hold() { alive $1 || bash $R/univlr006/ppu_hold.sh $1 >> $L/rehold.log 2>&1; }
trap 'for d in $CARDS; do hold $d; done; sleep 30; log "REHELD $(for d in $CARDS; do alive $d && echo -n "$d "; done)"' EXIT
for d in $CARDS; do alive $d && touch $HOLD/release_$d; done
for t in $(seq 1 60); do busy=0; for d in $CARDS; do [ "$(used $d)" -lt 4000 ] || busy=1; done; [ $busy = 0 ] && break; sleep 5; done
[ $busy = 0 ] || { log "RELEASE_TIMEOUT $(for d in $CARDS; do echo -n "$d:$(used $d) "; done)"; exit 1; }
log RELEASED
export LMUData=$R/Datasets/LMUData OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false UNIVLR_ROOT=$R/UniVLRplus
CUDA_VISIBLE_DEVICES=0 timeout 1800 $P $C/tbox2.py run --ds vstar --limit 2 > $L/smoke_tb.log 2>&1 &
( cd $K && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$K:$R/UniVLRplus:${PYTHONPATH:-} timeout 1800 $P $C/g0p.py --model s2 --ds HRBench4K --limit 1 > $L/smoke_g0.log 2>&1 ) &
wait
grep -q TB_DONE $L/smoke_tb.log || { log "SMOKE_TB_FAIL $(grep -a -v -E 'hggcrt|ALINPU' $L/smoke_tb.log | grep -a -E 'Error|assert' | tail -n 2 | tr '\n' ' ' | cut -c1-300)"; exit 1; }
grep -q G0_DONE $L/smoke_g0.log || { log "SMOKE_G0_FAIL $(grep -a -v -E 'hggcrt|ALINPU' $L/smoke_g0.log | grep -a -E 'Error|assert' | tail -n 2 | tr '\n' ' ' | cut -c1-300)"; exit 1; }
log "SMOKE_OK $(grep -a -h -E '^(TB |G0 )' $L/smoke_tb.log $L/smoke_g0.log | cut -c1-300 | tr '\n' ' ')"
job() { g=$1; n=$2; mk=$3; shift 3
  ( CUDA_VISIBLE_DEVICES=$g timeout 10800 "$@" > $L/$n.log 2>&1 < /dev/null
    grep -q $mk $L/$n.log && log "DONE $n on $g" || log "FAIL $n on $g $(grep -a -v -E 'hggcrt|ALINPU' $L/$n.log | grep -a -E 'Error|assert' | tail -n 2 | tr '\n' ' ' | cut -c1-250)" ) < /dev/null & }
log PHASE1
job 0 tb_v0 TB_DONE $P $C/tbox2.py run --ds vstar --shard 0 --nshard 2
job 1 tb_v1 TB_DONE $P $C/tbox2.py run --ds vstar --shard 1 --nshard 2
i=0; for g in 2 3 4 5 8 9; do job $g tb_h$i TB_DONE $P $C/tbox2.py run --ds hr4k --shard $i --nshard 6; i=$((i + 1)); done
wait
$P $C/tbox2.py report > $L/report1.log 2>&1; log "PHASE1_REPORTED"
log PHASE2
export PYTHONPATH=$K:$R/UniVLRplus:${PYTHONPATH:-}; cd $K
i=0; for g in 0 1 2; do job $g g0s2_hr$i G0_DONE $P $C/g0p.py --model s2 --ds HRBench4K --shard $i --nshard 3; i=$((i + 1)); done
i=0; for g in 3 4 5 8 9; do job $g g0s2_mme$i G0_DONE $P $C/g0p.py --model s2 --ds MME-RealWorld-Lite --shard $i --nshard 5; i=$((i + 1)); done
wait
$P $C/tbox2.py report > $L/report.log 2>&1 && log REPORTED || log REPORT_FAIL
log ALL_DONE
