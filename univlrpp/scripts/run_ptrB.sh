#!/bin/bash
# dsw3, all 16 PPU: Stage B pointer warm-start (ptrB.py). Release holders -> build data (CPU) -> unit tests U5-U8 on PPU 0
# (training starts only if they pass) -> train (torchrun 16) -> evalptr init (0) + evalptr ptrB (1) + e2e V*/HR4K 14 shards
# (2-15) -> report -> re-hold all 16. Launch via `bash -lc`.
source /mnt/cpfs/jieduan/univlr/experiments/p0p3_20261105/code/runtime_env.sh
R=/mnt/cpfs/jieduan/univlr; X=$R/experiments/ptrB_20261112; C=$R/experiments/p0_20261111/code; L=$X/logs; CTL=$R/tmp/ppu_hold_dsw3
mkdir -p $L
log() { echo "$(date '+%F %T') $*" >> $L/status; }
used() { ppu-smi | grep MiB | sed -n "$(($1 + 1))p" | awk -F'|' '{split($3,a,"MiB"); gsub(/ /,"",a[1]); print a[1]}'; }
alive() { [ -f $CTL/hold_$1.pid ] && [ -d /proc/$(cat $CTL/hold_$1.pid) ]; }
hold() { alive $1 && return 0; rm -f $CTL/release_$1
  CUDA_VISIBLE_DEVICES=$1 setsid nohup $P $R/univlr006/ppu_hold.py --dev-label $1 --ctl $CTL > $CTL/hold_$1.log 2>&1 < /dev/null & }
trap 'for d in $(seq 0 15); do hold $d; done; sleep 60; log "REHELD $(for d in $(seq 0 15); do alive $d && echo -n "$d "; done)"' EXIT
export LMUData=$R/Datasets/LMUData OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
B=$C/ptrB.py
log START
if [ ! -f $X/data/train.json ]; then
  CUDA_VISIBLE_DEVICES= $P $B build > $L/build.log 2>&1 && log "BUILT $(grep -a '^BUILD' $L/build.log | cut -c1-600)" || { log "BUILD_FAIL $(tail -n 3 $L/build.log | tr '\n' ' ' | cut -c1-300)"; exit 1; }
fi
for d in $(seq 0 15); do alive $d && touch $CTL/release_$d; done
for t in $(seq 1 60); do busy=0; for d in $(seq 0 15); do [ "$(used $d)" -lt 4000 ] || busy=1; done; [ $busy = 0 ] && break; sleep 5; done
[ $busy = 0 ] || { log RELEASE_TIMEOUT; exit 1; }
log RELEASED
CUDA_VISIBLE_DEVICES=0 timeout 1800 $P $B unit > $L/unit.log 2>&1
grep -q UNIT_DONE $L/unit.log || { log "UNIT_FAIL $(grep -a -E '^UNIT|Error|assert' $L/unit.log | tail -n 3 | tr '\n' ' ' | cut -c1-900)"; exit 1; }
log "UNIT_OK $(grep -a '^UNIT ' $L/unit.log | cut -c1-900)"
log TRAIN_START
timeout 14400 $P -m torch.distributed.run --nproc-per-node 16 --master-port 29651 $B train --accum 4 --workers 4 > $L/train.log 2>&1
if grep -q TRAIN_DONE $L/train.log && [ -f $X/run/ptrB.pt ]; then log "TRAIN_DONE $(grep -a -E '^(TRAIN ptrB|HOLD after|TRAIN_DONE)' $L/train.log | tr '\n' ' ' | cut -c1-900)"
else log "TRAIN_FAIL $(grep -a -v -E 'hggcrt|ALINPU|NCCL INFO' $L/train.log | grep -a -E 'Error|error' | tail -n 3 | tr '\n' ' ' | cut -c1-400)"; exit 1; fi
run() { g=$1; n=$2; mk=$3; shift 3
  ( CUDA_VISIBLE_DEVICES=$g timeout 7200 "$@" > $L/$n.log 2>&1 < /dev/null
    grep -q $mk $L/$n.log && log "DONE $n on $g" || log "FAIL $n on $g $(grep -a -v -E 'hggcrt|ALINPU' $L/$n.log | grep -a -E 'Error|assert' | tail -n 2 | tr '\n' ' ' | cut -c1-250)"
    hold $g ) < /dev/null & }
run 0 evalptr_init EVALPTR_DONE $P $B evalptr --ckpt init
run 1 evalptr_B EVALPTR_DONE $P $B evalptr --ckpt $X/run/ptrB.pt
for s in $(seq 0 13); do run $((s + 2)) e2e$s E2E_DONE $P $B e2e --shard $s --nshard 14; done
wait
$P $B report > $L/report.log 2>&1 && log REPORTED || log REPORT_FAIL
log ALL_DONE
