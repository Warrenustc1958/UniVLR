#!/bin/bash
# dsw3 PPU 0-7: G1'/G2'/G3' gates (g123.py, no training). Release holders 0-7 -> smoke (qwen teach + g3 p0b, 2 items
# each, cards 0/1) -> full: qwen 0,1 | s1 2,3 | s2 4,5 (2 shards each) | g3 p0b 6 | g3 p0 7 -> report -> re-hold 0-7.
# Launch via `bash -lc`.
source /mnt/cpfs/jieduan/univlr/experiments/p0p3_20261105/code/runtime_env.sh
R=/mnt/cpfs/jieduan/univlr; X=$R/experiments/p0_20261111; C=$X/code; G=$X/gates; L=$G/logs; CTL=$R/tmp/ppu_hold_dsw3
mkdir -p $L
log() { echo "$(date '+%F %T') $*" >> $L/status; }
used() { ppu-smi | grep MiB | sed -n "$(($1 + 1))p" | awk -F'|' '{split($3,a,"MiB"); gsub(/ /,"",a[1]); print a[1]}'; }
alive() { [ -f $CTL/hold_$1.pid ] && [ -d /proc/$(cat $CTL/hold_$1.pid) ]; }
hold() { alive $1 && return 0; rm -f $CTL/release_$1
  CUDA_VISIBLE_DEVICES=$1 setsid nohup $P $R/univlr006/ppu_hold.py --dev-label $1 --ctl $CTL > $CTL/hold_$1.log 2>&1 < /dev/null & }
trap 'for d in 0 1 2 3 4 5 6 7; do hold $d; done; sleep 60; log "REHELD $(for d in $(seq 0 15); do alive $d && echo -n "$d "; done)"' EXIT
for d in 0 1 2 3 4 5 6 7; do alive $d && touch $CTL/release_$d; done
for t in $(seq 1 60); do busy=0; for d in 0 1 2 3 4 5 6 7; do [ "$(used $d)" -lt 4000 ] || busy=1; done; [ $busy = 0 ] && break; sleep 5; done
[ $busy = 0 ] || { log RELEASE_TIMEOUT; exit 1; }
log RELEASED
export LMUData=$R/Datasets/LMUData OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
G3=$C/g123.py
CUDA_VISIBLE_DEVICES=0 timeout 1800 $P $G3 teach --teacher qwen --limit 2 > $L/smoke_teach.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 timeout 1800 $P $G3 g3 --ckpt p0b --limit 2 > $L/smoke_g3.log 2>&1 &
wait
ok=1
grep -q TEACH_DONE $L/smoke_teach.log || { ok=0; log "SMOKE_TEACH_FAIL $(grep -a -v -E 'hggcrt|ALINPU' $L/smoke_teach.log | grep -a -E 'Error|assert' | tail -n 2 | tr '\n' ' ' | cut -c1-300)"; }
grep -q G3_DONE $L/smoke_g3.log || { ok=0; log "SMOKE_G3_FAIL $(grep -a -v -E 'hggcrt|ALINPU' $L/smoke_g3.log | grep -a -E 'Error|assert' | tail -n 2 | tr '\n' ' ' | cut -c1-300)"; }
[ $ok = 1 ] || exit 1
log "SMOKE_OK $(grep -a -E '^(VSTAR|VCOT|LOAD|G3_CHECK)' $L/smoke_teach.log $L/smoke_g3.log | tr '\n' ' ' | cut -c1-600)"
run() { g=$1; n=$2; shift 2; ( CUDA_VISIBLE_DEVICES=$g timeout 10800 $P $G3 "$@" > $L/$n.log 2>&1; grep -q -E 'TEACH_DONE|G3_DONE' $L/$n.log && log "DONE $n" || log "FAIL $n $(grep -a -v -E 'hggcrt|ALINPU' $L/$n.log | grep -a -E 'Error|assert' | tail -n 2 | tr '\n' ' ' | cut -c1-300)" ) < /dev/null & }
log FULL_START
run 0 qwen0 teach --teacher qwen --shard 0 --nshard 2; run 1 qwen1 teach --teacher qwen --shard 1 --nshard 2
run 2 s10 teach --teacher s1 --shard 0 --nshard 2;     run 3 s11 teach --teacher s1 --shard 1 --nshard 2
run 4 s20 teach --teacher s2 --shard 0 --nshard 2;     run 5 s21 teach --teacher s2 --shard 1 --nshard 2
run 6 g3p0b g3 --ckpt p0b;                             run 7 g3p0 g3 --ckpt p0
wait
$P $G3 report > $L/report.log 2>&1 && log REPORTED || log REPORT_FAIL
log ALL_DONE
