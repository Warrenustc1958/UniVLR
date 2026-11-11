#!/bin/bash
# dsw3 PPU 0-7: gates2.py ask3 (layout follow-up + fixed text bbox), 8 shards; smoke 1 item first; re-hold each card.
source /mnt/cpfs/jieduan/univlr/experiments/p0p3_20261105/code/runtime_env.sh
R=/mnt/cpfs/jieduan/univlr; X=$R/experiments/p0_20261111; C=$X/code; L=$X/gates/ask3_logs; CTL=$R/tmp/ppu_hold_dsw3
mkdir -p $L
log() { echo "$(date '+%F %T') $*" >> $L/status; }
used() { ppu-smi | grep MiB | sed -n "$(($1 + 1))p" | awk -F'|' '{split($3,a,"MiB"); gsub(/ /,"",a[1]); print a[1]}'; }
alive() { [ -f $CTL/hold_$1.pid ] && [ -d /proc/$(cat $CTL/hold_$1.pid) ]; }
hold() { alive $1 && return 0; rm -f $CTL/release_$1
  CUDA_VISIBLE_DEVICES=$1 setsid nohup $P $R/univlr006/ppu_hold.py --dev-label $1 --ctl $CTL > $CTL/hold_$1.log 2>&1 < /dev/null & }
trap 'for d in $(seq 0 7); do hold $d; done; sleep 60; log "REHELD $(for d in $(seq 0 15); do alive $d && echo -n "$d "; done)"' EXIT
for d in $(seq 0 7); do alive $d && touch $CTL/release_$d; done
for t in $(seq 1 60); do busy=0; for d in $(seq 0 7); do [ "$(used $d)" -lt 4000 ] || busy=1; done; [ $busy = 0 ] && break; sleep 5; done
[ $busy = 0 ] || { log RELEASE_TIMEOUT; exit 1; }
log RELEASED
export LMUData=$R/Datasets/LMUData OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=0 timeout 1800 $P $C/gates2.py ask3 --limit 1 > $L/smoke.log 2>&1
grep -q ASK3_DONE $L/smoke.log || { log "SMOKE_FAIL $(grep -a -v -E 'hggcrt|ALINPU' $L/smoke.log | grep -a -E 'Error|assert' | tail -n 2 | tr '\n' ' ' | cut -c1-300)"; exit 1; }
log "SMOKE_OK $(grep -a '^ASK3 ' $L/smoke.log | cut -c1-300)"
for s in $(seq 0 7); do
  ( CUDA_VISIBLE_DEVICES=$s timeout 5400 $P $C/gates2.py ask3 --shard $s --nshard 8 > $L/shard$s.log 2>&1 < /dev/null
    grep -q ASK3_DONE $L/shard$s.log && log "DONE shard$s" || log "FAIL shard$s $(grep -a -v -E 'hggcrt|ALINPU' $L/shard$s.log | grep -a -E 'Error|assert' | tail -n 2 | tr '\n' ' ' | cut -c1-250)"
    hold $s ) < /dev/null &
done
wait
$P $C/gates2.py report > $L/report.log 2>&1 && log REPORTED || log REPORT_FAIL
log ALL_DONE
