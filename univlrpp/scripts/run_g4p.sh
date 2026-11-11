#!/bin/bash
# dsw3 PPU 0 1: G4' (g4p.py). Release holders 0 1 -> point (P0b, 2 shards) -> ask (Qwen-Instruct, 2 shards) -> report
# -> re-hold 0 1. Launch via `bash -lc`.
source /mnt/cpfs/jieduan/univlr/experiments/p0p3_20261105/code/runtime_env.sh
R=/mnt/cpfs/jieduan/univlr; X=$R/experiments/p0_20261111; C=$X/code; L=$X/gates/g4p/logs; CTL=$R/tmp/ppu_hold_dsw3
mkdir -p $L
log() { echo "$(date '+%F %T') $*" >> $L/status; }
used() { ppu-smi | grep MiB | sed -n "$(($1 + 1))p" | awk -F'|' '{split($3,a,"MiB"); gsub(/ /,"",a[1]); print a[1]}'; }
alive() { [ -f $CTL/hold_$1.pid ] && [ -d /proc/$(cat $CTL/hold_$1.pid) ]; }
hold() { alive $1 && return 0; rm -f $CTL/release_$1
  CUDA_VISIBLE_DEVICES=$1 setsid nohup $P $R/univlr006/ppu_hold.py --dev-label $1 --ctl $CTL > $CTL/hold_$1.log 2>&1 < /dev/null & }
trap 'for d in 0 1; do hold $d; done; sleep 60; log "REHELD $(for d in $(seq 0 15); do alive $d && echo -n "$d "; done)"' EXIT
for d in 0 1; do alive $d && touch $CTL/release_$d; done
for t in $(seq 1 60); do busy=0; for d in 0 1; do [ "$(used $d)" -lt 4000 ] || busy=1; done; [ $busy = 0 ] && break; sleep 5; done
[ $busy = 0 ] || { log RELEASE_TIMEOUT; exit 1; }
log RELEASED
export LMUData=$R/Datasets/LMUData OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
for ph in point ask; do
  for s in 0 1; do CUDA_VISIBLE_DEVICES=$s timeout 5400 $P $C/g4p.py $ph --shard $s --nshard 2 > $L/${ph}$s.log 2>&1 & done
  wait
  for s in 0 1; do grep -q -E 'POINT_DONE|ASK_DONE' $L/${ph}$s.log || { log "FAIL $ph$s $(grep -a -v -E 'hggcrt|ALINPU' $L/${ph}$s.log | grep -a -E 'Error|assert' | tail -n 2 | tr '\n' ' ' | cut -c1-300)"; exit 1; }; done
  log "DONE $ph"
done
$P $C/g4p.py report > $L/report.log 2>&1 && log REPORTED || log REPORT_FAIL
log ALL_DONE
