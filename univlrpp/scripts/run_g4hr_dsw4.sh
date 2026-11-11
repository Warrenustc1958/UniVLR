#!/bin/bash
# dsw4 PPU 0-5 8 9 (user-authorised 11-12; 10-13 = WM Safety, never touched): G4'-HR (g4hr.py), 8 shards.
# Holders on these cards are released with release_<d> and re-held at exit with ppu_hold.sh (restores the state).
# Smoke (2 rows on PPU 0) first. Non-login shell (bash -lc hangs on dsw4).
source /mnt/cpfs/jieduan/univlr/experiments/p0p3_20261105/code/runtime_env.sh
R=/mnt/cpfs/jieduan/univlr; X=$R/experiments/p0_20261111; C=$X/code; L=$X/gates/g4hr/logs; HOLD=$R/tmp/ppu_hold
CARDS="0 1 2 3 4 5 8 9"
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
export LMUData=$R/Datasets/LMUData OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=0 timeout 1800 $P $C/g4hr.py run --shard 0 --nshard 8 --limit 2 > $L/smoke.log 2>&1
grep -q HR_DONE $L/smoke.log || { log "SMOKE_FAIL $(grep -a -v -E 'hggcrt|ALINPU' $L/smoke.log | grep -a -E 'Error|assert' | tail -n 2 | tr '\n' ' ' | cut -c1-300)"; exit 1; }
log SMOKE_OK
i=0
for g in $CARDS; do
  ( CUDA_VISIBLE_DEVICES=$g timeout 7200 $P $C/g4hr.py run --shard $i --nshard 8 > $L/shard$i.log 2>&1
    grep -q HR_DONE $L/shard$i.log && log "DONE shard$i on $g" || log "FAIL shard$i on $g $(grep -a -v -E 'hggcrt|ALINPU' $L/shard$i.log | grep -a -E 'Error|assert' | tail -n 2 | tr '\n' ' ' | cut -c1-250)"
    hold $g ) < /dev/null &
  i=$((i + 1))
done
wait
$P $C/g4hr.py report > $L/report.log 2>&1 && log REPORTED || log REPORT_FAIL
log ALL_DONE
