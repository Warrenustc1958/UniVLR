#!/bin/bash
# dsw3, all 16 PPU (user 11-12): gate suite 2. Release holders -> smoke on PPU 0-3 (ask, g3q qwen, g0p HR4K, unit full)
# -> full: ask V* 4 shards (0-3) | g3q qwen (4) | g3q p0b (5) | unit already done | g0p HR4K 4 shards (7-10) |
# g0p MME 5 shards (11-15) | PPU 6 idle -> each card re-held as soon as its job ends -> report. Launch via `bash -lc`.
source /mnt/cpfs/jieduan/univlr/experiments/p0p3_20261105/code/runtime_env.sh
R=/mnt/cpfs/jieduan/univlr; X=$R/experiments/p0_20261111; C=$X/code; L=$X/gates/gates2_logs; CTL=$R/tmp/ppu_hold_dsw3
K=$R/UniVLRplus/VLMEvalKit
mkdir -p $L
log() { echo "$(date '+%F %T') $*" >> $L/status; }
used() { ppu-smi | grep MiB | sed -n "$(($1 + 1))p" | awk -F'|' '{split($3,a,"MiB"); gsub(/ /,"",a[1]); print a[1]}'; }
alive() { [ -f $CTL/hold_$1.pid ] && [ -d /proc/$(cat $CTL/hold_$1.pid) ]; }
hold() { alive $1 && return 0; rm -f $CTL/release_$1
  CUDA_VISIBLE_DEVICES=$1 setsid nohup $P $R/univlr006/ppu_hold.py --dev-label $1 --ctl $CTL > $CTL/hold_$1.log 2>&1 < /dev/null & }
trap 'for d in $(seq 0 15); do hold $d; done; sleep 60; log "REHELD $(for d in $(seq 0 15); do alive $d && echo -n "$d "; done)"' EXIT
for d in $(seq 0 15); do alive $d && touch $CTL/release_$d; done
for t in $(seq 1 60); do busy=0; for d in $(seq 0 15); do [ "$(used $d)" -lt 4000 ] || busy=1; done; [ $busy = 0 ] && break; sleep 5; done
[ $busy = 0 ] || { log RELEASE_TIMEOUT; exit 1; }
log RELEASED
export LMUData=$R/Datasets/LMUData OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false UNIVLR_ROOT=$R/UniVLRplus
G2=$C/gates2.py; G0="$P $C/g0p.py"
CUDA_VISIBLE_DEVICES=0 timeout 1800 $P $G2 ask --limit 2 > $L/smoke_ask.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 timeout 1800 $P $G2 g3q --model qwen --limit 3 > $L/smoke_g3q.log 2>&1 &
( cd $K && CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$K:$R/UniVLRplus:${PYTHONPATH:-} timeout 1800 $G0 --ds HRBench4K --limit 2 > $L/smoke_g0.log 2>&1 ) &
CUDA_VISIBLE_DEVICES=3 timeout 1800 $P $G2 unit > $L/unit.log 2>&1 &
wait
ok=1
for f in smoke_ask:ASK_DONE smoke_g3q:G3Q_DONE smoke_g0:G0_DONE unit:UNIT_DONE; do n=${f%%:*}; k=${f##*:}
  grep -q $k $L/$n.log || { ok=0; log "SMOKE_FAIL $n $(grep -a -v -E 'hggcrt|ALINPU' $L/$n.log | grep -a -E 'Error|assert' | tail -n 2 | tr '\n' ' ' | cut -c1-300)"; }; done
[ $ok = 1 ] || exit 1
log "SMOKE_OK $(grep -a -h -E '^(ASK |G0 |UNIT )' $L/smoke_ask.log $L/smoke_g0.log $L/unit.log | cut -c1-400 | tr '\n' ' ' | cut -c1-1500)"
run() { g=$1; n=$2; mk=$3; shift 3
  ( CUDA_VISIBLE_DEVICES=$g timeout 10800 "$@" > $L/$n.log 2>&1 < /dev/null
    grep -q $mk $L/$n.log && log "DONE $n on $g" || log "FAIL $n on $g $(grep -a -v -E 'hggcrt|ALINPU' $L/$n.log | grep -a -E 'Error|assert' | tail -n 2 | tr '\n' ' ' | cut -c1-250)"
    hold $g ) < /dev/null & }
log FULL_START
for s in 0 1 2 3; do run $s ask$s ASK_DONE $P $G2 ask --shard $s --nshard 4; done
run 4 g3q_qwen G3Q_DONE $P $G2 g3q --model qwen
run 5 g3q_p0b G3Q_DONE $P $G2 g3q --model p0b
hold 6
export PYTHONPATH=$K:$R/UniVLRplus:${PYTHONPATH:-}; cd $K
for s in 0 1 2 3; do run $((7 + s)) g0hr$s G0_DONE $G0 --ds HRBench4K --shard $s --nshard 4; done
for s in 0 1 2 3 4; do run $((11 + s)) g0mme$s G0_DONE $G0 --ds MME-RealWorld-Lite --shard $s --nshard 5; done
wait
$P $G2 report > $L/report.log 2>&1 && log REPORTED || log REPORT_FAIL
log ALL_DONE
