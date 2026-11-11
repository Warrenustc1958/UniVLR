#!/bin/bash
# dsw3 P0 full training: all of Monet Visual_CoT minus the 500-sample holdout (118,054 samples; UniVLR-Stage1 trained on
# the same 118,561-row Visual_CoT subset), 1 epoch, global batch 64 (16 PPU x accum 4), lr 1e-5 cosine, warmup 5%,
# K=384, mask anneal 0 -> 1 over the first 10%, canvas slots p=0.5. Releases the 16 dsw3 holders, re-holds them at exit.
# Output: experiments/p0_20261111/full/{ckpt, ckpt_last, logs/train.jsonl}. Launch through `bash -lc`.
source /mnt/cpfs/jieduan/univlr/experiments/p0p3_20261105/code/runtime_env.sh
R=/mnt/cpfs/jieduan/univlr; X=$R/experiments/p0_20261111; C=$X/code; CTL=$R/tmp/ppu_hold_dsw3; L=$X/full/logs
mkdir -p $L
log() { echo "$(date '+%F %T') $*" >> $L/status; }
used() { ppu-smi | grep MiB | sed -n "$(($1 + 1))p" | awk -F'|' '{split($3,a,"MiB"); gsub(/ /,"",a[1]); print a[1]}'; }
alive() { [ -f $CTL/hold_$1.pid ] && [ -d /proc/$(cat $CTL/hold_$1.pid) ]; }
hold() { alive $1 && return 0; rm -f $CTL/release_$1
  CUDA_VISIBLE_DEVICES=$1 setsid nohup $P $R/univlr006/ppu_hold.py --dev-label $1 --ctl $CTL > $CTL/hold_$1.log 2>&1 < /dev/null & }
rehold_all() { for d in $(seq 0 15); do hold $d; done; sleep 60; log "REHELD $(for d in $(seq 0 15); do alive $d && echo -n "$d "; done)"; }
trap rehold_all EXIT
log TRAIN_START
[ -d $X/full/ckpt ] && { log "CKPT_EXISTS refusing to overwrite"; exit 1; }
for d in $(seq 0 15); do alive $d && touch $CTL/release_$d; done
for t in $(seq 1 60); do busy=0; for d in $(seq 0 15); do [ "$(used $d)" -lt 4000 ] || busy=1; done; [ $busy = 0 ] && break; sleep 5; done
[ $busy = 0 ] || { log RELEASE_TIMEOUT; exit 1; }
log RELEASED
timeout 36000 $P -m torch.distributed.run --nproc-per-node 16 --master-port 29613 $C/p0.py train --sub full \
  --accum 4 --workers 4 --log-every 10 --save-every 500 > $L/train.log 2>&1
if grep -q TRAIN_DONE $L/train.log && [ -f $X/full/ckpt/config.json ]; then
  log "TRAIN_DONE $(grep -a -e '^TRAIN P0' -e '^HOLD' -e TRAIN_DONE $L/train.log | tr '\n' ' ' | cut -c1-900)"
else log "TRAIN_FAIL $(grep -a -v -E 'hggcrt|ALINPU' $L/train.log | grep -a -E 'Error|error|assert' | tail -n 4 | tr '\n' ' ' | cut -c1-500)"; fi
log ALL_DONE
