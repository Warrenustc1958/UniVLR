#!/bin/bash
# dsw3 P0 smoke: build data (CPU) -> release the 16 dsw3 holders -> NCCL all_reduce check -> p0.py smoke on 16 PPUs
# (2048 samples, accum 4 = global batch 64, 32 updates; selftest + holdout CE before/after) -> re-hold all 16 cards.
# Launch through `bash -lc` (NCCL needs the DSW login profile env on dsw3).
source /mnt/cpfs/jieduan/univlr/experiments/p0p3_20261105/code/runtime_env.sh
R=/mnt/cpfs/jieduan/univlr; X=$R/experiments/p0_20261111; C=$X/code; CTL=$R/tmp/ppu_hold_dsw3
mkdir -p $X/logs
log() { echo "$(date '+%F %T') $*" >> $X/logs/status; }
used() { ppu-smi | grep MiB | sed -n "$(($1 + 1))p" | awk -F'|' '{split($3,a,"MiB"); gsub(/ /,"",a[1]); print a[1]}'; }
alive() { [ -f $CTL/hold_$1.pid ] && [ -d /proc/$(cat $CTL/hold_$1.pid) ]; }
hold() { alive $1 && return 0; rm -f $CTL/release_$1
  CUDA_VISIBLE_DEVICES=$1 setsid nohup $P $R/univlr006/ppu_hold.py --dev-label $1 --ctl $CTL > $CTL/hold_$1.log 2>&1 < /dev/null & }
rehold_all() { for d in $(seq 0 15); do hold $d; done; sleep 60; log "REHELD $(for d in $(seq 0 15); do alive $d && echo -n "$d "; done)"; }
trap rehold_all EXIT
log SMOKE_START
if [ ! -f $X/data/train.json ]; then
  $P $C/p0.py build > $X/logs/build.log 2>&1 && log "BUILT $(grep -a '^BUILD' $X/logs/build.log | cut -c1-200)" || { log BUILD_FAIL; exit 1; }
fi
for d in $(seq 0 15); do alive $d && touch $CTL/release_$d; done
for t in $(seq 1 60); do busy=0; for d in $(seq 0 15); do [ "$(used $d)" -lt 4000 ] || busy=1; done; [ $busy = 0 ] && break; sleep 5; done
[ $busy = 0 ] || { log RELEASE_TIMEOUT; exit 1; }
log RELEASED
cat > $X/logs/nccl_check.py <<'EOF'
import os, torch, torch.distributed as dist
dist.init_process_group('nccl'); r = dist.get_rank(); torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
x = torch.full((1 << 20,), float(r + 1), device='cuda'); dist.all_reduce(x); torch.cuda.synchronize()
assert abs(float(x[0]) - sum(range(1, dist.get_world_size() + 1))) < 1e-3, float(x[0])
if r == 0: print('NCCL_OK', dist.get_world_size(), float(x[0]), flush=True)
dist.destroy_process_group()
EOF
timeout 600 $P -m torch.distributed.run --nproc-per-node 16 --master-port 29611 $X/logs/nccl_check.py > $X/logs/nccl.log 2>&1
grep -q NCCL_OK $X/logs/nccl.log && log "NCCL_OK" || { log "NCCL_FAIL $(grep -a -v -E 'hggcrt|ALINPU' $X/logs/nccl.log | tail -n 3 | tr '\n' ' ' | cut -c1-300)"; exit 1; }
log SMOKE_RUN
timeout 7200 $P -m torch.distributed.run --nproc-per-node 16 --master-port 29612 $C/p0.py smoke --sub smoke \
  --max-train 2048 --accum 4 --workers 4 --log-every 2 > $X/logs/smoke.log 2>&1
if grep -q TRAIN_DONE $X/logs/smoke.log; then log "SMOKE_DONE $(grep -a -e '^SELFTEST' -e '^HOLD' -e TRAIN_DONE $X/logs/smoke.log | tr '\n' ' ' | cut -c1-900)"
else log "SMOKE_FAIL $(grep -a -v -E 'hggcrt|ALINPU' $X/logs/smoke.log | grep -a -E 'Error|error|SELFTEST|assert' | tail -n 4 | tr '\n' ' ' | cut -c1-500)"; fi
log ALL_DONE
