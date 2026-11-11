#!/bin/bash
# dsw3 PPU 6: g0probe.py (raw official-wrapper generations, latent token counts). Release holder 6 -> run -> re-hold 6.
source /mnt/cpfs/jieduan/univlr/experiments/p0p3_20261105/code/runtime_env.sh
R=/mnt/cpfs/jieduan/univlr; C=$R/experiments/p0_20261111/code; L=$R/experiments/p0_20261111/gates/gates2_logs; CTL=$R/tmp/ppu_hold_dsw3
alive() { [ -f $CTL/hold_$1.pid ] && [ -d /proc/$(cat $CTL/hold_$1.pid) ]; }
touch $CTL/release_6; for t in $(seq 1 30); do alive 6 || break; sleep 2; done; sleep 5
export LMUData=$R/Datasets/LMUData UNIVLR_ROOT=$R/UniVLRplus PYTHONPATH=$R/UniVLRplus/VLMEvalKit:$R/UniVLRplus:${PYTHONPATH:-}
cd $R/UniVLRplus/VLMEvalKit
CUDA_VISIBLE_DEVICES=6 timeout 2400 $P $C/g0probe.py > $L/g0probe.log 2>&1
rm -f $CTL/release_6
CUDA_VISIBLE_DEVICES=6 setsid nohup $P $R/univlr006/ppu_hold.py --dev-label 6 --ctl $CTL > $CTL/hold_6.log 2>&1 < /dev/null &
echo PROBE_FINISHED > $L/g0probe.done
