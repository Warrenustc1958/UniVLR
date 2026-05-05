#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU UniVLR inference for VLMEvalKit.
# Default is infer-only: no vLLM, no GPT judge.
#
# Examples:
#   bash univlr_eval/eval_univlr.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash univlr_eval/eval_univlr.sh
#   NPROC_PER_NODE=4 WORK_DIR=./outputs_univlr_debug bash univlr_eval/eval_univlr.sh
#   MODEL_PATH=/path/to/checkpoint bash univlr_eval/eval_univlr.sh
#   MODEL_ALIAS=UniVLR_Stage1_univlr1 DECODING_STRATEGY=univlr UNIVLR_STEPS=1 bash univlr_eval/eval_univlr.sh
#   DRY_RUN=1 bash univlr_eval/eval_univlr.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
UNIVLR_ROOT="${UNIVLR_ROOT:-$(cd "${REPO_DIR}/.." && pwd)}"
TEMPLATE_CONFIG="${CONFIG:-config/univlr_stage1_config.json}"
WORK_DIR="${WORK_DIR:-./outputs}"
MODE="${MODE:-infer}"
LMUData="${LMUData:-${REPO_DIR}/data/UniVLR_Bench}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# UniVLR evaluation defaults. These are written into a temporary config so
# you can override them from the shell without hand-editing JSON.
MODEL_ALIAS="${MODEL_ALIAS:-UniVLR_Stage1}"
MODEL_PATH="${MODEL_PATH:-}"
DECODING_STRATEGY="${DECODING_STRATEGY:-univlr}"
UNIVLR_STEPS="${UNIVLR_STEPS:-2}"
CLEAN_UNIVLR_OUTPUT="${CLEAN_UNIVLR_OUTPUT:-true}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
CONFIG_OUT="${CONFIG_OUT:-}"
DRY_RUN="${DRY_RUN:-0}"

NPROC_PER_NODE="${NPROC_PER_NODE:-${GPUS:-}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29533}"
VLMEVAL_EVAL_ID="${VLMEVAL_EVAL_ID:-T$(date +%Y%m%d-%H%M%S)}"
DIST_TIMEOUT="${DIST_TIMEOUT:-7200}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

VERBOSE="${VERBOSE:-1}"
REUSE="${REUSE:-1}"
REUSE_AUX="${REUSE_AUX:-infer}"
KEEP_FAILED="${KEEP_FAILED:-0}"
SKIP_ERR="${SKIP_ERR:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export UNIVLR_ROOT
export LMUData
export MASTER_ADDR
export MASTER_PORT
export VLMEVAL_EVAL_ID
export DIST_TIMEOUT
export OMP_NUM_THREADS
export TOKENIZERS_PARALLELISM
export SKIP_ERR
export PYTHONPATH="${REPO_DIR}:${UNIVLR_ROOT}:${PYTHONPATH:-}"

cd "${REPO_DIR}"

if [[ ! -d "${UNIVLR_ROOT}" ]]; then
    echo "UNIVLR_ROOT does not exist: ${UNIVLR_ROOT}"
    exit 1
fi

if [[ ! -d "${LMUData}" ]]; then
    echo "LMUData does not exist: ${LMUData}"
    exit 1
fi

if [[ ! -f "${TEMPLATE_CONFIG}" ]]; then
    echo "Template config file does not exist: ${TEMPLATE_CONFIG}"
    exit 1
fi

cleanup_temp_config=0
if [[ -z "${CONFIG_OUT}" ]]; then
    CONFIG_OUT="$(mktemp "/tmp/univlr_eval_config.XXXXXX.json")"
    cleanup_temp_config=1
fi

if [[ "${cleanup_temp_config}" == "1" ]]; then
    trap 'rm -f "${CONFIG_OUT}"' EXIT
fi

export TEMPLATE_CONFIG CONFIG_OUT MODEL_ALIAS MODEL_PATH DECODING_STRATEGY UNIVLR_STEPS CLEAN_UNIVLR_OUTPUT MAX_NEW_TOKENS
"${PYTHON_BIN}" - <<'PY'
import json
import os
import sys

template_config = os.environ["TEMPLATE_CONFIG"]
config_out = os.environ["CONFIG_OUT"]
model_alias = os.environ["MODEL_ALIAS"].strip()
model_path = os.environ.get("MODEL_PATH", "").strip()
decoding_strategy = os.environ["DECODING_STRATEGY"].strip()
univlr_steps_raw = os.environ["UNIVLR_STEPS"].strip()
clean_univlr_output_raw = os.environ["CLEAN_UNIVLR_OUTPUT"].strip().lower()
max_new_tokens_raw = os.environ.get("MAX_NEW_TOKENS", "").strip()

with open(template_config, "r", encoding="utf-8") as f:
    cfg = json.load(f)

models = cfg.get("model") or {}
if not models:
    raise SystemExit(f"Template config has no model entry: {template_config}")

first_name, first_model_cfg = next(iter(models.items()))
new_model_cfg = dict(first_model_cfg)
new_model_name = model_alias or first_name

if model_path:
    new_model_cfg["model_path"] = model_path
new_model_cfg["decoding_strategy"] = decoding_strategy

if "," in univlr_steps_raw:
    univlr_steps = [int(x.strip()) for x in univlr_steps_raw.split(",") if x.strip()]
else:
    univlr_steps = int(univlr_steps_raw)
new_model_cfg["univlr_steps"] = univlr_steps
new_model_cfg["clean_univlr_output"] = clean_univlr_output_raw in {"1", "true", "yes", "y", "on"}

if max_new_tokens_raw:
    new_model_cfg["max_new_tokens"] = int(max_new_tokens_raw)

cfg["model"] = {new_model_name: new_model_cfg}

with open(config_out, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)

print(f"[CONFIG] template={template_config}")
print(f"[CONFIG] generated={config_out}")
print(f"[CONFIG] model_alias={new_model_name}")
print(f"[CONFIG] model_path={new_model_cfg.get('model_path')}")
print(f"[CONFIG] decoding_strategy={new_model_cfg.get('decoding_strategy')}")
print(f"[CONFIG] univlr_steps={new_model_cfg.get('univlr_steps')}")
print(f"[CONFIG] clean_univlr_output={new_model_cfg.get('clean_univlr_output')}")
if "max_new_tokens" in new_model_cfg:
    print(f"[CONFIG] max_new_tokens={new_model_cfg['max_new_tokens']}")
PY

if [[ -z "${NPROC_PER_NODE}" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
        NPROC_PER_NODE="${#visible_devices[@]}"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        NPROC_PER_NODE="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
    else
        NPROC_PER_NODE="1"
    fi
fi

if [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
    NPROC_PER_NODE="1"
fi

cmd=(
    torchrun
    --nproc-per-node "${NPROC_PER_NODE}"
    --master-addr "${MASTER_ADDR}"
    --master-port "${MASTER_PORT}"
    run.py
    --config "${CONFIG_OUT}"
    --work-dir "${WORK_DIR}"
    --mode "${MODE}"
)

if [[ "${VERBOSE}" == "1" ]]; then
    cmd+=(--verbose)
fi

if [[ "${REUSE}" == "1" ]]; then
    cmd+=(--reuse --reuse-aux "${REUSE_AUX}")
fi

if [[ "${KEEP_FAILED}" == "1" ]]; then
    cmd+=(--keep-failed)
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    cmd+=(${EXTRA_ARGS})
fi

echo "Running UniVLR VLMEvalKit inference"
echo "  repo: ${REPO_DIR}"
echo "  template_config: ${TEMPLATE_CONFIG}"
echo "  effective_config: ${CONFIG_OUT}"
echo "  work_dir: ${WORK_DIR}"
echo "  mode: ${MODE}"
echo "  eval_id: ${VLMEVAL_EVAL_ID}"
echo "  nproc_per_node: ${NPROC_PER_NODE}"
echo "  LMUData: ${LMUData}"
echo "  UNIVLR_ROOT: ${UNIVLR_ROOT}"
echo "  model_alias: ${MODEL_ALIAS}"
echo "  model_path: ${MODEL_PATH:-<from template>}"
echo "  decoding_strategy: ${DECODING_STRATEGY}"
echo "  univlr_steps: ${UNIVLR_STEPS}"
echo "  clean_univlr_output: ${CLEAN_UNIVLR_OUTPUT}"
if [[ -n "${MAX_NEW_TOKENS}" ]]; then
    echo "  max_new_tokens: ${MAX_NEW_TOKENS}"
fi
echo "  vLLM: disabled"
echo "  judge: disabled by default because MODE=infer"
printf 'Command:\n  '
printf '%q ' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1, not launching VLMEvalKit."
    exit 0
fi

"${cmd[@]}"
