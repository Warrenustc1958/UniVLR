#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# Judge envs are intentionally unset by default for anonymous release.
export AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:-}"
export AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-}"
export AZURE_OPENAI_DEPLOYMENT_NAME="${AZURE_OPENAI_DEPLOYMENT_NAME:-}"
export OPENAI_API_VERSION="${OPENAI_API_VERSION:-}"
export LMUData="${LMUData:-${REPO_DIR}/data/UniVLR_Bench}"

CONFIG="${CONFIG:-config/vanilla_config.json}"
WORK_DIR="${WORK_DIR:-./outputs}"
MODE="${MODE:-all}"            # all | infer | eval
USE_VLLM="${USE_VLLM:-1}"      # 1/0
VERBOSE="${VERBOSE:-1}"        # 1/0
REUSE="${REUSE:-1}"            # 1/0
REUSE_AUX="${REUSE_AUX:-1}"    # 1/0
IGNORE_FAILED="${IGNORE_FAILED:-0}"  # 1/0, skip failed samples when resuming
JUDGE="${JUDGE:-${AZURE_OPENAI_DEPLOYMENT_NAME}}"
JUDGE_ARGS="${JUDGE_ARGS:-{}}"
CHECK_AZURE_API="${CHECK_AZURE_API:-0}"  # 1/0, quick judge API preflight before run
EXTRA_ARGS="${EXTRA_ARGS:-}"   # e.g. '--data VStarBench --mode eval'

cd "${REPO_DIR}"

if [[ "${CHECK_AZURE_API}" == "1" ]]; then
    if [[ -z "${AZURE_OPENAI_API_KEY}" || -z "${AZURE_OPENAI_ENDPOINT}" || -z "${AZURE_OPENAI_DEPLOYMENT_NAME}" || -z "${OPENAI_API_VERSION}" ]]; then
        echo "Azure env is incomplete. Please set AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_DEPLOYMENT_NAME / OPENAI_API_VERSION."
        exit 1
    fi
    echo "Running Azure judge API preflight..."
    export JUDGE
    python - <<'PY'
import os
import sys
from vlmeval.api import OpenAIWrapper

judge = os.environ.get("JUDGE") or os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
try:
    model = OpenAIWrapper(
        model=judge,
        use_azure=True,
        retry=1,
        timeout=60,
        max_tokens=16,
        verbose=True,
    )
    code, answer, _ = model.generate_inner([{"type": "text", "value": "Reply with OK only."}])
    if code != 0:
        print(f"[API CHECK] failed, code={code}, answer={answer}")
        sys.exit(1)
    print("[API CHECK] success")
except Exception as e:
    print(f"[API CHECK] exception: {e}")
    sys.exit(1)
PY
fi

cmd=(python run.py --config "${CONFIG}" --work-dir "${WORK_DIR}" --mode "${MODE}" --judge-args "${JUDGE_ARGS}")

if [[ "${USE_VLLM}" == "1" ]]; then
    cmd+=(--use-vllm)
fi
if [[ "${VERBOSE}" == "1" ]]; then
    cmd+=(--verbose)
fi
if [[ "${REUSE}" == "1" ]]; then
    cmd+=(--reuse --reuse-aux "${REUSE_AUX}")
fi
if [[ "${IGNORE_FAILED}" == "1" ]]; then
    cmd+=(--ignore)
fi
if [[ -n "${JUDGE}" ]]; then
    cmd+=(--judge "${JUDGE}")
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    cmd+=(${EXTRA_ARGS})
fi

printf 'Running command:\n  '
printf '%q ' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
