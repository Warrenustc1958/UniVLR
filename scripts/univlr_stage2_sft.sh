#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"

# Data roots.
MONET_ROOT="${MONET_ROOT:-data/Monet-SFT-125K}"
ZEBRA_STEP1_DIR="${ZEBRA_STEP1_DIR:-${MONET_ROOT}/Zebra_CoT_step1_vertical_ablation}"
VISUAL_COT_DIR="${VISUAL_COT_DIR:-${MONET_ROOT}/Visual_CoT}"
USE_OFFLINE_TARGETS="${USE_OFFLINE_TARGETS:-True}"
USE_OFFLINE_TARGETS_NORMALIZED="${USE_OFFLINE_TARGETS,,}"
IMAGE_LATENT_TOKENS="${IMAGE_LATENT_TOKENS:-24}"
UNIVLR_TARGET_RESAMPLE_MODE_RAW="${UNIVLR_TARGET_RESAMPLE_MODE:-pool_avg}"
UNIVLR_TARGET_RESAMPLE_MODE_NORMALIZED="${UNIVLR_TARGET_RESAMPLE_MODE_RAW,,}"
UNIVLR_TARGET_RESAMPLE_MODE_NORMALIZED="${UNIVLR_TARGET_RESAMPLE_MODE_NORMALIZED//-/_}"

case "${UNIVLR_TARGET_RESAMPLE_MODE_NORMALIZED}" in
    avg|avgpool|avg_pool|pool_avg|adaptive_pool_image_grid)
        UNIVLR_TARGET_RESAMPLE_MODE="pool_avg"
        DEFAULT_OFFLINE_TARGET_DIRNAME="qwen2_5_vl_latent_targets_${IMAGE_LATENT_TOKENS}token_2dpool"
        DEFAULT_OFFLINE_MANIFEST_NAME="train_offline_k${IMAGE_LATENT_TOKENS}.json"
        ;;
    mlerp|pool_mlerp)
        UNIVLR_TARGET_RESAMPLE_MODE="pool_mlerp"
        DEFAULT_OFFLINE_TARGET_DIRNAME="qwen2_5_vl_latent_targets_${IMAGE_LATENT_TOKENS}token_pool_mlerp"
        DEFAULT_OFFLINE_MANIFEST_NAME="train_offline_k${IMAGE_LATENT_TOKENS}_pool_mlerp.json"
        ;;
    *)
        echo "Unsupported UNIVLR_TARGET_RESAMPLE_MODE: ${UNIVLR_TARGET_RESAMPLE_MODE_RAW}" >&2
        echo "Supported aliases: avgpool/pool_avg, mlerp/pool_mlerp" >&2
        exit 1
        ;;
esac

OFFLINE_TARGET_DIRNAME="${OFFLINE_TARGET_DIRNAME:-${DEFAULT_OFFLINE_TARGET_DIRNAME}}"
OFFLINE_MANIFEST_NAME="${OFFLINE_MANIFEST_NAME:-${DEFAULT_OFFLINE_MANIFEST_NAME}}"

if [[ "$USE_OFFLINE_TARGETS_NORMALIZED" == "true" ]]; then
    SOURCE_DATA_REL_PATH="${OFFLINE_TARGET_DIRNAME}/${OFFLINE_MANIFEST_NAME}"
    MIXED_DATA_MODE_TAG="_offline_k${IMAGE_LATENT_TOKENS}_${UNIVLR_TARGET_RESAMPLE_MODE}"
else
    SOURCE_DATA_REL_PATH="train.json"
    MIXED_DATA_MODE_TAG="_online_${UNIVLR_TARGET_RESAMPLE_MODE}"
fi

ZEBRA_STEP1_PATH="${ZEBRA_STEP1_PATH:-${ZEBRA_STEP1_DIR}/${SOURCE_DATA_REL_PATH}}"
VISUAL_COT_PATH="${VISUAL_COT_PATH:-${VISUAL_COT_DIR}/${SOURCE_DATA_REL_PATH}}"

# Mix rule: use all Zebra_CoT_step1_vertical_ablation as the "7" part, and sample Visual_CoT
# to fill the remaining "3" part.
ZEBRA_RATIO_NUM="${ZEBRA_RATIO_NUM:-7}"
VISUAL_RATIO_NUM="${VISUAL_RATIO_NUM:-3}"
RANDOM_SEED="${RANDOM_SEED:-42}"
FORCE_REBUILD_MIXED_DATA="${FORCE_REBUILD_MIXED_DATA:-False}"
BUILD_DATASET_ONLY="${BUILD_DATASET_ONLY:-False}"

MIXED_DATA_DIR="${MIXED_DATA_DIR:-${MONET_ROOT}/UniVLR_Stage1B_Mixed}"
MIXED_DATA_PATH="${MIXED_DATA_PATH:-${MIXED_DATA_DIR}/stage1b_zebra_step1_vertical_ablation_mix${ZEBRA_RATIO_NUM}to${VISUAL_RATIO_NUM}_seed${RANDOM_SEED}${MIXED_DATA_MODE_TAG}.jsonl}"
DATA_PATH="${DATA_PATH:-${MIXED_DATA_PATH}}"

export MONET_ROOT
export ZEBRA_STEP1_PATH
export VISUAL_COT_PATH
export ZEBRA_RATIO_NUM
export VISUAL_RATIO_NUM
export RANDOM_SEED
export MIXED_DATA_PATH
export USE_OFFLINE_TARGETS_NORMALIZED

mkdir -p "${MIXED_DATA_DIR}"

build_mixed_dataset() {
    python3 - <<'PY'
import copy
import json
import os
import random

zebra_path = os.environ["ZEBRA_STEP1_PATH"]
visual_cot_path = os.environ["VISUAL_COT_PATH"]
mixed_path = os.environ["MIXED_DATA_PATH"]
seed = int(os.environ["RANDOM_SEED"])
zebra_ratio_num = int(os.environ["ZEBRA_RATIO_NUM"])
visual_ratio_num = int(os.environ["VISUAL_RATIO_NUM"])
use_offline_targets = os.environ["USE_OFFLINE_TARGETS_NORMALIZED"] == "true"

if zebra_ratio_num <= 0 or visual_ratio_num < 0:
    raise ValueError(
        f"Invalid mix ratio: zebra={zebra_ratio_num}, visual={visual_ratio_num}. "
        "Require zebra > 0 and visual >= 0."
    )


def load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def rewrite_target_paths(item, source_path, mixed_path):
    source_dir = os.path.dirname(os.path.abspath(source_path))
    mixed_dir = os.path.dirname(os.path.abspath(mixed_path))
    for step in item.get("steps", []) or item.get("latent_steps", []) or item.get("cot_steps", []):
        for key in ("target_path", "image_target_path", "text_target_path", "latent_target_path"):
            rel_path = step.get(key)
            if not rel_path:
                continue
            abs_path = rel_path if os.path.isabs(rel_path) else os.path.abspath(os.path.join(source_dir, rel_path))
            step[key] = os.path.relpath(abs_path, mixed_dir)


random.seed(seed)

zebra_rows = load_rows(zebra_path)
visual_rows = load_rows(visual_cot_path)

zebra_count = len(zebra_rows)
visual_needed = round(zebra_count * visual_ratio_num / zebra_ratio_num)
if visual_needed > len(visual_rows):
    raise ValueError(
        f"Need {visual_needed} Visual_CoT samples for mix {zebra_ratio_num}:{visual_ratio_num}, "
        f"but only {len(visual_rows)} are available."
    )

sampled_visual_rows = random.sample(visual_rows, visual_needed) if visual_needed > 0 else []

mixed_rows = []
for row in zebra_rows:
    item = copy.deepcopy(row)
    rewrite_target_paths(item, zebra_path, mixed_path)
    item.setdefault("metadata", {})
    item["metadata"]["mix_source_dataset"] = item["metadata"].get("dataset_name", "Zebra_CoT_step1_vertical_ablation")
    item["metadata"]["mix_source_manifest"] = zebra_path
    mixed_rows.append(item)
for row in sampled_visual_rows:
    item = copy.deepcopy(row)
    rewrite_target_paths(item, visual_cot_path, mixed_path)
    item.setdefault("metadata", {})
    item["metadata"]["mix_source_dataset"] = item["metadata"].get("dataset_name", "Visual_CoT")
    item["metadata"]["mix_source_manifest"] = visual_cot_path
    mixed_rows.append(item)

random.shuffle(mixed_rows)

os.makedirs(os.path.dirname(mixed_path), exist_ok=True)
with open(mixed_path, "w", encoding="utf-8") as f:
    for row in mixed_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

total = len(mixed_rows)
actual_zebra_ratio = zebra_count / total if total else 0.0
actual_visual_ratio = visual_needed / total if total else 0.0

print("Built Stage1-B mixed dataset:")
print(f"  output: {mixed_path}")
print(f"  zebra_step1_vertical_ablation: {zebra_count}")
print(f"  visual_cot_sampled: {visual_needed}")
print(f"  total: {total}")
print(f"  offline_targets: {use_offline_targets}")
print(f"  actual_ratio zebra={actual_zebra_ratio:.6f} visual={actual_visual_ratio:.6f}")
PY
}

if [[ "${DATA_PATH}" == "${MIXED_DATA_PATH}" ]]; then
    if [[ ! -f "${ZEBRA_STEP1_PATH}" ]]; then
        if [[ "$USE_OFFLINE_TARGETS_NORMALIZED" == "true" ]]; then
            echo "Missing Zebra offline manifest: ${ZEBRA_STEP1_PATH}" >&2
            echo "Please run scripts/precompute_target.sh with DATASET_ROOT=${ZEBRA_STEP1_DIR} or set USE_OFFLINE_TARGETS=False." >&2
        else
            echo "Missing Zebra data file: ${ZEBRA_STEP1_PATH}" >&2
        fi
        exit 1
    fi

    if [[ ! -f "${VISUAL_COT_PATH}" ]]; then
        if [[ "$USE_OFFLINE_TARGETS_NORMALIZED" == "true" ]]; then
            echo "Missing Visual_CoT offline manifest: ${VISUAL_COT_PATH}" >&2
            echo "Please run scripts/precompute_target.sh with DATASET_ROOT=${VISUAL_COT_DIR} or set USE_OFFLINE_TARGETS=False." >&2
        else
            echo "Missing Visual_CoT data file: ${VISUAL_COT_PATH}" >&2
        fi
        exit 1
    fi

    if [[ ! -f "${MIXED_DATA_PATH}" || "${FORCE_REBUILD_MIXED_DATA}" == "True" ]]; then
        echo "Building Stage1-B mixed dataset at ${MIXED_DATA_PATH}"
        build_mixed_dataset
    else
        echo "Reusing existing mixed dataset: ${MIXED_DATA_PATH}"
    fi
fi

if [[ "${BUILD_DATASET_ONLY}" == "True" ]]; then
    echo "BUILD_DATASET_ONLY=True, dataset is ready at: ${DATA_PATH}"
    exit 0
fi

if [[ ! -f "${DATA_PATH}" ]]; then
    echo "DATA_PATH does not exist: ${DATA_PATH}" >&2
    exit 1
fi

# Model config.
MODEL_SIZE="${MODEL_SIZE:-7B}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-${MODEL_SIZE}-Instruct}"
A1_BASE_CHECKPOINT="${A1_BASE_CHECKPOINT:-}"
A1_BASE_CHECKPOINT="${A1_BASE_CHECKPOINT:-PATH_TO_UNIVLR_STAGE1_CHECKPOINT}"

if [[ "${ONLINE_CHECKPOINT:-False}" != "True" && ! -d "${A1_BASE_CHECKPOINT}" ]]; then
    echo "A1_BASE_CHECKPOINT does not exist: ${A1_BASE_CHECKPOINT}" >&2
    exit 1
fi

export WANDB_PROJECT="${WANDB_PROJECT:-UniVLR-Qwen25-VL-${MODEL_SIZE}-UniVLR-Stage1B-Monet}"

# Training batch config.
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-1}"
if [[ -n "${NUM_DEVICES:-}" ]]; then
    NUM_DEVICES="${NUM_DEVICES}"
elif [[ -n "${WORLD_SIZE:-}" ]]; then
    NUM_DEVICES="${WORLD_SIZE}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
    NUM_DEVICES="${#GPU_LIST[@]}"
else
    NUM_DEVICES=4
fi

TOTAL_MICRO_BATCH=$((BATCH_PER_DEVICE * NUM_DEVICES))
if (( TOTAL_MICRO_BATCH <= 0 )); then
    echo "Invalid micro batch setup: BATCH_PER_DEVICE=${BATCH_PER_DEVICE}, NUM_DEVICES=${NUM_DEVICES}" >&2
    exit 1
fi
if (( GLOBAL_BATCH_SIZE % TOTAL_MICRO_BATCH != 0 )); then
    echo "GLOBAL_BATCH_SIZE (${GLOBAL_BATCH_SIZE}) must be divisible by BATCH_PER_DEVICE * NUM_DEVICES (${TOTAL_MICRO_BATCH})" >&2
    exit 1
fi
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / TOTAL_MICRO_BATCH))

# Optimization / loss config.
LR="${LR:-1e-5}"
UNIVLR_LOSS_FCT="${UNIVLR_LOSS_FCT:-mse}"
LOSS_UNIVLR_FCT="${LOSS_UNIVLR_FCT:-ln_mse_cosine}"
LAMBDA_UNIVLR="${LAMBDA_UNIVLR:-0.1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
TEXT_LOSS_LAMBDA="${TEXT_LOSS_LAMBDA:-1.0}"
IMAGE_LOSS_LAMBDA="${IMAGE_LOSS_LAMBDA:-1.0}"
TEXT_LATENT_TOKENS="${TEXT_LATENT_TOKENS:-8}"
MASK_FIRST_TOKEN_AFTER_LATENT="${MASK_FIRST_TOKEN_AFTER_LATENT:-False}"
UNIVLR_HEAD="${UNIVLR_HEAD:-True}"
UNIVLR_HEAD_TYPE="${UNIVLR_HEAD_TYPE:-simple}"
UNIVLR_HEAD_LR="${UNIVLR_HEAD_LR:-1e-4}"
UNIVLR_ALIGN_LAYER="${UNIVLR_ALIGN_LAYER:-12}"
# Stage1-B keeps the original teacher-forcing path by default. The
# univlr_stage1_b_a2_sft.sh wrapper opts into half teacher-forcing,
# half suffix self-replay.
UNIVLR_STAGE="${UNIVLR_STAGE:-a1}"
A2_REPLAY_MODE="${A2_REPLAY_MODE:-${UNIVLR_A2_REPLAY_MODE:-suffix}}"
A2_REPLAY_PROB="${A2_REPLAY_PROB:-${UNIVLR_A2_REPLAY_PROB:-0.5}}"

UNIVLR_STAGE_NORMALIZED="${UNIVLR_STAGE,,}"
case "${UNIVLR_STAGE_NORMALIZED}" in
    a1|stage1|stage1_a1|stage1-a1)
        UNIVLR_STAGE="a1"
        UNIVLR_STAGE_RUN_TAG=""
        ;;
    a2|stage1_a2|stage1-a2)
        UNIVLR_STAGE="a2"
        UNIVLR_STAGE_RUN_TAG="_a2replay${A2_REPLAY_MODE}${A2_REPLAY_PROB}"
        ;;
    *)
        echo "Unsupported UNIVLR_STAGE: ${UNIVLR_STAGE}" >&2
        echo "Supported stages: a1, a2" >&2
        exit 1
        ;;
esac

# Model freezing config.
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-True}"
FREEZE_MERGER="${FREEZE_MERGER:-True}"
FREEZE_LLM="${FREEZE_LLM:-False}"

# Runtime config.
MAX_TOKEN="${MAX_TOKEN:-5120}"
MIN_TOKEN="${MIN_TOKEN:-128}"
MAX_STEPS="${MAX_STEPS:--1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
REPORT_TO="${REPORT_TO:-tensorboard}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/zero3.json}"
ONLINE_CHECKPOINT="${ONLINE_CHECKPOINT:-False}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"

ALIGN_LAYER_RUN_TAG=""
if [[ "${UNIVLR_ALIGN_LAYER}" != "-1" ]]; then
    ALIGN_LAYER_RUN_TAG="_alignL${UNIVLR_ALIGN_LAYER//-/m}"
fi
RUN_NAME="${RUN_NAME:-univlr_stage2_offline_${MIXED_DATA_MODE_TAG}_zebra_step1_vertical_ablation_mix${ZEBRA_RATIO_NUM}to${VISUAL_RATIO_NUM}${UNIVLR_STAGE_RUN_TAG}${ALIGN_LAYER_RUN_TAG}_${MODEL_SIZE}_lr${LR}_head${UNIVLR_HEAD}_${UNIVLR_HEAD_TYPE}_bsz${GLOBAL_BATCH_SIZE}_loss${LOSS_UNIVLR_FCT}_lambda${LAMBDA_UNIVLR}_imgLat${IMAGE_LATENT_TOKENS}_maxImgToken${MAX_TOKEN}}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/${RUN_NAME}}"

EXTRA_STAGE_ARGS=(
    --univlr_stage "$UNIVLR_STAGE"
    --univlr_align_layer "$UNIVLR_ALIGN_LAYER"
)
if [[ "${UNIVLR_STAGE}" == "a2" ]]; then
    EXTRA_STAGE_ARGS+=(
        --univlr_a2_replay_mode "$A2_REPLAY_MODE"
        --univlr_a2_replay_prob "$A2_REPLAY_PROB"
    )
fi

echo "Using DATA_PATH=${DATA_PATH}"
echo "Using ZEBRA_STEP1_PATH=${ZEBRA_STEP1_PATH}"
echo "Using VISUAL_COT_PATH=${VISUAL_COT_PATH}"
echo "Using USE_OFFLINE_TARGETS=${USE_OFFLINE_TARGETS}"
echo "Using UNIVLR_TARGET_RESAMPLE_MODE=${UNIVLR_TARGET_RESAMPLE_MODE}"
echo "Using OFFLINE_TARGET_DIRNAME=${OFFLINE_TARGET_DIRNAME}"
echo "Using OFFLINE_MANIFEST_NAME=${OFFLINE_MANIFEST_NAME}"
echo "Using A1_BASE_CHECKPOINT=${A1_BASE_CHECKPOINT}"
echo "Using NUM_DEVICES=${NUM_DEVICES}, GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}"
echo "Using LOSS_UNIVLR_FCT=${LOSS_UNIVLR_FCT}, LAMBDA_UNIVLR=${LAMBDA_UNIVLR}, MAX_GRAD_NORM=${MAX_GRAD_NORM}"
echo "Using UNIVLR_HEAD=${UNIVLR_HEAD}, UNIVLR_HEAD_TYPE=${UNIVLR_HEAD_TYPE}, UNIVLR_HEAD_LR=${UNIVLR_HEAD_LR}"
echo "Using UNIVLR_STAGE=${UNIVLR_STAGE}"
echo "Using UNIVLR_ALIGN_LAYER=${UNIVLR_ALIGN_LAYER}"
if [[ "${UNIVLR_STAGE}" == "a2" ]]; then
    echo "Using A2_REPLAY_MODE=${A2_REPLAY_MODE}, A2_REPLAY_PROB=${A2_REPLAY_PROB}"
fi
echo "Using MIX_RATIO=${ZEBRA_RATIO_NUM}:${VISUAL_RATIO_NUM}, RANDOM_SEED=${RANDOM_SEED}"
echo "Using MASK_FIRST_TOKEN_AFTER_LATENT=${MASK_FIRST_TOKEN_AFTER_LATENT}"

deepspeed src/train/train_univlr_stage1.py \
    --run_name "$RUN_NAME" \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --model_id "$MODEL_NAME" \
    --checkpoint_name "$A1_BASE_CHECKPOINT" \
    --data_path "$DATA_PATH" \
    --image_folder "$MONET_ROOT" \
    --remove_unused_columns False \
    --univlr_head "$UNIVLR_HEAD" \
    --univlr_head_type "$UNIVLR_HEAD_TYPE" \
    --freeze_vision_tower "$FREEZE_VISION_TOWER" \
    --freeze_merger "$FREEZE_MERGER" \
    --freeze_llm "$FREEZE_LLM" \
    --learning_rate "$LR" \
    "${EXTRA_STAGE_ARGS[@]}" \
    --univlr_head_lr "$UNIVLR_HEAD_LR" \
    --loss_univlr_fct "$UNIVLR_LOSS_FCT" \
    --loss_univlr_lambda "$LAMBDA_UNIVLR" \
    --loss_univlr_fct "$LOSS_UNIVLR_FCT" \
    --loss_text_lambda "$TEXT_LOSS_LAMBDA" \
    --loss_image_lambda "$IMAGE_LOSS_LAMBDA" \
    --text_latent_tokens "$TEXT_LATENT_TOKENS" \
    --image_latent_tokens "$IMAGE_LATENT_TOKENS" \
    --univlr_target_resample_mode "$UNIVLR_TARGET_RESAMPLE_MODE" \
    --mask_first_token_after_latent "$MASK_FIRST_TOKEN_AFTER_LATENT" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --mode_switch_loss False \
    --online_checkpoint "$ONLINE_CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps "$MAX_STEPS" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --per_device_train_batch_size "$BATCH_PER_DEVICE" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --image_min_pixels $((MIN_TOKEN * 28 * 28)) \
    --image_max_pixels $((MAX_TOKEN * 28 * 28)) \
    --weight_decay 0.1 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --gradient_checkpointing True \
    --report_to "$REPORT_TO" \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --enable_data_packing False \
    --random_seed "$RANDOM_SEED"
