#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"

# Data config. In offline mode we consume precomputed manifests such as:
#   <subset>/qwen2_5_vl_latent_targets_24token_2dpool/train_offline_k24.json
# and each manifest resolves its own relative target_path entries.
MONET_ROOT="${MONET_ROOT:-data/Monet-SFT-125K}"
SUBSETS="${SUBSETS:-}"
DEFAULT_OFFLINE_SUBSETS="${DEFAULT_OFFLINE_SUBSETS:-Visual_CoT}"
DEFAULT_ONLINE_SUBSETS="${DEFAULT_ONLINE_SUBSETS:-Visual_CoT}"
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
        DEFAULT_TARGET_RUN_TAG="latend_2dpool"
        ;;
    mlerp|pool_mlerp)
        UNIVLR_TARGET_RESAMPLE_MODE="pool_mlerp"
        DEFAULT_OFFLINE_TARGET_DIRNAME="qwen2_5_vl_latent_targets_${IMAGE_LATENT_TOKENS}token_pool_mlerp"
        DEFAULT_OFFLINE_MANIFEST_NAME="train_offline_k${IMAGE_LATENT_TOKENS}_pool_mlerp.json"
        DEFAULT_TARGET_RUN_TAG="latend_pool_mlerp"
        ;;
    *)
        echo "Unsupported UNIVLR_TARGET_RESAMPLE_MODE: ${UNIVLR_TARGET_RESAMPLE_MODE_RAW}" >&2
        echo "Supported aliases: avgpool/pool_avg, mlerp/pool_mlerp" >&2
        exit 1
        ;;
esac

OFFLINE_TARGET_DIRNAME="${OFFLINE_TARGET_DIRNAME:-${DEFAULT_OFFLINE_TARGET_DIRNAME}}"
OFFLINE_MANIFEST_NAME="${OFFLINE_MANIFEST_NAME:-${DEFAULT_OFFLINE_MANIFEST_NAME}}"

if [[ -z "${DATA_PATH:-}" ]]; then
    if [[ ! -d "$MONET_ROOT" ]]; then
        echo "MONET_ROOT does not exist: $MONET_ROOT" >&2
        exit 1
    fi

    if [[ "$USE_OFFLINE_TARGETS_NORMALIZED" == "true" ]]; then
        DATA_REL_PATH="${OFFLINE_TARGET_DIRNAME}/${OFFLINE_MANIFEST_NAME}"
    else
        DATA_REL_PATH="train.json"
    fi

    if [[ -z "$SUBSETS" ]]; then
        if [[ "$USE_OFFLINE_TARGETS_NORMALIZED" == "true" ]]; then
            SUBSETS="$DEFAULT_OFFLINE_SUBSETS"
        else
            SUBSETS="$DEFAULT_ONLINE_SUBSETS"
        fi
    fi

    if [[ -n "$SUBSETS" ]]; then
        read -r -a SUBSET_ARRAY <<< "$SUBSETS"
    else
        mapfile -t SUBSET_ARRAY < <(
            find "$MONET_ROOT" -mindepth 1 -maxdepth 1 -type d | while read -r subset_dir; do
                if [[ -f "$subset_dir/${DATA_REL_PATH}" ]]; then
                    basename "$subset_dir"
                fi
            done | sort
        )
    fi

    if [[ ${#SUBSET_ARRAY[@]} -eq 0 ]]; then
        echo "No Monet subsets with ${DATA_REL_PATH} found under $MONET_ROOT" >&2
        exit 1
    fi

    DATA_FILES=()
    for subset in "${SUBSET_ARRAY[@]}"; do
        train_path="${MONET_ROOT}/${subset}/${DATA_REL_PATH}"
        if [[ ! -f "$train_path" ]]; then
            if [[ "$USE_OFFLINE_TARGETS_NORMALIZED" == "true" ]]; then
                echo "Missing offline manifest for subset ${subset}: ${train_path}" >&2
                echo "Please run scripts/precompute_target.sh for this subset or set USE_OFFLINE_TARGETS=False." >&2
            else
                echo "Missing train.json for subset ${subset}: ${train_path}" >&2
            fi
            exit 1
        fi
        DATA_FILES+=("$train_path")
    done
    DATA_PATH="$(IFS=,; echo "${DATA_FILES[*]}")"
fi

# Model config.
MODEL_SIZE="${MODEL_SIZE:-7B}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-${MODEL_SIZE}-Instruct}"
export WANDB_PROJECT="${WANDB_PROJECT:-UniVLR-Qwen25-VL-${MODEL_SIZE}-UniVLR-Stage1-Monet}"

RANDOM_SEED="${RANDOM_SEED:-42}"

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

# Optimization / loss config. Keep Stage1 close to UniVLR's stable warmup:
# raw MSE target alignment, small latent weight, clipped gradients.
LR="${LR:-1e-5}"
UNIVLR_LOSS_FCT="${UNIVLR_LOSS_FCT:-mse}"
LOSS_UNIVLR_FCT="${LOSS_UNIVLR_FCT:-${UNIVLR_LOSS_FCT:-ln_mse_cosine}}"
LAMBDA_UNIVLR="${LAMBDA_UNIVLR:-0.1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
TEXT_LOSS_LAMBDA="${TEXT_LOSS_LAMBDA:-1.0}"
IMAGE_LOSS_LAMBDA="${IMAGE_LOSS_LAMBDA:-1.0}"
TEXT_LATENT_TOKENS="${TEXT_LATENT_TOKENS:-8}"
MASK_FIRST_TOKEN_AFTER_LATENT="${MASK_FIRST_TOKEN_AFTER_LATENT:-False}"
UNIVLR_HEAD="${UNIVLR_HEAD:-True}"
UNIVLR_HEAD_TYPE="${UNIVLR_HEAD_TYPE:-simple}"
UNIVLR_HEAD_LR="${UNIVLR_HEAD_LR:-1e-4}"
LOSS_VAE_NLL_WEIGHT="${LOSS_VAE_NLL_WEIGHT:-0.25}"
LOSS_VAE_KL_BETA="${LOSS_VAE_KL_BETA:-0}"
LOSS_VAE_DET_WEIGHT="${LOSS_VAE_DET_WEIGHT:-1.0}"
LOSS_VAE_LOGVAR_REG_WEIGHT="${LOSS_VAE_LOGVAR_REG_WEIGHT:-0.0}"
UNIVLR_VAE_LOGVAR_MIN="${UNIVLR_VAE_LOGVAR_MIN:--4.5}"
UNIVLR_VAE_LOGVAR_MAX="${UNIVLR_VAE_LOGVAR_MAX:-0.5}"
UNIVLR_VAE_PRIOR_TYPE="${UNIVLR_VAE_PRIOR_TYPE:-target}"
UNIVLR_VAE_TARGET_LOGVAR="${UNIVLR_VAE_TARGET_LOGVAR:-0.0}"
UNIVLR_ALIGN_LAYER="${UNIVLR_ALIGN_LAYER:-16}"

# Model freezing config. Vision is the online target teacher in this stage.
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
SAVE_STEPS="${SAVE_STEPS:-2000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"

VAE_RUN_TAG=""
if [[ "${UNIVLR_HEAD_TYPE,,}" == "vae" ]]; then
    VAE_RUN_TAG="_det${LOSS_VAE_DET_WEIGHT}_prior${UNIVLR_VAE_PRIOR_TYPE}_plogvar${UNIVLR_VAE_TARGET_LOGVAR}_univlreg${LOSS_VAE_LOGVAR_REG_WEIGHT}"
fi
ALIGN_LAYER_RUN_TAG=""
if [[ "${UNIVLR_ALIGN_LAYER}" != "-1" ]]; then
    ALIGN_LAYER_RUN_TAG="_alignL${UNIVLR_ALIGN_LAYER//-/m}"
fi
RUN_NAME="${RUN_NAME:-univlr_offline_align_zebra_cot_viscot_stage1_a1_monet_${MODEL_SIZE}_lr${LR}_head${UNIVLR_HEAD}_${UNIVLR_HEAD_TYPE}${VAE_RUN_TAG}${ALIGN_LAYER_RUN_TAG}_bsz${GLOBAL_BATCH_SIZE}_loss${LOSS_UNIVLR_FCT}_lambda${LAMBDA_UNIVLR}_imgLat${IMAGE_LATENT_TOKENS}_${DEFAULT_TARGET_RUN_TAG}_maxImgToken${MAX_TOKEN}}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/${RUN_NAME}}"

echo "Using DATA_PATH=${DATA_PATH}"
echo "Using SUBSETS=${SUBSETS}"
echo "Using USE_OFFLINE_TARGETS=${USE_OFFLINE_TARGETS}"
echo "Using UNIVLR_TARGET_RESAMPLE_MODE=${UNIVLR_TARGET_RESAMPLE_MODE}"
echo "Using OFFLINE_TARGET_DIRNAME=${OFFLINE_TARGET_DIRNAME}"
echo "Using OFFLINE_MANIFEST_NAME=${OFFLINE_MANIFEST_NAME}"
echo "Using NUM_DEVICES=${NUM_DEVICES}, GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}"
echo "Using LOSS_UNIVLR_FCT=${LOSS_UNIVLR_FCT}, LAMBDA_UNIVLR=${LAMBDA_UNIVLR}, MAX_GRAD_NORM=${MAX_GRAD_NORM}"
echo "Using UNIVLR_HEAD=${UNIVLR_HEAD}, UNIVLR_HEAD_TYPE=${UNIVLR_HEAD_TYPE}, UNIVLR_HEAD_LR=${UNIVLR_HEAD_LR}"
echo "Using UNIVLR_ALIGN_LAYER=${UNIVLR_ALIGN_LAYER}"
echo "Using LOSS_VAE_NLL_WEIGHT=${LOSS_VAE_NLL_WEIGHT}, LOSS_VAE_KL_BETA=${LOSS_VAE_KL_BETA}, LOSS_VAE_DET_WEIGHT=${LOSS_VAE_DET_WEIGHT}, LOSS_VAE_LOGVAR_REG_WEIGHT=${LOSS_VAE_LOGVAR_REG_WEIGHT}, LOGVAR=[${UNIVLR_VAE_LOGVAR_MIN}, ${UNIVLR_VAE_LOGVAR_MAX}], PRIOR_TYPE=${UNIVLR_VAE_PRIOR_TYPE}, TARGET_LOGVAR=${UNIVLR_VAE_TARGET_LOGVAR}"
echo "Using MASK_FIRST_TOKEN_AFTER_LATENT=${MASK_FIRST_TOKEN_AFTER_LATENT}"

deepspeed src/train/train_univlr_stage1.py \
    --run_name "$RUN_NAME" \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --model_id "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --image_folder "$MONET_ROOT" \
    --remove_unused_columns False \
    --univlr_head "$UNIVLR_HEAD" \
    --univlr_head_type "$UNIVLR_HEAD_TYPE" \
    --freeze_vision_tower "$FREEZE_VISION_TOWER" \
    --freeze_merger "$FREEZE_MERGER" \
    --freeze_llm "$FREEZE_LLM" \
    --learning_rate "$LR" \
    --univlr_stage a1 \
    --univlr_align_layer "$UNIVLR_ALIGN_LAYER" \
    --univlr_head_lr "$UNIVLR_HEAD_LR" \
    --loss_univlr_fct "$UNIVLR_LOSS_FCT" \
    --loss_univlr_lambda "$LAMBDA_UNIVLR" \
    --loss_univlr_fct "$LOSS_UNIVLR_FCT" \
    --loss_vae_nll_weight "$LOSS_VAE_NLL_WEIGHT" \
    --loss_vae_kl_beta "$LOSS_VAE_KL_BETA" \
    --loss_vae_det_weight "$LOSS_VAE_DET_WEIGHT" \
    --loss_vae_logvar_reg_weight "$LOSS_VAE_LOGVAR_REG_WEIGHT" \
    --loss_text_lambda "$TEXT_LOSS_LAMBDA" \
    --loss_image_lambda "$IMAGE_LOSS_LAMBDA" \
    --univlr_vae_logvar_min "$UNIVLR_VAE_LOGVAR_MIN" \
    --univlr_vae_logvar_max "$UNIVLR_VAE_LOGVAR_MAX" \
    --univlr_vae_prior_type "$UNIVLR_VAE_PRIOR_TYPE" \
    --univlr_vae_target_logvar "$UNIVLR_VAE_TARGET_LOGVAR" \
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
