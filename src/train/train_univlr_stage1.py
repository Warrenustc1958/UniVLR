import os

import torch
from transformers import AutoConfig, AutoProcessor, HfArgumentParser
from transformers.trainer_utils import SaveStrategy

from src.dataset import make_supervised_data_module_univlr_stage1
from src.model.qwen_univlr_model import QwenWithUniVLR
from src.params import DataArguments, ModelArguments, TrainingArguments
from src.s3_checkpoints_univlr import OCIFolderCheckpointHandler, create_temp_dir
from src.trainer import QwenUniVLRSFTTrainer
from src.train.monkey_patch_patch_emb import replace_qwen_2_5_vl_patch_emb
from train.train_utils import safe_save_model_for_hf_trainer, resolve_model_path_and_resume_checkpoint
from monkey_patch_forward_univlr import (
    replace_qwen2_5_with_mixed_modality_forward_univlr_stage1,
    replace_qwen2_5_with_mixed_modality_forward_univlr_stage1_a2,
    replace_qwen2_5_with_mixed_modality_forward_univlr_stage2,
)


local_rank = None


def rank0_print(*args):
    if local_rank == 0 or local_rank == "0" or local_rank is None:
        print(*args)


def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)

    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, not training_args.freeze_merger)


def configure_llm(model, training_args):
    set_requires_grad(model.lm_head.parameters(), not training_args.freeze_llm)
    set_requires_grad(model.model.parameters(), not training_args.freeze_llm)


def initialize_univlr_special_tokens(model, processor):
    """Start new latent tokens from stable existing embeddings instead of random rows."""
    tokenizer = processor.tokenizer
    token_to_source = {
        "<|univlr|>": getattr(model.config, "image_token_id", None),
        "<|univlr_start|>": tokenizer.convert_tokens_to_ids("<|im_start|>"),
        "<|univlr_end|>": tokenizer.convert_tokens_to_ids("<|im_end|>"),
        "<|univlr_latent_end|>": tokenizer.convert_tokens_to_ids("<|im_end|>"),
    }

    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data if model.get_output_embeddings() is not None else None

    for token, source_id in token_to_source.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id < 0 or source_id is None or source_id < 0:
            continue
        input_embeddings[token_id].copy_(input_embeddings[source_id])
        if output_embeddings is not None and token_id < output_embeddings.size(0) and source_id < output_embeddings.size(0):
            output_embeddings[token_id].copy_(output_embeddings[source_id])


def _configure_online_checkpointing(model_args, training_args):
    if not training_args.online_checkpoint:
        return None, None, None

    access_key_id = os.environ.get("ACCESS_KEY_ID")
    secret_access_key = os.environ.get("SECRET_ACCESS_KEY")
    endpoint_url = os.environ.get("ENDPOINT_URL")
    bucket_name = os.environ.get("BUCKET_NAME")
    region_name = os.environ.get("REGION_NAME")

    model_name = model_args.model_id.split("/")[-1]
    cache_dir = os.getenv("CACHE_DIR")
    local_model_name_or_path = create_temp_dir(
        base_path=os.path.join(cache_dir, model_name),
        prefix=training_args.run_name + "-",
    )

    remote_dir = os.path.join(training_args.output_dir, model_name, training_args.run_name)
    training_args.remote_output_dir = remote_dir
    training_args.output_dir = local_model_name_or_path.name

    oci_handler = OCIFolderCheckpointHandler(
        access_key_id,
        secret_access_key,
        endpoint_url,
        bucket_name,
        region_name,
    )
    return oci_handler, local_model_name_or_path, model_name


def train():
    global local_rank

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    debug_skip_save = os.environ.get("UNIVLR_DEBUG_SKIP_SAVE", "0") == "1"
    if debug_skip_save:
        training_args.save_strategy = SaveStrategy.NO

    if training_args.enable_data_packing:
        raise ValueError("UniVLR Stage1 does not support data packing yet. Set --enable_data_packing false.")
    if training_args.mode_switch_loss:
        raise ValueError("UniVLR Stage1 does not use mode-switch loss. Set --mode_switch_loss false.")
    if not training_args.freeze_vision_tower:
        rank0_print("UniVLR Stage1 freezes vision_tower by default because online visual targets use it as teacher.")
        training_args.freeze_vision_tower = True
    if not training_args.freeze_merger:
        rank0_print("UniVLR Stage1 freezes merger by default because online visual targets use it as teacher.")
        training_args.freeze_merger = True
    if training_args.max_grad_norm is None or training_args.max_grad_norm <= 0:
        rank0_print("UniVLR Stage1 sets max_grad_norm=1.0 for the atomic warmup.")
        training_args.max_grad_norm = 1.0
    if model_args.univlr_head:
        rank0_print(
            "UniVLR Stage1 uses univlr_head as a projection head for latent alignment loss."
        )
        if str(model_args.univlr_head_type).lower() == "vae":
            rank0_print(
                "UniVLR Stage1 VAE mode uses posterior NLL + KL(q||p) with a tiny teacher-side prior head."
            )
            rank0_print(
                "VAE settings: "
                f"prior_type={training_args.univlr_vae_prior_type}, "
                f"det_weight={training_args.loss_vae_det_weight}, "
                f"logvar_reg_weight={training_args.loss_vae_logvar_reg_weight}, "
                f"target_logvar={training_args.univlr_vae_target_logvar}."
            )
    rank0_print(f"UniVLR target resample mode: {data_args.univlr_target_resample_mode}")
    rank0_print(f"UniVLR alignment layer: {training_args.univlr_align_layer}")
    oci_handler, temp_folder, model_name = _configure_online_checkpointing(model_args, training_args)

    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    model_pth, resume_from_checkpoint = resolve_model_path_and_resume_checkpoint(
        model_id=model_args.model_id,
        checkpoint_name=training_args.checkpoint_name,
        resume_from_checkpoint=training_args.resume_from_checkpoint,
        online_checkpoint=training_args.online_checkpoint,
        model_name=model_name if training_args.online_checkpoint else None,
        oci_handler=oci_handler,
        download_prefix="univlr-stage1",
    )
    training_args.resume_from_checkpoint = resume_from_checkpoint

    config = AutoConfig.from_pretrained(model_pth, trust_remote_code=True)
    config.latent_end_token = False
    config.univlr_head = model_args.univlr_head
    config.univlr_head_type = model_args.univlr_head_type
    config.loss_univlr_fct = training_args.loss_univlr_fct
    config.loss_univlr_lambda = training_args.loss_univlr_lambda
    config.loss_univlr_fct = training_args.loss_univlr_fct
    config.loss_vae_nll_weight = training_args.loss_vae_nll_weight
    config.loss_vae_kl_beta = training_args.loss_vae_kl_beta
    config.loss_vae_det_weight = training_args.loss_vae_det_weight
    config.loss_vae_logvar_reg_weight = training_args.loss_vae_logvar_reg_weight
    config.loss_text_lambda = training_args.loss_text_lambda
    config.loss_image_lambda = training_args.loss_image_lambda
    config.univlr_stage = training_args.univlr_stage
    config.univlr_align_layer = training_args.univlr_align_layer
    config.image_latent_tokens = data_args.image_latent_tokens
    config.univlr_slots_per_block = data_args.image_latent_tokens
    config.univlr_target_resample_mode = data_args.univlr_target_resample_mode
    config.univlr_a2_replay_mode = training_args.univlr_a2_replay_mode
    config.univlr_vae_logvar_min = training_args.univlr_vae_logvar_min
    config.univlr_vae_logvar_max = training_args.univlr_vae_logvar_max
    config.univlr_vae_prior_type = training_args.univlr_vae_prior_type
    config.univlr_vae_target_logvar = training_args.univlr_vae_target_logvar
    if training_args.univlr_a2_replay_prob is not None:
        config.univlr_a2_replay_prob = training_args.univlr_a2_replay_prob

    if "Qwen2.5" not in model_args.model_id:
        raise ValueError("UniVLR Stage1 currently follows the UniVLR Qwen2.5-VL monkey patch path.")

    univlr_stage = str(training_args.univlr_stage).lower()
    if univlr_stage in ("a2", "stage1_a2", "stage1-a2"):
        rank0_print(
            "UniVLR Stage1-A2 uses detached self-replay scheduled sampling "
            f"with mode={training_args.univlr_a2_replay_mode}, "
            f"prob={training_args.univlr_a2_replay_prob}."
        )
        replace_qwen2_5_with_mixed_modality_forward_univlr_stage1_a2()
    elif univlr_stage in ("a1", "stage1", "stage1_a1", "stage1-a1"):
        replace_qwen2_5_with_mixed_modality_forward_univlr_stage1()
    elif univlr_stage in ("stage2", "s2", "stage2_sft", "stage2-sft"):
        rank0_print(
            "UniVLR Stage2 uses multi-block chain-latent teacher forcing. "
            "<|univlr_latent_end|> is reserved for decoding-time control and is masked from CE."
        )
        replace_qwen2_5_with_mixed_modality_forward_univlr_stage2()
    else:
        raise ValueError(f"Unsupported univlr_stage: {training_args.univlr_stage}")

    model = QwenWithUniVLR.from_pretrained(
        model_pth,
        config=config,
        torch_dtype=compute_dtype,
        attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
    )

    replace_qwen_2_5_vl_patch_emb()

    model.config.use_cache = False
    if model_args.univlr_head and hasattr(model, "univlr_head"):
        model.univlr_head.to(dtype=compute_dtype)
    if model_args.univlr_head and hasattr(model, "univlr_head_prior"):
        model.univlr_head_prior.to(dtype=compute_dtype)
    configure_llm(model, training_args)
    configure_vision_tower(model, training_args, compute_dtype, training_args.device)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    processor = AutoProcessor.from_pretrained(
        model_args.model_id,
        min_pixels=data_args.image_min_pixels,
        max_pixels=data_args.image_max_pixels,
    )

    processor.tokenizer.add_tokens("<|univlr_start|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|univlr|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|univlr_latent_end|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|univlr_end|>", special_tokens=True)

    model.config.univlr_id = processor.tokenizer.convert_tokens_to_ids("<|univlr|>")
    model.config.univlr_latent_end_id = processor.tokenizer.convert_tokens_to_ids("<|univlr_latent_end|>")
    model.config.univlr_start_id = processor.tokenizer.convert_tokens_to_ids("<|univlr_start|>")
    model.config.univlr_end_id = processor.tokenizer.convert_tokens_to_ids("<|univlr_end|>")

    resized_token_embeddings = False
    if model.config.vocab_size < len(processor.tokenizer):
        model.resize_token_embeddings(len(processor.tokenizer))
        resized_token_embeddings = True
    if resized_token_embeddings and not training_args.checkpoint_name:
        initialize_univlr_special_tokens(model, processor)

    data_module = make_supervised_data_module_univlr_stage1(
        model_id=model_args.model_id,
        processor=processor,
        data_args=data_args,
    )

    trainer = QwenUniVLRSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        temp_folder=temp_folder,
        oci_handler=oci_handler,
        **data_module,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    if debug_skip_save:
        rank0_print("UNIVLR_DEBUG_SKIP_SAVE=1: skipping final trainer/model save.")
        return
    trainer.save_state()
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
