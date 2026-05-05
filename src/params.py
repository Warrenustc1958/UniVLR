from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments as HFTrainingArguments


@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="Qwen/Qwen2-VL-7B-Instruct")
    # set continuous reasoning mode
    coconut: bool = field(default=True)
    univlr_head: bool = field(default=False)
    univlr_head_type: str = field(default="simple")
    latent_end_token: bool = field(default=False)
    max_univlr_tokens: int = field(default=None)


@dataclass
class TrainingArguments(HFTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    loss_univlr_fct: str = field(default="mse")
    loss_univlr_lambda: float = field(default=1e-1)
    loss_univlr_fct: str = field(default="mse")
    loss_vae_nll_weight: float = field(default=1.0)
    loss_vae_kl_beta: float = field(default=1e-3)
    loss_vae_det_weight: float = field(default=0.0)
    loss_vae_logvar_reg_weight: float = field(default=0.0)
    loss_text_lambda: float = field(default=1.0)
    loss_image_lambda: float = field(default=1.0)
    univlr_stage: str = field(default="a1")
    univlr_align_layer: int = field(
        default=-1,
        metadata={
            "help": (
                "Hidden-state layer used for UniVLR target alignment. "
                "-1 keeps the final last_hidden_state path; non-negative values "
                "index the Transformers hidden_states tuple, where 0 is embeddings "
                "and N is the output after decoder layer N."
            )
        },
    )
    univlr_a2_replay_prob: Optional[float] = field(default=None)
    univlr_a2_replay_mode: str = field(default="prefix")
    univlr_vae_logvar_min: float = field(default=-6.0)
    univlr_vae_logvar_max: float = field(default=2.0)
    univlr_vae_prior_type: str = field(default="learned")
    univlr_vae_target_logvar: float = field(default=0.0)


    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)

    max_seq_length: int = field(
        default=32768, # This is the default value of the qwen2-vl model
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    univlr_head_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    # use_liger: bool = True
    run_name: Optional[str] = field(default="vscode debugger", metadata={"help": "Name of the run for logging purposes."})
    # True if serving the checkpoints and data on oci
    online_checkpoint: Optional[bool] = False
    checkpoint_name:Optional[str] = None
    # data packing-related params
    enable_data_packing: bool = False
    max_packed_tokens:Optional[int] = None
    # long_seq_cut:Optional[int] = field(default=25600, metadata={"help": "Max Len of long single data instnace allowed"})
    long_seq_threshold:Optional[int] = field(default=4096, metadata={"help": "Threshold to be a long single instance"})
    max_instance_per_batch:Optional[int] = 4
    max_steps:Optional[int] = 2500

    mode_switch_loss: Optional[bool] = False
    loss_mode_switch_fct: Optional[str] = field(default="mse")
    loss_mode_switch_lambda:Optional[float] = field(default=1e-1)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False
    image_folder: Optional[str] = field(default=None)
    image_min_pixels: Optional[int] = field(default=3136)
    image_max_pixels: Optional[int] = field(default=12845056)
    video_min_pixels: Optional[int] = field(default=100352)
    video_max_pixels: Optional[int] = field(default=602112)
    image_resized_width: int = field(default=None)
    image_resized_height: int = field(default=None)
    video_resized_width: int = field(default=None)
    video_resized_height: int = field(default=None)
    fps: float = 1.0
    random_seed: Optional[int] = field(default=None)
    univlr_target_folder: Optional[str] = field(
        default=None,
        metadata={"help": "Optional root for offline UniVLR target_path files."},
    )
    text_latent_tokens: int = field(default=8)
    image_latent_tokens: int = field(default=24)
    univlr_target_resample_mode: str = field(
        default="pool_avg",
        metadata={
            "help": (
                "Shared UniVLR target resample mode for both online teacher extraction and "
                "offline target precompute/load. Examples: pool_avg, pool_mlerp, interp_bilinear."
            )
        },
    )
    mask_first_token_after_latent: bool = field(default=False)
    
