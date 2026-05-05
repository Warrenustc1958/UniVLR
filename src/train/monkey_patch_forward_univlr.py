import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import ModelOutput
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, is_torchdynamo_compiling

from src.constants import IGNORE_INDEX
from src.dataset.univlr_stage1_dataset import SLOT_TYPE_IMAGE, SLOT_TYPE_TEXT


def replace_qwen2_5_with_mixed_modality_forward_univlr_stage1():
    print("#" * 42)
    print("Activated UniVLR Stage1 atomic-operation forward.")
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = (
        qwen2_5_mixed_modality_forward_univlr_stage1
    )
    print("#" * 42)


def replace_qwen2_5_with_mixed_modality_forward_univlr_stage1_a2():
    print("#" * 42)
    print("Activated UniVLR Stage1-A2 self-replay forward.")
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = (
        qwen2_5_mixed_modality_forward_univlr_stage1_a2
    )
    print("#" * 42)


def replace_qwen2_5_with_mixed_modality_forward_univlr_stage2():
    print("#" * 42)
    print("Activated UniVLR Stage2 chain-latent forward.")
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = (
        qwen2_5_mixed_modality_forward_univlr_stage2
    )
    print("#" * 42)


@dataclass
class Qwen2_5_VLUniVLROutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_univlr: Optional[torch.FloatTensor] = None
    loss_ce: Optional[torch.FloatTensor] = None
    loss_mode_switch: Optional[torch.FloatTensor] = None
    loss_text_latent: Optional[torch.FloatTensor] = None
    loss_image_latent: Optional[torch.FloatTensor] = None
    loss_latent_nll: Optional[torch.FloatTensor] = None
    loss_latent_kl: Optional[torch.FloatTensor] = None
    loss_latent_det: Optional[torch.FloatTensor] = None
    loss_latent_logvar_reg: Optional[torch.FloatTensor] = None
    latent_logvar_q_mean: Optional[torch.FloatTensor] = None
    latent_logvar_p_mean: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    last_position_hidden_state: Optional[torch.FloatTensor] = None
    univlr_replay_ratio: Optional[torch.FloatTensor] = None


_UNIVLR_TARGET_RESAMPLE_ALIASES = {
    "adaptive_pool_image_grid": "pool_avg",
    "avg": "pool_avg",
    "max": "pool_max",
    "mlerp": "pool_mlerp",
    "nearest": "interp_nearest",
    "bilinear": "interp_bilinear",
    "bicubic": "interp_bicubic",
    "area": "interp_area",
}
_UNIVLR_TARGET_RESAMPLE_MODES = {
    "pool_avg",
    "pool_max",
    "pool_mlerp",
    "interp_nearest",
    "interp_bilinear",
    "interp_bicubic",
    "interp_area",
}


def _normalize_univlr_target_resample_mode(mode: Optional[str]) -> str:
    normalized = str(mode or "pool_avg").strip().lower().replace("-", "_")
    normalized = _UNIVLR_TARGET_RESAMPLE_ALIASES.get(normalized, normalized)
    if normalized not in _UNIVLR_TARGET_RESAMPLE_MODES:
        raise ValueError(
            "Unsupported UniVLR target resample mode: "
            f"{mode}. Supported modes: {sorted(_UNIVLR_TARGET_RESAMPLE_MODES)}"
        )
    return normalized


def build_univlr_target_resample_spec(
    mode: Optional[str] = None,
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    raw_spec = None
    if config is not None:
        raw_spec = getattr(config, "univlr_target_resample_spec", None)
        if raw_spec is None:
            raw_spec = getattr(config, "univlr_target_resample_mode", None)

    if isinstance(raw_spec, dict):
        mode = raw_spec.get("mode", raw_spec.get("pooling", mode))
    elif raw_spec is not None and mode is None:
        mode = raw_spec

    normalized_mode = _normalize_univlr_target_resample_mode(mode)
    return {
        "version": 1,
        "mode": normalized_mode,
    }


def _apply_mlerp_pool1d(features_1d: torch.Tensor, target_len: int) -> torch.Tensor:
    avg_pooled = F.adaptive_avg_pool1d(features_1d, int(target_len))
    squared_norms = (features_1d ** 2).sum(dim=1, keepdim=True)
    max_squared_norms = F.adaptive_max_pool1d(squared_norms, int(target_len))
    max_norms = torch.sqrt(max_squared_norms)
    avg_norms = torch.norm(avg_pooled, dim=1, keepdim=True)
    scale = max_norms / (avg_norms + 1e-8)
    return avg_pooled * scale


def _apply_mlerp_pool2d(grid: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
    avg_pooled = F.adaptive_avg_pool2d(grid, output_size)
    squared_norms = (grid ** 2).sum(dim=1, keepdim=True)
    max_squared_norms = F.adaptive_max_pool2d(squared_norms, output_size)
    max_norms = torch.sqrt(max_squared_norms)
    avg_norms = torch.norm(avg_pooled, dim=1, keepdim=True)
    scale = max_norms / (avg_norms + 1e-8)
    return avg_pooled * scale


def _resample_sequence_features(
    features: torch.Tensor,
    target_len: int,
    target_resample_spec: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    target_resample_spec = target_resample_spec or build_univlr_target_resample_spec()
    if features.ndim != 2:
        raise ValueError(f"Expected visual features [num_tokens, hidden], got {tuple(features.shape)}")
    if features.size(0) == target_len:
        return features

    mode = target_resample_spec["mode"]
    features_1d = features.transpose(0, 1).unsqueeze(0).float()
    if mode == "pool_avg":
        pooled = F.adaptive_avg_pool1d(features_1d, int(target_len))
    elif mode == "pool_max":
        pooled = F.adaptive_max_pool1d(features_1d, int(target_len))
    elif mode == "pool_mlerp":
        pooled = _apply_mlerp_pool1d(features_1d, int(target_len))
    elif mode.startswith("interp_"):
        interp_mode = mode[len("interp_") :]
        interp_mode_1d = "linear" if interp_mode in {"bilinear", "bicubic"} else interp_mode
        interpolate_kwargs = {}
        if interp_mode_1d == "linear":
            interpolate_kwargs["align_corners"] = False
        pooled = F.interpolate(features_1d, size=int(target_len), mode=interp_mode_1d, **interpolate_kwargs)
    else:
        raise ValueError(f"Unexpected UniVLR target resample mode: {mode}")

    return pooled.squeeze(0).transpose(0, 1).to(features.dtype)


def _adaptive_pool_sequence(
    features: torch.Tensor,
    target_len: int,
    target_resample_spec: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    return _resample_sequence_features(features, target_len, target_resample_spec=target_resample_spec)


def _choose_pool_hw(source_h: int, source_w: int, target_len: int) -> Tuple[int, int]:
    source_h = max(int(source_h), 1)
    source_w = max(int(source_w), 1)
    target_len = int(target_len)
    target_aspect = float(source_w) / float(source_h)
    best_hw = (1, target_len)
    best_score = None
    for h in range(1, int(math.sqrt(target_len)) + 1):
        if target_len % h != 0:
            continue
        w = target_len // h
        for pool_h, pool_w in ((h, w), (w, h)):
            pooled_aspect = float(pool_w) / float(max(pool_h, 1))
            score = abs(math.log((pooled_aspect + 1e-6) / (target_aspect + 1e-6)))
            if best_score is None or score < best_score:
                best_score = score
                best_hw = (pool_h, pool_w)
    return best_hw


def _adaptive_pool_image_grid(
    features: torch.Tensor,
    grid_thw: torch.Tensor,
    target_len: int,
    spatial_merge_size: int,
    target_resample_spec: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    target_resample_spec = target_resample_spec or build_univlr_target_resample_spec()
    if features.ndim != 2:
        raise ValueError(f"Expected visual features [num_tokens, hidden], got {tuple(features.shape)}")

    t, h, w = [int(x) for x in grid_thw.tolist()]
    merge = max(int(spatial_merge_size), 1)
    h = max(h // merge, 1)
    w = max(w // merge, 1)

    if t * h * w != features.size(0):
        return _adaptive_pool_sequence(features, target_len, target_resample_spec=target_resample_spec)
    if t == 1 and features.size(0) == target_len:
        return features

    hidden = features.size(-1)
    grid = features.reshape(t, h, w, hidden).mean(dim=0).permute(2, 0, 1).unsqueeze(0).float()
    pool_h, pool_w = _choose_pool_hw(h, w, target_len)
    mode = target_resample_spec["mode"]
    if mode == "pool_avg":
        pooled = F.adaptive_avg_pool2d(grid, (pool_h, pool_w))
    elif mode == "pool_max":
        pooled = F.adaptive_max_pool2d(grid, (pool_h, pool_w))
    elif mode == "pool_mlerp":
        pooled = _apply_mlerp_pool2d(grid, (pool_h, pool_w))
    elif mode.startswith("interp_"):
        interp_mode = mode[len("interp_") :]
        interpolate_kwargs = {}
        if interp_mode in {"bilinear", "bicubic"}:
            interpolate_kwargs["align_corners"] = False
        pooled = F.interpolate(grid, size=(pool_h, pool_w), mode=interp_mode, **interpolate_kwargs)
    else:
        raise ValueError(f"Unexpected UniVLR target resample mode: {mode}")
    pooled = pooled.squeeze(0).permute(1, 2, 0).reshape(pool_h * pool_w, hidden)
    return pooled.to(features.dtype)


def _make_latent_targets_from_images(
    self,
    univlr_pixel_values: torch.Tensor,
    univlr_image_grid_thw: torch.Tensor,
    univlr_token_counts: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    target_resample_spec = build_univlr_target_resample_spec(config=self.config)
    univlr_pixel_values = univlr_pixel_values.to(self.model.visual.device, dtype=self.model.visual.dtype)
    univlr_image_grid_thw = univlr_image_grid_thw.to(self.model.visual.device)
    token_counts = univlr_token_counts.detach().cpu().tolist()

    with torch.no_grad():
        feature_chunks = self.model.get_image_features(univlr_pixel_values, univlr_image_grid_thw)

    if len(feature_chunks) != len(token_counts):
        raise ValueError(
            f"Target image feature chunks ({len(feature_chunks)}) do not match "
            f"latent token counts ({len(token_counts)})."
        )

    if len(feature_chunks) != univlr_image_grid_thw.shape[0]:
        raise ValueError(
            f"Target image feature chunks ({len(feature_chunks)}) do not match "
            f"image_grid_thw rows ({univlr_image_grid_thw.shape[0]})."
        )

    spatial_merge_size = getattr(self.model.visual, "spatial_merge_size", 1)
    pooled_targets = []
    for features, grid_thw, num_tokens in zip(feature_chunks, univlr_image_grid_thw.detach().cpu(), token_counts):
        pooled_targets.append(
            _adaptive_pool_image_grid(
                features=features.detach(),
                grid_thw=grid_thw,
                target_len=int(num_tokens),
                spatial_merge_size=spatial_merge_size,
                target_resample_spec=target_resample_spec,
            )
        )
    return torch.cat(pooled_targets, dim=0).to(device=device, dtype=dtype)


def _tokenwise_latent_loss(pred: torch.Tensor, target: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "mse":
        return F.mse_loss(pred, target, reduction="none").mean(dim=-1)
    if loss_name == "mae":
        return F.l1_loss(pred, target, reduction="none").mean(dim=-1)
    if loss_name == "cosine":
        return 1 - F.cosine_similarity(pred, target, dim=-1)
    if loss_name == "smooth_l1_cosine":
        smooth_l1 = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)
        cosine = 1 - F.cosine_similarity(pred, target, dim=-1)
        return smooth_l1 + cosine
    if loss_name == "ln_mse_cosine":
        pred_ln = F.layer_norm(pred, pred.shape[-1:])
        target_ln = F.layer_norm(target, target.shape[-1:])
        mse = F.mse_loss(pred_ln, target_ln, reduction="none").mean(dim=-1)
        cosine = 1 - F.cosine_similarity(pred_ln, target_ln, dim=-1)
        return mse + cosine
    raise ValueError(f"Unsupported UniVLR latent loss: {loss_name}")


def _mean_by_sample(token_losses: torch.Tensor, sample_ids: Optional[torch.Tensor]) -> torch.Tensor:
    if token_losses.numel() == 0:
        return token_losses.new_zeros(())
    if sample_ids is None:
        return token_losses.mean()
    sample_ids = sample_ids.to(token_losses.device)
    per_sample = []
    for sample_id in torch.unique(sample_ids, sorted=True):
        mask = sample_ids == sample_id
        if mask.any():
            per_sample.append(token_losses[mask].mean())
    if not per_sample:
        return token_losses.new_zeros(())
    return torch.stack(per_sample).mean()


def _split_typed_token_losses(
    token_losses: torch.Tensor,
    slot_types: Optional[torch.Tensor],
    sample_ids: Optional[torch.Tensor],
    text_weight: float = 1.0,
    image_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if slot_types is None:
        image_loss = _mean_by_sample(token_losses, sample_ids)
        return image_loss, token_losses.new_zeros(()), image_loss

    slot_types = slot_types.to(token_losses.device)
    sample_ids = sample_ids.to(token_losses.device) if sample_ids is not None else None

    text_mask = slot_types == SLOT_TYPE_TEXT
    image_mask = slot_types == SLOT_TYPE_IMAGE
    text_loss = _mean_by_sample(
        token_losses[text_mask],
        sample_ids[text_mask] if sample_ids is not None else None,
    )
    image_loss = _mean_by_sample(
        token_losses[image_mask],
        sample_ids[image_mask] if sample_ids is not None else None,
    )

    total_loss = text_weight * text_loss + image_weight * image_loss
    return total_loss, text_loss, image_loss


def _split_typed_latent_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    slot_types: Optional[torch.Tensor],
    sample_ids: Optional[torch.Tensor],
    loss_name: str,
    text_weight: float = 1.0,
    image_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token_losses = _tokenwise_latent_loss(pred, target, loss_name)
    return _split_typed_token_losses(
        token_losses,
        slot_types,
        sample_ids,
        text_weight=text_weight,
        image_weight=image_weight,
    )


def _project_to_univlr_space(self, hidden_states: torch.Tensor) -> torch.Tensor:
    if getattr(self.config, "univlr_head", False):
        if not hasattr(self, "univlr_head"):
            raise ValueError("config.univlr_head=True, but model has no univlr_head module.")
        return self.univlr_head(hidden_states)
    return hidden_states


def _get_univlr_align_layer(self) -> int:
    return int(getattr(self.config, "univlr_align_layer", -1))


def _univlr_alignment_requires_hidden_states(self) -> bool:
    return _get_univlr_align_layer(self) != -1


def _select_univlr_alignment_hidden(
    self,
    final_hidden_states: torch.Tensor,
    all_hidden_states: Optional[Tuple[torch.Tensor, ...]],
) -> torch.Tensor:
    align_layer = _get_univlr_align_layer(self)
    if align_layer == -1:
        return final_hidden_states
    if all_hidden_states is None:
        raise ValueError(
            "univlr_align_layer requires output_hidden_states=True, "
            f"but hidden_states were not returned. align_layer={align_layer}"
        )

    num_states = len(all_hidden_states)
    layer_idx = align_layer if align_layer >= 0 else num_states + align_layer
    if layer_idx < 0 or layer_idx >= num_states:
        raise ValueError(
            f"Invalid univlr_align_layer={align_layer}; "
            f"available hidden_states tuple has {num_states} entries "
            "(0 is embeddings, N is decoder layer N output)."
        )
    return all_hidden_states[layer_idx]


def _is_univlr_vae(self) -> bool:
    return bool(getattr(self.config, "univlr_head", False)) and str(
        getattr(self.config, "univlr_head_type", "")
    ).lower() == "vae"


def _predict_univlr_distribution(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if not getattr(self.config, "univlr_head", False):
        return hidden_states, torch.zeros_like(hidden_states)
    if not hasattr(self, "predict_univlr_distribution"):
        raise ValueError("UniVLR VAE requires model.predict_univlr_distribution().")
    return self.predict_univlr_distribution(hidden_states)


def _predict_univlr_prior_distribution(
    self, hidden_states: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(self, "predict_univlr_prior_distribution"):
        raise ValueError("UniVLR VAE requires model.predict_univlr_prior_distribution().")
    return self.predict_univlr_prior_distribution(hidden_states)


def _resolve_univlr_prior_distribution(
    self, target_hidden: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    prior_type = str(getattr(self.config, "univlr_vae_prior_type", "learned")).lower()
    if prior_type == "learned":
        return _predict_univlr_prior_distribution(self, target_hidden)
    target_logvar = float(getattr(self.config, "univlr_vae_target_logvar", 0.0))
    if prior_type in {"target", "target_detached", "identity", "fixed"}:
        return target_hidden.detach(), torch.full_like(target_hidden, target_logvar)
    if prior_type in {"standard", "standard_normal", "gaussian"}:
        return torch.zeros_like(target_hidden), torch.full_like(target_hidden, target_logvar)
    raise ValueError(
        f"Unsupported UniVLR VAE prior_type='{prior_type}'. "
        "Use 'learned', 'target', or 'standard'."
    )


def _tokenwise_gaussian_nll(
    target: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    log_two_pi = math.log(2.0 * math.pi)
    inv_var = torch.exp(-logvar)
    nll = 0.5 * (((target - mean) ** 2) * inv_var + logvar + log_two_pi)
    return nll.mean(dim=-1)


def _tokenwise_gaussian_kl(
    mean_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mean_p: torch.Tensor,
    logvar_p: torch.Tensor,
) -> torch.Tensor:
    var_ratio = torch.exp(logvar_q - logvar_p)
    mean_diff = (mean_p - mean_q) ** 2 * torch.exp(-logvar_p)
    kl = 0.5 * (logvar_p - logvar_q + var_ratio + mean_diff - 1.0)
    return kl.mean(dim=-1)


def _tokenwise_logvar_reg(logvar_q: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
    return (logvar_q - logvar_p).pow(2).mean(dim=-1)


def _compute_univlr_alignment_loss(
    self,
    pred_hidden: torch.Tensor,
    target_hidden: torch.Tensor,
    latent_slot_types: Optional[torch.Tensor],
    latent_sample_ids: Optional[torch.Tensor],
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    pred_hidden = pred_hidden.to(torch.float32)
    target_hidden = target_hidden.to(torch.float32)
    text_weight = float(getattr(self.config, "loss_text_lambda", 1.0))
    image_weight = float(getattr(self.config, "loss_image_lambda", 1.0))
    loss_name = getattr(self.config, "loss_univlr_fct", getattr(self.config, "loss_univlr_fct", "ln_mse_cosine"))

    if _is_univlr_vae(self):
        mu_q, logvar_q = _predict_univlr_distribution(self, pred_hidden)
        mu_p, logvar_p = _resolve_univlr_prior_distribution(self, target_hidden)

        mu_q = mu_q.to(torch.float32)
        logvar_q = logvar_q.to(torch.float32)
        mu_p = mu_p.to(torch.float32)
        logvar_p = logvar_p.to(torch.float32)

        token_nll = _tokenwise_gaussian_nll(target_hidden, mu_q, logvar_q)
        token_kl = _tokenwise_gaussian_kl(mu_q, logvar_q, mu_p, logvar_p)
        token_det = _tokenwise_latent_loss(mu_q, target_hidden, loss_name)
        token_logvar_reg = _tokenwise_logvar_reg(logvar_q, logvar_p)
        token_losses = (
            float(getattr(self.config, "loss_vae_nll_weight", 1.0)) * token_nll
            + float(getattr(self.config, "loss_vae_kl_beta", 1e-3)) * token_kl
            + float(getattr(self.config, "loss_vae_det_weight", 0.0)) * token_det
            + float(getattr(self.config, "loss_vae_logvar_reg_weight", 0.0)) * token_logvar_reg
        )
        loss_univlr, loss_text_latent, loss_image_latent = _split_typed_token_losses(
            token_losses,
            latent_slot_types,
            latent_sample_ids,
            text_weight=text_weight,
            image_weight=image_weight,
        )
        loss_latent_nll = _mean_by_sample(token_nll, latent_sample_ids)
        loss_latent_kl = _mean_by_sample(token_kl, latent_sample_ids)
        loss_latent_det = _mean_by_sample(token_det, latent_sample_ids)
        loss_latent_logvar_reg = _mean_by_sample(token_logvar_reg, latent_sample_ids)
        latent_logvar_q_mean = logvar_q.mean()
        latent_logvar_p_mean = logvar_p.mean()
        return (
            loss_univlr,
            loss_text_latent,
            loss_image_latent,
            loss_latent_nll,
            loss_latent_kl,
            loss_latent_det,
            loss_latent_logvar_reg,
            latent_logvar_q_mean,
            latent_logvar_p_mean,
        )

    loss_univlr, loss_text_latent, loss_image_latent = _split_typed_latent_loss(
        pred_hidden,
        target_hidden,
        latent_slot_types,
        latent_sample_ids,
        loss_name,
        text_weight=text_weight,
        image_weight=image_weight,
    )
    return loss_univlr, loss_text_latent, loss_image_latent, None, None, None, None, None, None


def _get_univlr_a2_replay_prob(self) -> float:
    for attr_name in (
        "univlr_a2_replay_prob",
        "univlr_replay_prob",
        "scheduled_sampling_prob",
    ):
        value = getattr(self.config, attr_name, None)
        if value is not None:
            return max(0.0, min(1.0, float(value)))

    start = float(getattr(self.config, "univlr_replay_prob_start", 0.0))
    end = float(getattr(self.config, "univlr_replay_prob_end", 0.5))
    current_step = getattr(self.config, "univlr_replay_current_step", None)
    total_steps = getattr(self.config, "univlr_replay_total_steps", None)
    warmup_ratio = float(getattr(self.config, "univlr_replay_warmup_ratio", 1.0))
    if current_step is None or total_steps is None or total_steps <= 0:
        return max(0.0, min(1.0, end))

    warmup_steps = max(1.0, float(total_steps) * warmup_ratio)
    progress = max(0.0, min(1.0, float(current_step) / warmup_steps))
    prob = start + (end - start) * progress
    return max(0.0, min(1.0, prob))


def _make_univlr_replay_mask(
    num_latents: int,
    latent_step_ids: Optional[torch.Tensor],
    latent_sample_ids: Optional[torch.Tensor],
    replay_prob: float,
    replay_mode: str,
    device: torch.device,
) -> torch.Tensor:
    replay_mask = torch.zeros(num_latents, dtype=torch.bool, device=device)
    replay_mode = replay_mode.lower()
    if num_latents == 0 or replay_prob <= 0.0 or replay_mode in ("none", "off", "false"):
        return replay_mask
    if replay_prob >= 1.0 and replay_mode in ("all", "prefix", "suffix", "tail", "postfix", "bernoulli", "token"):
        replay_mask[:] = True
        return replay_mask

    if replay_mode in ("bernoulli", "token", "tokenwise"):
        return torch.rand(num_latents, device=device) < replay_prob

    if replay_mode == "all":
        if bool((torch.rand((), device=device) < replay_prob).item()):
            replay_mask[:] = True
        return replay_mask

    suffix_modes = {"suffix", "tail", "postfix", "after_prefix", "second_half"}
    if replay_mode not in {"prefix", *suffix_modes}:
        raise ValueError(f"Unsupported UniVLR A2 replay mode: {replay_mode}")

    if latent_sample_ids is None:
        sample_ids = [0] * num_latents
    else:
        sample_ids = latent_sample_ids.detach().cpu().tolist()
    if latent_step_ids is None:
        step_ids = [0] * num_latents
    else:
        step_ids = latent_step_ids.detach().cpu().tolist()

    group_start = 0
    while group_start < num_latents:
        group_key = (sample_ids[group_start], step_ids[group_start])
        group_end = group_start + 1
        while group_end < num_latents and (sample_ids[group_end], step_ids[group_end]) == group_key:
            group_end += 1

        group_len = group_end - group_start
        if replay_mode in suffix_modes:
            replay_count = min(group_len, max(0, int(math.ceil(group_len * replay_prob))))
            if replay_count > 0:
                replay_mask[group_end - replay_count : group_end] = True
            group_start = group_end
            continue

        replay_count = int((torch.rand(group_len, device=device) < replay_prob).sum().item())
        if replay_count > 0:
            replay_mask[group_start : group_start + replay_count] = True
        group_start = group_end

    return replay_mask


def qwen2_5_mixed_modality_forward_univlr_stage1(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    latent_targets: Optional[torch.Tensor] = None,
    latent_slot_types: Optional[torch.Tensor] = None,
    latent_step_ids: Optional[torch.Tensor] = None,
    latent_sample_ids: Optional[torch.Tensor] = None,
    univlr_pixel_values: Optional[torch.Tensor] = None,
    univlr_image_grid_thw: Optional[torch.LongTensor] = None,
    univlr_token_counts: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[Tuple, Qwen2_5_VLUniVLROutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    if _univlr_alignment_requires_hidden_states(self):
        output_hidden_states = True
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    if pixel_values is None and pixel_values_videos is None:
        dummy_pixel = torch.zeros(784, 1176, device=self.model.visual.device, dtype=self.model.visual.dtype)
        dummy_grid = torch.tensor([[1, 28, 28]], device=self.model.visual.device)
        dummy_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
        inputs_embeds = inputs_embeds + dummy_embeds.mean().to(inputs_embeds.dtype) * 0

    if pixel_values is not None:
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0)
        image_mask = input_ids == self.config.image_token_id
        n_image_tokens = image_mask.sum()
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds = self.model.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0)
        video_mask = input_ids == self.config.video_token_id
        n_video_tokens = video_mask.sum()
        n_video_features = video_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )
        video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    univlr_mask = input_ids == self.config.univlr_id
    batch_indices, seq_positions = torch.nonzero(univlr_mask, as_tuple=True)
    num_univlr_tokens = int(univlr_mask.sum().item())

    if latent_targets is not None:
        selected_latent_targets = latent_targets.to(inputs_embeds.device, inputs_embeds.dtype)
    elif univlr_pixel_values is not None:
        selected_latent_targets = _make_latent_targets_from_images(
            self,
            univlr_pixel_values=univlr_pixel_values,
            univlr_image_grid_thw=univlr_image_grid_thw,
            univlr_token_counts=univlr_token_counts,
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
    else:
        selected_latent_targets = None

    if selected_latent_targets is None:
        raise ValueError("UniVLR Stage1 forward requires latent_targets or univlr target images.")
    if selected_latent_targets.shape[0] != num_univlr_tokens:
        raise ValueError(
            f"latent target count ({selected_latent_targets.shape[0]}) does not match "
            f"<|univlr|> token count ({num_univlr_tokens})."
        )

    inputs_embeds[batch_indices, seq_positions] = selected_latent_targets

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids += delta.to(position_ids.device)

    outputs = self.model.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:, -1, :]
    logits = self.lm_head(hidden_states)

    loss = None
    loss_ce = None
    loss_univlr = None
    loss_text_latent = None
    loss_image_latent = None
    loss_mode_switch = None
    loss_latent_nll = None
    loss_latent_kl = None
    loss_latent_det = None
    loss_latent_logvar_reg = None
    latent_logvar_q_mean = None
    latent_logvar_p_mean = None

    if labels is not None:
        logits = logits.float()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_labels = shift_labels.masked_fill(shift_labels == self.config.univlr_id, IGNORE_INDEX)
        latent_end_id = getattr(self.config, "univlr_latent_end_id", None)
        if latent_end_id is not None and int(latent_end_id) >= 0:
            shift_labels = shift_labels.masked_fill(shift_labels == int(latent_end_id), IGNORE_INDEX)

        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1).to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        alignment_hidden_states = _select_univlr_alignment_hidden(
            self,
            hidden_states,
            outputs.hidden_states,
        )
        pred_hidden = alignment_hidden_states[batch_indices, seq_positions - 1]
        (
            loss_univlr,
            loss_text_latent,
            loss_image_latent,
            loss_latent_nll,
            loss_latent_kl,
            loss_latent_det,
            loss_latent_logvar_reg,
            latent_logvar_q_mean,
            latent_logvar_p_mean,
        ) = (
            _compute_univlr_alignment_loss(
                self,
                pred_hidden,
                selected_latent_targets,
                latent_slot_types,
                latent_sample_ids,
            )
        )

        loss = loss_ce
        if loss_univlr is not None:
            loss = loss + float(getattr(self.config, "loss_univlr_lambda", 1.0)) * loss_univlr

    if not return_dict:
        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return Qwen2_5_VLUniVLROutputWithPast(
        loss=loss,
        loss_ce=loss_ce,
        loss_univlr=loss_univlr,
        loss_mode_switch=loss_mode_switch,
        loss_text_latent=loss_text_latent,
        loss_image_latent=loss_image_latent,
        loss_latent_nll=loss_latent_nll,
        loss_latent_kl=loss_latent_kl,
        loss_latent_det=loss_latent_det,
        loss_latent_logvar_reg=loss_latent_logvar_reg,
        latent_logvar_q_mean=latent_logvar_q_mean,
        latent_logvar_p_mean=latent_logvar_p_mean,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state=last_position_hidden_state,
    )


def qwen2_5_mixed_modality_forward_univlr_stage2(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    latent_targets: Optional[torch.Tensor] = None,
    latent_slot_types: Optional[torch.Tensor] = None,
    latent_step_ids: Optional[torch.Tensor] = None,
    latent_sample_ids: Optional[torch.Tensor] = None,
    univlr_pixel_values: Optional[torch.Tensor] = None,
    univlr_image_grid_thw: Optional[torch.LongTensor] = None,
    univlr_token_counts: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[Tuple, Qwen2_5_VLUniVLROutputWithPast]:
    """Stage2 chain-latent SFT forward.

    Stage2 keeps the same latent-alignment target as Stage1, but applies it
    across longer, multi-step latent chains. ``<|univlr_latent_end|>`` is treated
    as a decoding-time control token and is therefore masked from CE if it
    appears in the sequence.
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    if _univlr_alignment_requires_hidden_states(self):
        output_hidden_states = True
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    if pixel_values is None and pixel_values_videos is None:
        dummy_pixel = torch.zeros(784, 1176, device=self.model.visual.device, dtype=self.model.visual.dtype)
        dummy_grid = torch.tensor([[1, 28, 28]], device=self.model.visual.device)
        dummy_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
        inputs_embeds = inputs_embeds + dummy_embeds.mean().to(inputs_embeds.dtype) * 0

    if pixel_values is not None:
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0)
        image_mask = input_ids == self.config.image_token_id
        n_image_tokens = image_mask.sum()
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds = self.model.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0)
        video_mask = input_ids == self.config.video_token_id
        n_video_tokens = video_mask.sum()
        n_video_features = video_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )
        video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    univlr_mask = input_ids == self.config.univlr_id
    batch_indices, seq_positions = torch.nonzero(univlr_mask, as_tuple=True)
    num_univlr_tokens = int(univlr_mask.sum().item())

    if latent_targets is not None:
        selected_latent_targets = latent_targets.to(inputs_embeds.device, inputs_embeds.dtype)
    elif univlr_pixel_values is not None:
        selected_latent_targets = _make_latent_targets_from_images(
            self,
            univlr_pixel_values=univlr_pixel_values,
            univlr_image_grid_thw=univlr_image_grid_thw,
            univlr_token_counts=univlr_token_counts,
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
    else:
        selected_latent_targets = None

    if selected_latent_targets is None:
        raise ValueError("UniVLR Stage2 forward requires latent_targets or univlr target images.")
    if selected_latent_targets.shape[0] != num_univlr_tokens:
        raise ValueError(
            f"latent target count ({selected_latent_targets.shape[0]}) does not match "
            f"<|univlr|> token count ({num_univlr_tokens})."
        )

    inputs_embeds[batch_indices, seq_positions] = selected_latent_targets

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids += delta.to(position_ids.device)

    outputs = self.model.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:, -1, :]
    logits = self.lm_head(hidden_states)

    loss = None
    loss_ce = None
    loss_univlr = None
    loss_text_latent = None
    loss_image_latent = None
    loss_mode_switch = None
    loss_latent_nll = None
    loss_latent_kl = None
    loss_latent_det = None
    loss_latent_logvar_reg = None
    latent_logvar_q_mean = None
    latent_logvar_p_mean = None

    if labels is not None:
        logits = logits.float()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_labels = shift_labels.masked_fill(shift_labels == self.config.univlr_id, IGNORE_INDEX)
        latent_end_id = getattr(self.config, "univlr_latent_end_id", None)
        if latent_end_id is not None and int(latent_end_id) >= 0:
            shift_labels = shift_labels.masked_fill(shift_labels == int(latent_end_id), IGNORE_INDEX)

        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1).to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        alignment_hidden_states = _select_univlr_alignment_hidden(
            self,
            hidden_states,
            outputs.hidden_states,
        )
        pred_hidden = alignment_hidden_states[batch_indices, seq_positions - 1]
        (
            loss_univlr,
            loss_text_latent,
            loss_image_latent,
            loss_latent_nll,
            loss_latent_kl,
            loss_latent_det,
            loss_latent_logvar_reg,
            latent_logvar_q_mean,
            latent_logvar_p_mean,
        ) = (
            _compute_univlr_alignment_loss(
                self,
                pred_hidden,
                selected_latent_targets,
                latent_slot_types,
                latent_sample_ids,
            )
        )

        loss = loss_ce
        if loss_univlr is not None:
            loss = loss + float(getattr(self.config, "loss_univlr_lambda", 1.0)) * loss_univlr

    if not return_dict:
        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return Qwen2_5_VLUniVLROutputWithPast(
        loss=loss,
        loss_ce=loss_ce,
        loss_univlr=loss_univlr,
        loss_mode_switch=loss_mode_switch,
        loss_text_latent=loss_text_latent,
        loss_image_latent=loss_image_latent,
        loss_latent_nll=loss_latent_nll,
        loss_latent_kl=loss_latent_kl,
        loss_latent_det=loss_latent_det,
        loss_latent_logvar_reg=loss_latent_logvar_reg,
        latent_logvar_q_mean=latent_logvar_q_mean,
        latent_logvar_p_mean=latent_logvar_p_mean,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state=last_position_hidden_state,
    )


def qwen2_5_mixed_modality_forward_univlr_stage1_a2(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    latent_targets: Optional[torch.Tensor] = None,
    latent_slot_types: Optional[torch.Tensor] = None,
    latent_step_ids: Optional[torch.Tensor] = None,
    latent_sample_ids: Optional[torch.Tensor] = None,
    univlr_pixel_values: Optional[torch.Tensor] = None,
    univlr_image_grid_thw: Optional[torch.LongTensor] = None,
    univlr_token_counts: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[Tuple, Qwen2_5_VLUniVLROutputWithPast]:
    """Stage1-A2 self-replay / scheduled-sampling forward.

    The loss target is still the Stage1 teacher latent v_t, but the latent-slot
    input can be replaced by a detached model replay z_t. This trains the model
    on the distribution it will see during UniVLR inference.
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    if _univlr_alignment_requires_hidden_states(self):
        output_hidden_states = True
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    if pixel_values is None and pixel_values_videos is None:
        dummy_pixel = torch.zeros(784, 1176, device=self.model.visual.device, dtype=self.model.visual.dtype)
        dummy_grid = torch.tensor([[1, 28, 28]], device=self.model.visual.device)
        dummy_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
        inputs_embeds = inputs_embeds + dummy_embeds.mean().to(inputs_embeds.dtype) * 0

    if pixel_values is not None:
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0)
        image_mask = input_ids == self.config.image_token_id
        n_image_tokens = image_mask.sum()
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds = self.model.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0)
        video_mask = input_ids == self.config.video_token_id
        n_video_tokens = video_mask.sum()
        n_video_features = video_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )
        video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    univlr_mask = input_ids == self.config.univlr_id
    batch_indices, seq_positions = torch.nonzero(univlr_mask, as_tuple=True)
    num_univlr_tokens = int(univlr_mask.sum().item())

    if latent_targets is not None:
        selected_latent_targets = latent_targets.to(inputs_embeds.device, inputs_embeds.dtype)
    elif univlr_pixel_values is not None:
        selected_latent_targets = _make_latent_targets_from_images(
            self,
            univlr_pixel_values=univlr_pixel_values,
            univlr_image_grid_thw=univlr_image_grid_thw,
            univlr_token_counts=univlr_token_counts,
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
    else:
        selected_latent_targets = None

    if selected_latent_targets is None:
        raise ValueError("UniVLR Stage1-A2 forward requires latent_targets or univlr target images.")
    if selected_latent_targets.shape[0] != num_univlr_tokens:
        raise ValueError(
            f"latent target count ({selected_latent_targets.shape[0]}) does not match "
            f"<|univlr|> token count ({num_univlr_tokens})."
        )

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids += delta.to(position_ids.device)

    replay_prob = _get_univlr_a2_replay_prob(self)
    replay_mode = str(
        getattr(
            self.config,
            "univlr_a2_replay_mode",
            getattr(self.config, "univlr_replay_mode", "prefix"),
        )
    ).lower()
    replay_mask = _make_univlr_replay_mask(
        num_univlr_tokens,
        latent_step_ids,
        latent_sample_ids,
        replay_prob,
        replay_mode,
        inputs_embeds.device,
    )
    replay_ratio = replay_mask.float().mean() if replay_mask.numel() else inputs_embeds.new_zeros(())

    selected_latent_inputs = selected_latent_targets
    if bool(replay_mask.any().item()):
        teacher_inputs_embeds = inputs_embeds.clone()
        teacher_inputs_embeds[batch_indices, seq_positions] = selected_latent_targets
        with torch.no_grad():
            replay_outputs = self.model.language_model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=teacher_inputs_embeds,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=_univlr_alignment_requires_hidden_states(self),
                return_dict=True,
                cache_position=cache_position,
            )
            replay_source_hidden = _select_univlr_alignment_hidden(
                self,
                replay_outputs[0],
                replay_outputs.hidden_states,
            )
            replay_latents = _project_to_univlr_space(
                self,
                replay_source_hidden[batch_indices, seq_positions - 1],
            )
            replay_latents = replay_latents.detach().to(inputs_embeds.device, inputs_embeds.dtype)

        selected_latent_inputs = selected_latent_targets.clone()
        selected_latent_inputs[replay_mask] = replay_latents[replay_mask]

    inputs_embeds = inputs_embeds.clone()
    inputs_embeds[batch_indices, seq_positions] = selected_latent_inputs

    outputs = self.model.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:, -1, :]
    logits = self.lm_head(hidden_states)

    loss = None
    loss_ce = None
    loss_univlr = None
    loss_text_latent = None
    loss_image_latent = None
    loss_mode_switch = None
    loss_latent_nll = None
    loss_latent_kl = None
    loss_latent_det = None
    loss_latent_logvar_reg = None
    latent_logvar_q_mean = None
    latent_logvar_p_mean = None

    if labels is not None:
        logits = logits.float()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_labels = shift_labels.masked_fill(shift_labels == self.config.univlr_id, IGNORE_INDEX)
        latent_end_id = getattr(self.config, "univlr_latent_end_id", None)
        if latent_end_id is not None and int(latent_end_id) >= 0:
            shift_labels = shift_labels.masked_fill(shift_labels == int(latent_end_id), IGNORE_INDEX)

        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1).to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        alignment_hidden_states = _select_univlr_alignment_hidden(
            self,
            hidden_states,
            outputs.hidden_states,
        )
        pred_hidden = alignment_hidden_states[batch_indices, seq_positions - 1]
        (
            loss_univlr,
            loss_text_latent,
            loss_image_latent,
            loss_latent_nll,
            loss_latent_kl,
            loss_latent_det,
            loss_latent_logvar_reg,
            latent_logvar_q_mean,
            latent_logvar_p_mean,
        ) = (
            _compute_univlr_alignment_loss(
                self,
                pred_hidden,
                selected_latent_targets,
                latent_slot_types,
                latent_sample_ids,
            )
        )

        loss = loss_ce
        if loss_univlr is not None:
            loss = loss + float(getattr(self.config, "loss_univlr_lambda", 1.0)) * loss_univlr

    if not return_dict:
        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return Qwen2_5_VLUniVLROutputWithPast(
        loss=loss,
        loss_ce=loss_ce,
        loss_univlr=loss_univlr,
        loss_mode_switch=loss_mode_switch,
        loss_text_latent=loss_text_latent,
        loss_image_latent=loss_image_latent,
        loss_latent_nll=loss_latent_nll,
        loss_latent_kl=loss_latent_kl,
        loss_latent_det=loss_latent_det,
        loss_latent_logvar_reg=loss_latent_logvar_reg,
        latent_logvar_q_mean=latent_logvar_q_mean,
        latent_logvar_p_mean=latent_logvar_p_mean,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state=last_position_hidden_state,
        univlr_replay_ratio=replay_ratio.detach(),
    )
