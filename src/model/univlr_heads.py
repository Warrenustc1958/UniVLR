import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from typing import Tuple

class UniVLRHead(nn.Module):
    """
        The simplest mlp w/o up_proj
    """
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.ln_q = LayerNorm(hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x))
        return x


from transformers.activations import ACT2FN
class UniVLRHeadGLU(nn.Module):
    ''' 
        The Gated Liner Unit MLP
    '''
    def __init__(self, hidden_size, intermediate_size, hidden_act, bias: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        # 11008 for 3b; 18944 for 7b; 27648 for 32b
        self.intermediate_size = intermediate_size  
        self.hidden_act = hidden_act
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[self.hidden_act]    #silu

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class UniVLRHeadVAE(nn.Module):
    """
        Lightweight variational latent head.
        `forward()` stays backward-compatible by returning the posterior mean.
    """

    def __init__(self, hidden_size: int, logvar_min: float = -6.0, logvar_max: float = 2.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.input_ln = LayerNorm(hidden_size, eps=1e-6)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.hidden_ln = LayerNorm(hidden_size, eps=1e-6)
        self.act = nn.GELU()
        self.mu_proj = nn.Linear(hidden_size, hidden_size)
        self.logvar_proj = nn.Linear(hidden_size, hidden_size)

    @staticmethod
    def _layer_norm_fp32(x: torch.Tensor, layer: LayerNorm) -> torch.Tensor:
        # Keep the variational statistics path in fp32 so bf16 mixed precision
        # training does not trip over LayerNorm parameter/input dtype mismatches.
        return F.layer_norm(
            x.to(torch.float32),
            layer.normalized_shape,
            layer.weight.to(torch.float32) if layer.weight is not None else None,
            layer.bias.to(torch.float32) if layer.bias is not None else None,
            layer.eps,
        )

    @staticmethod
    def _linear_fp32(x: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        return F.linear(
            x.to(torch.float32),
            layer.weight.to(torch.float32),
            layer.bias.to(torch.float32) if layer.bias is not None else None,
        )

    def _shared(self, x: torch.Tensor) -> torch.Tensor:
        x = self._layer_norm_fp32(x, self.input_ln)
        x = self.act(self._linear_fp32(x, self.fc1))
        x = self._linear_fp32(x, self.fc2)
        x = self._layer_norm_fp32(x, self.hidden_ln)
        return x

    def distribution(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self._shared(x)
        mu = self._linear_fp32(h, self.mu_proj)
        logvar = self._linear_fp32(h, self.logvar_proj).clamp_(
            min=self.logvar_min,
            max=self.logvar_max,
        )
        return mu, logvar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.distribution(x)
        # The variational path runs in fp32 for stability, but callers often
        # write the projected hidden states back into mixed-precision activations.
        return mu.to(dtype=x.dtype)


class UniVLRHeadPrior(nn.Module):
    """
        Tiny teacher-side prior head operating on latent targets that are already
        in the Qwen hidden space after the built-in visual merger.
    """

    def __init__(self, hidden_size: int, logvar_min: float = -6.0, logvar_max: float = 2.0) -> None:
        super().__init__()
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.input_ln = LayerNorm(hidden_size, eps=1e-6)
        self.mu_proj = nn.Linear(hidden_size, hidden_size)
        self.logvar_proj = nn.Linear(hidden_size, hidden_size)

    def distribution(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = UniVLRHeadVAE._layer_norm_fp32(x, self.input_ln)
        mu = UniVLRHeadVAE._linear_fp32(x, self.mu_proj)
        logvar = UniVLRHeadVAE._linear_fp32(x, self.logvar_proj).clamp_(
            min=self.logvar_min,
            max=self.logvar_max,
        )
        return mu, logvar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.distribution(x)
        return mu.to(dtype=x.dtype)
    



# from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2RMSNorm
# class UniVLRHeadRMS(nn.Module):
#     """
#         Modified Patch Merger from transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
#         This inherits from the mm projector of qwen 2.5 vl
#     """
#     def __init__(self, hidden_size: int) -> None:
#         super().__init__()
#         self.ln_q = Qwen2RMSNorm(hidden_size, eps=1e-6)
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.GELU(),
#             nn.Linear(hidden_size, hidden_size),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.mlp(self.ln_q(x))
#         return x
