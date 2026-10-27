from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
from dataclasses import dataclass
from typing import Callable
import torch
from torch.nn import functional as F

def sdpa_attention_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Equivalent attention using PyTorch's fused SDPA dispatch."""
    batch, tokens, _ = x.shape
    qkv_bias = None
    if self.q_bias is not None:
        qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
    qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    qkv = qkv.reshape(batch, tokens, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = (qkv[0], qkv[1], qkv[2])
    q = q * self.scale
    dropout_p = float(self.attn_drop.p) if self.training else 0.0
    output = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=False, scale=1.0)
    output = output.transpose(1, 2).reshape(batch, tokens, -1)
    return self.proj_drop(self.proj(output))

def _is_videomae_attention(module: torch.nn.Module) -> bool:
    cls = type(module)
    return cls.__name__ == 'Attention' and 'modeling_videomaev2' in cls.__module__ and all((hasattr(module, name) for name in ('qkv', 'q_bias', 'v_bias', 'attn_drop', 'proj', 'proj_drop', 'scale', 'num_heads')))

@dataclass
class SDPAPatch:
    """A reversible patch over the unique Attention classes found in a model."""
    originals: list[tuple[type, Callable]]

    @property
    def class_count(self) -> int:
        return len(self.originals)

    def restore(self) -> None:
        for cls, original in self.originals:
            cls.forward = original

    def apply(self) -> None:
        for cls, _ in self.originals:
            cls.forward = sdpa_attention_forward

def enable_sdpa(model: torch.nn.Module) -> SDPAPatch:
    """Patch only VideoMAEv2 ``Attention`` classes reachable from ``model``."""
    classes = {type(module) for module in model.modules() if _is_videomae_attention(module)}
    if not classes:
        raise ValueError('no VideoMAEv2 Attention modules found; refusing broad monkeypatch')
    originals = [(cls, cls.forward) for cls in classes]
    handle = SDPAPatch(originals)
    handle.apply()
    return handle
