"""Loads HF Llama safetensors straight onto the target device, fusing as it goes.

Fused at load time:
- q, k, v projections -> one [q + 2*kv, hidden] matrix. One matmul per layer
  instead of three. At small batch, each matmul is a separate kernel launch,
  and launch overhead is what dominates decode.
- gate, up -> one [2*intermediate, hidden] matrix. Same reason.

Peak memory: tensors are read one at a time with safe_open, and each layer's
pieces are freed right after they're fused. Loading whole shards first
would need ~2x the model size on the device.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from prefixserve.model.config import LlamaConfig


class WeightLoadError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LayerWeights:
    input_norm: torch.Tensor  # [hidden]
    qkv: torch.Tensor  # [q_size + 2 * kv_size, hidden]
    o: torch.Tensor  # [hidden, q_size]
    post_norm: torch.Tensor  # [hidden]
    gate_up: torch.Tensor  # [2 * intermediate, hidden]
    down: torch.Tensor  # [hidden, intermediate]


@dataclass(frozen=True, slots=True)
class LlamaWeights:
    embed: torch.Tensor  # [vocab, hidden]
    layers: list[LayerWeights]
    norm: torch.Tensor  # [hidden]
    lm_head: torch.Tensor  # [vocab, hidden]; same tensor as embed when tied

    def num_bytes(self) -> int:
        seen: set[int] = set()
        total = 0
        tensors = [self.embed, self.norm, self.lm_head]
        for lw in self.layers:
            tensors += [lw.input_norm, lw.qkv, lw.o, lw.post_norm, lw.gate_up, lw.down]
        for t in tensors:
            if t.data_ptr() not in seen:  # tied lm_head counted once
                seen.add(t.data_ptr())
                total += t.numel() * t.element_size()
        return total


def _shard_files(path: Path) -> list[Path]:
    index = path / "model.safetensors.index.json"
    if index.exists():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        return [path / n for n in names]
    single = path / "model.safetensors"
    if single.exists():
        return [single]
    raise WeightLoadError(f"no safetensors weights in {path}")


class _Reader:
    """Name -> tensor across shards, loaded lazily onto the device."""

    def __init__(self, files: list[Path], device: torch.device, dtype: torch.dtype, stack: ExitStack) -> None:
        self._where: dict[str, object] = {}
        self._device = device
        self._dtype = dtype
        for f in files:
            # safetensors wants a plain device string ("cuda:0", "cpu").
            handle = stack.enter_context(safe_open(str(f), framework="pt", device=str(device)))
            for name in handle.keys():  # noqa: SIM118 (safe_open isn't a dict)
                self._where[name] = handle
        self._used: set[str] = set()

    def has(self, name: str) -> bool:
        return name in self._where

    def get(self, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        handle = self._where.get(name)
        if handle is None:
            raise WeightLoadError(f"missing weight {name}")
        t: torch.Tensor = handle.get_tensor(name)  # type: ignore[attr-defined]
        if tuple(t.shape) != shape:
            raise WeightLoadError(f"{name}: shape {tuple(t.shape)}, expected {shape}")
        self._used.add(name)
        return t.to(dtype=self._dtype)

    def unused(self) -> Iterator[str]:
        return (n for n in self._where if n not in self._used)


def load_llama_weights(
    path: str | Path, cfg: LlamaConfig, device: torch.device, dtype: torch.dtype
) -> LlamaWeights:
    path = Path(path)
    h, q, kv, inter = cfg.hidden_size, cfg.q_size, cfg.kv_size, cfg.intermediate_size
    with ExitStack() as stack:
        r = _Reader(_shard_files(path), device, dtype, stack)
        embed = r.get("model.embed_tokens.weight", (cfg.vocab_size, h))
        layers: list[LayerWeights] = []
        for i in range(cfg.num_layers):
            p = f"model.layers.{i}."
            qkv = torch.cat([
                r.get(p + "self_attn.q_proj.weight", (q, h)),
                r.get(p + "self_attn.k_proj.weight", (kv, h)),
                r.get(p + "self_attn.v_proj.weight", (kv, h)),
            ])
            gate_up = torch.cat([
                r.get(p + "mlp.gate_proj.weight", (inter, h)),
                r.get(p + "mlp.up_proj.weight", (inter, h)),
            ])
            layers.append(LayerWeights(
                input_norm=r.get(p + "input_layernorm.weight", (h,)),
                qkv=qkv,
                o=r.get(p + "self_attn.o_proj.weight", (h, q)),
                post_norm=r.get(p + "post_attention_layernorm.weight", (h,)),
                gate_up=gate_up,
                down=r.get(p + "mlp.down_proj.weight", (h, inter)),
            ))
        norm = r.get("model.norm.weight", (h,))
        if cfg.tie_word_embeddings:
            lm_head = embed
        else:
            lm_head = r.get("lm_head.weight", (cfg.vocab_size, h))
        # Rotary inv_freq buffers are recomputed from config, so they're fine
        # to skip. Anything else left over means a layout we don't understand.
        extra = [n for n in r.unused() if not n.endswith("rotary_emb.inv_freq")]
        if cfg.tie_word_embeddings:
            extra = [n for n in extra if n != "lm_head.weight"]
        if extra:
            raise WeightLoadError(f"unexpected weights: {extra[:5]}{'...' if len(extra) > 5 else ''}")
    return LlamaWeights(embed=embed, layers=layers, norm=norm, lm_head=lm_head)
