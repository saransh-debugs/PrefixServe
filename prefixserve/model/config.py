"""Llama config, read from a HF model folder (config.json + generation_config.json).

Anything we don't support raises instead of being ignored. A silently
ignored rope_scaling gives a model that loads fine and then writes garbage
past a few thousand tokens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class UnsupportedModelError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Llama3RopeScaling:
    """Llama-3.1's "llama3" RoPE scaling. Llama-3.0 has none."""

    factor: float
    low_freq_factor: float
    high_freq_factor: float
    original_max_position_embeddings: int


@dataclass(frozen=True, slots=True)
class LlamaConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: Llama3RopeScaling | None
    max_position_embeddings: int
    tie_word_embeddings: bool
    stop_token_ids: frozenset[int]

    @property
    def q_size(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_kv_heads * self.head_dim

    @classmethod
    def from_dir(cls, path: str | Path) -> LlamaConfig:
        path = Path(path)
        raw: dict[str, Any] = json.loads((path / "config.json").read_text())
        gen_path = path / "generation_config.json"
        gen: dict[str, Any] = json.loads(gen_path.read_text()) if gen_path.exists() else {}
        return cls.from_dicts(raw, gen)

    @classmethod
    def from_dicts(cls, raw: dict[str, Any], gen: dict[str, Any] | None = None) -> LlamaConfig:
        if raw.get("model_type") != "llama":
            raise UnsupportedModelError(f"model_type {raw.get('model_type')!r} is not 'llama'")
        num_heads = int(raw["num_attention_heads"])
        num_kv_heads = int(raw.get("num_key_value_heads", num_heads))
        if num_heads % num_kv_heads:
            raise UnsupportedModelError(f"{num_heads} query heads not divisible by {num_kv_heads} KV heads")
        hidden = int(raw["hidden_size"])
        # 3.2-1B sets head_dim explicitly; older configs don't.
        head_dim = int(raw.get("head_dim") or hidden // num_heads)
        # Two layouts for the same thing. transformers 4 (and Meta's Hub
        # checkpoints): top-level rope_theta + rope_scaling. transformers 5
        # (anything re-saved with it): one rope_parameters dict holding both.
        # Reading only the old keys from a new file silently gives theta
        # 10000 and no scaling: a model that loads fine and outputs garbage.
        rope = raw.get("rope_parameters") or raw.get("rope_scaling")
        theta_raw = (raw.get("rope_parameters") or {}).get("rope_theta", raw.get("rope_theta"))
        theta = float(theta_raw) if theta_raw is not None else 10000.0  # Llama-1/2 default
        if raw.get("attention_bias") or raw.get("mlp_bias"):
            raise UnsupportedModelError("biases in attention/MLP are not supported")

        # Llama-3.1 stops on several ids (<|end_of_text|>, <|eom_id|>,
        # <|eot_id|>). generation_config.json has the full list; config.json
        # often has only one of them.
        eos = (gen or {}).get("eos_token_id", raw.get("eos_token_id"))
        stop = frozenset(int(t) for t in (eos if isinstance(eos, list) else [eos]) if t is not None)

        return cls(
            vocab_size=int(raw["vocab_size"]),
            hidden_size=hidden,
            intermediate_size=int(raw["intermediate_size"]),
            num_layers=int(raw["num_hidden_layers"]),
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rms_norm_eps=float(raw["rms_norm_eps"]),
            rope_theta=theta,
            rope_scaling=_parse_rope_scaling(rope),
            max_position_embeddings=int(raw["max_position_embeddings"]),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
            stop_token_ids=stop,
        )


def _parse_rope_scaling(rs: dict[str, Any] | None) -> Llama3RopeScaling | None:
    if rs is None:
        return None
    kind = rs.get("rope_type", rs.get("type"))
    if kind in (None, "default"):
        return None
    if kind != "llama3":
        raise UnsupportedModelError(f"rope_scaling type {kind!r} is not supported")
    return Llama3RopeScaling(
        factor=float(rs["factor"]),
        low_freq_factor=float(rs["low_freq_factor"]),
        high_freq_factor=float(rs["high_freq_factor"]),
        original_max_position_embeddings=int(rs["original_max_position_embeddings"]),
    )
