"""Config parsing. Pure Python, no torch."""

from __future__ import annotations

from typing import Any

import pytest

from prefixserve.model.config import Llama3RopeScaling, LlamaConfig, UnsupportedModelError

LLAMA31_SCALING = {
    "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192, "rope_type": "llama3",
}
WANT_SCALING = Llama3RopeScaling(8.0, 1.0, 4.0, 8192)


def base(**kw: Any) -> dict[str, Any]:
    cfg = {
        "model_type": "llama", "vocab_size": 128256, "hidden_size": 4096, "intermediate_size": 14336,
        "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8,
        "rms_norm_eps": 1e-5, "max_position_embeddings": 131072, "eos_token_id": 128001,
    }
    cfg.update(kw)
    return cfg


def test_transformers4_layout() -> None:
    """What Meta's Llama-3.1-8B-Instruct config.json looks like on the Hub."""
    c = LlamaConfig.from_dicts(
        base(rope_theta=500000.0, rope_scaling=LLAMA31_SCALING),
        {"eos_token_id": [128001, 128008, 128009]},
    )
    assert c.rope_theta == 500000.0 and c.rope_scaling == WANT_SCALING
    assert c.head_dim == 128 and c.q_size == 4096 and c.kv_size == 1024
    assert c.stop_token_ids == {128001, 128008, 128009}  # from generation_config, not config


def test_transformers5_layout() -> None:
    """transformers 5 moves theta and scaling into one rope_parameters dict."""
    c = LlamaConfig.from_dicts(base(rope_parameters={**LLAMA31_SCALING, "rope_theta": 500000.0}))
    assert c.rope_theta == 500000.0 and c.rope_scaling == WANT_SCALING


def test_no_scaling() -> None:
    """Llama-3.0: theta, no scaling. Also the v5 form of that."""
    for raw in (base(rope_theta=500000.0), base(rope_parameters={"rope_type": "default", "rope_theta": 500000.0})):
        c = LlamaConfig.from_dicts(raw)
        assert c.rope_theta == 500000.0 and c.rope_scaling is None


def test_stop_ids_fall_back_to_config() -> None:
    assert LlamaConfig.from_dicts(base(rope_theta=1.0)).stop_token_ids == {128001}


@pytest.mark.parametrize("bad", [
    {"model_type": "mistral"},
    {"rope_scaling": {"rope_type": "yarn", "factor": 4.0}},
    {"attention_bias": True},
    {"num_key_value_heads": 5},
])
def test_refuses_what_it_cant_run(bad: dict[str, Any]) -> None:
    with pytest.raises(UnsupportedModelError):
        LlamaConfig.from_dicts(base(**bad))
