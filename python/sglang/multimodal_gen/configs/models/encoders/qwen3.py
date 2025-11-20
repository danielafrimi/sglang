# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.encoders.base import (
    TextEncoderArchConfig,
    TextEncoderConfig,
)


@dataclass
class Qwen3ArchConfig(TextEncoderArchConfig):
    """
    Minimal Qwen3 arch config. Values will be populated from checkpoint config.json
    via update_model_arch during load.
    """
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int | None = None
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    rope_scaling: float | None = None
    attention_bias: bool = False
    head_dim: int | None = None
    # Provide common stacked param mapping (qkv, gate/up)
    stacked_params_mapping: list[tuple[str, str, str]] = field(
        default_factory=lambda: [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
    )


@dataclass
class Qwen3Config(TextEncoderConfig):
    arch_config: TextEncoderArchConfig = field(default_factory=Qwen3ArchConfig)
    # prefix can be used if namespacing weights is necessary
    # prefix: str = "qwen3"


