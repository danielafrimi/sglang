# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any, Optional

import torch
from transformers import AutoModel

from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput, TextEncoderConfig
from sglang.multimodal_gen.runtime.models.encoders.base import TextEncoder


class HFDiffEncoderTextWrapper(TextEncoder):
    """
    Wrap a HuggingFace text model (e.g., DiffEncoderModel) and expose hidden states as BaseEncoderOutput.

    - If the HF model has an `.encoder` attribute (e.g., DiffEncoderModel), we call it directly to get the
      encoder's last_hidden_state.
    - Otherwise, we call the model with output_hidden_states=True and use the last layer hidden states.
    """

    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__(config)
        # model_path is injected into arch_config.extra_attrs by the loader
        model_path: Optional[str] = getattr(self.config.arch_config, "model_path", None)
        if model_path is None:
            raise ValueError("model_path not provided in config.arch_config for HFDiffEncoderTextWrapper")
        trust_remote_code: bool = bool(getattr(self.config.arch_config, "trust_remote_code", True))
        dtype: Optional[torch.dtype] = getattr(self.config.arch_config, "compute_dtype", None)
        device: Optional[torch.device | str] = getattr(self.config.arch_config, "device", None)

        hf_model = AutoModel.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        if device is not None:
            hf_model = hf_model.to(device=device)
        if dtype is not None:
            try:
                hf_model = hf_model.to(dtype=dtype)
            except Exception:
                pass
        self.hf_model = hf_model.eval()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs: Any,
    ) -> BaseEncoderOutput:
        # Prefer direct encoder call if available (DiffEncoderModel exposes .encoder)
        if hasattr(self.hf_model, "encoder"):
            encoder = getattr(self.hf_model, "encoder")
            enc_out = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )
            last_hidden_state = enc_out.last_hidden_state
            return BaseEncoderOutput(
                last_hidden_state=last_hidden_state,
                hidden_states=(last_hidden_state,) if output_hidden_states else None,
            )

        # Fallback: rely on HF outputs.hidden_states
        outputs = self.hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            **kwargs,
        )
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            last_hidden_state = outputs.hidden_states[-1]
        elif hasattr(outputs, "last_hidden_state"):
            last_hidden_state = outputs.last_hidden_state
        else:
            raise RuntimeError("HF model did not return hidden states or last_hidden_state.")

        return BaseEncoderOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else (last_hidden_state,),
        )

    @torch.no_grad()
    def load_weights(self, *args, **kwargs):
        # We load weights via AutoModel in __init__, so nothing to do here.
        return set()

EntryClass = ["HFDiffEncoderTextWrapper"]


