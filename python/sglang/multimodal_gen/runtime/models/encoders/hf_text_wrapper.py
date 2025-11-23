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

        hf_model = AutoModel.from_pretrained(model_path, trust_remote_code=trust_remote_code, )
        # hf_model = hf_model.to(device=device) # todo we need to pass the model into the same device, and this is not as
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

        # Pass output_hidden_states to the HF model/encoder if requested or implied
        # We force it True here because we typically need hidden states for downstream tasks.
        return self.hf_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True, # This is important to return
            **kwargs,
        )

       


    @torch.no_grad()
    def load_weights(self, *args, **kwargs):
        # We load weights via AutoModel in __init__, so nothing to do here.
        return set()

EntryClass = ["HFDiffEncoderTextWrapper"]


