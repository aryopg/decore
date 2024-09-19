from typing import List, Optional, Tuple

import copy
import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.configs import DecoderConfigs, ModelConfigs

from src.models.base_model import BaseModel


class ContextAwareDecoding(BaseModel):
    def __init__(
        self,
        model_configs: ModelConfigs,
        decoder_configs: DecoderConfigs,
    ):
        super().__init__(model_configs, decoder_configs)

    def generate(
        self,
        inputs,
        return_attentions: bool = False,
    ) -> dict:
        assert (
            not return_attentions
        ), "Return attentions not supported for DeCoReVanilla"
        self.model.eval()

        prompt = inputs["prompted_question"][0]
        prompt_wo_context = inputs["prompted_question_wo_context"][0]

        if len(inputs["verbalised_instruction"][0]):
            use_system_prompt = True
        else:
            use_system_prompt = False

        tokenised_inputs = self._verbalise_input(
            prompt, use_system_prompt=use_system_prompt
        ).to(self.model.device)

        tokenised_inputs_wo_context = self._verbalise_input(
            prompt_wo_context, use_system_prompt=use_system_prompt
        ).to(self.model.device)

        # Predict
        with torch.inference_mode():
            input_logits = self.model(
                input_ids=tokenised_inputs[:, :-1], use_cache=True, return_dict=True
            )
            input_hallucinated_logits = self.model(
                input_ids=tokenised_inputs_wo_context[:, :-1],
                use_cache=True,
                return_dict=True,
            )
            generated_ids = []
            last_input_token = tokenised_inputs[:, -1]
            base_past_kv = copy.deepcopy(input_logits.past_key_values)
            hallucinated_past_kv = copy.deepcopy(
                input_hallucinated_logits.past_key_values
            )
            for _ in range(self.max_new_tokens):
                last_input_token = last_input_token.view(1, 1)

                base_outputs = self.model(
                    input_ids=last_input_token,
                    past_key_values=base_past_kv,
                    use_cache=True,
                    attn_mode=self.attn_mode,
                )
                hallucinated_outputs = self.model(
                    input_ids=last_input_token,
                    past_key_values=hallucinated_past_kv,
                    use_cache=True,
                    attn_mode=self.attn_mode,
                )

                base_past_kv = base_outputs.past_key_values
                hallucinated_past_kv = hallucinated_outputs.past_key_values

                base_logits = base_outputs.logits[0, -1]
                hallucinated_logits = hallucinated_outputs.logits[0, -1]

                next_token_logits = (
                    (1 + self.decoder_configs.configs.alpha) * base_logits
                    - self.decoder_configs.configs.alpha * hallucinated_logits
                )

                last_input_token = next_token_logits.argmax()
                generated_ids.append(last_input_token.item())
                if last_input_token.item() == self.tokenizer.eos_token_id:
                    break
            decoded_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

        return {"decoded_text": decoded_text, "attentions": {}}

    def lm_score(
        self,
        prompt,
        answer,
    ):
        raise NotImplementedError("Context Aware Decoding does not support LM scoring")
