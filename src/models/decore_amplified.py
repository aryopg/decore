from typing import List, Optional, Tuple

import copy
import random
import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.configs import DecoderConfigs, ModelConfigs

from src.models.base_model import BaseModel


class DeCoReAmplified(BaseModel):
    def __init__(
        self,
        model_configs: ModelConfigs,
        decoder_configs: DecoderConfigs,
    ):
        super().__init__(model_configs, decoder_configs)

        self.num_retrieval_heads = self.decoder_configs.configs.num_retrieval_heads
        self.retrieval_heads = self._load_retrieval_heads()
        self.random_heads = self._construct_random_head()
        print("Retrieval heads: ", self.retrieval_heads)
        print("Random heads: ", self.random_heads)

    def _load_retrieval_heads(self):
        model_base_name = self.model_configs.configs.model_name_or_path.split("/")[1]

        with open(
            os.path.join(
                self.decoder_configs.configs.retrieval_heads_dir,
                f"{model_base_name}.json",
            )
        ) as file:
            head_list = json.loads(file.readline())

        stable_block_list = [(l[0], np.mean(l[1])) for l in head_list.items()]
        stable_block_list = sorted(stable_block_list, key=lambda x: x[1], reverse=True)
        return [[int(ll) for ll in l[0].split("-")] for l in stable_block_list][
            : self.num_retrieval_heads
        ]

    def _construct_random_head(self):
        results = []
        seed_list = [
            i for i in range(32)
        ]  # FIXME: 32 is hardcoded, copied from Retrieval_Head repo
        random.shuffle(seed_list)
        while len(results) < self.num_retrieval_heads:
            l, h = random.choices(seed_list, k=2)
            if (l, h) in results or (l, h) in self.retrieval_heads:
                continue
            else:
                results.append((l, h))
        return results

    def generate(
        self,
        inputs,
    ) -> str:
        self.model.eval()

        inputs = self._verbalise_input(inputs).to(self.model.device)

        # Predict
        with torch.inference_mode():
            input_logits = self.model(
                input_ids=inputs[:, :-1], use_cache=True, return_dict=True
            )
            generated_ids = []
            last_input_token = inputs[:, -1]
            base_past_kv = copy.deepcopy(input_logits.past_key_values)
            random_mask_past_kv = copy.deepcopy(input_logits.past_key_values)
            retrieval_mask_past_kv = copy.deepcopy(input_logits.past_key_values)
            for _ in range(self.max_new_tokens):
                last_input_token = last_input_token.view(1, 1)

                base_outputs = self.model(
                    input_ids=last_input_token,
                    past_key_values=base_past_kv,
                    use_cache=True,
                    attn_mode="torch",
                )
                random_mask_outputs = self.model(
                    input_ids=last_input_token,
                    past_key_values=random_mask_past_kv,
                    use_cache=True,
                    attn_mode="torch",
                    block_list=self.random_heads,
                )
                retrieval_mask_outputs = self.model(
                    input_ids=last_input_token,
                    past_key_values=retrieval_mask_past_kv,
                    use_cache=True,
                    attn_mode="torch",
                    block_list=self.retrieval_heads,
                )

                base_past_kv = base_outputs.past_key_values
                random_mask_past_kv = random_mask_outputs.past_key_values
                retrieval_mask_past_kv = retrieval_mask_outputs.past_key_values

                # Random masked output is supposed to be more faithful than retrieval masked output

                next_token_logits = base_outputs.logits[
                    0, -1
                ] + self.decoder_configs.configs.alpha * (
                    random_mask_outputs.logits[0, -1]
                    - retrieval_mask_outputs.logits[0, -1]
                )

                last_input_token = next_token_logits.argmax()
                generated_ids.append(last_input_token.item())
                if last_input_token.item() == self.tokenizer.eos_token_id:
                    break
            decoded_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

        return decoded_text

    def lm_score(
        self,
        prompt,
        answer,
    ):
        with torch.no_grad():
            if type(prompt) == list:
                input_text = prompt + [answer]
            else:
                input_text = prompt + answer
            input_ids = self._verbalise_input(input_text).to(self.model.device)
            prefix_ids = self._verbalise_input(prompt).to(self.model.device)
            continue_ids = input_ids[0, prefix_ids.shape[-1] :]

            base_outputs = self.model(input_ids)[0]
            random_mask_outputs = self.model(input_ids, block_list=self.random_heads)[0]
            retrieval_mask_outputs = self.model(
                input_ids, block_list=self.retrieval_heads
            )[0]

            base_logits = base_outputs[0, prefix_ids.shape[-1] - 1 : -1, :]
            random_mask_logits = random_mask_outputs[
                0, prefix_ids.shape[-1] - 1 : -1, :
            ]
            retrieval_mask_logits = retrieval_mask_outputs[
                0, prefix_ids.shape[-1] - 1 : -1, :
            ]

            # base_logits = base_logits.log_softmax(dim=-1)
            # hallucinated_logits = hallucinated_logits.log_softmax(dim=-1)
            diff_logits = base_logits + self.decoder_configs.configs.alpha * (
                random_mask_logits - retrieval_mask_logits
            )

            diff_logits = diff_logits.log_softmax(dim=-1)

            log_probs = (
                diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()
            )

        return log_probs
