import json
import os
from typing import List, Optional, Tuple

import numpy as np
import torch

from src.configs import DecoderConfigs, ModelConfigs
from src.models.base_model import BaseModel


class BaselineMaskedHonestHead(BaseModel):
    def __init__(
        self,
        model_configs: ModelConfigs,
        decoder_configs: DecoderConfigs,
    ):
        super().__init__(model_configs, decoder_configs)

        self._load_honest_heads()
        print("Honest heads: ", self.honest_heads)
        print("Honest head scores: ", self.honest_head_scores)

    def _load_honest_heads(self):
        self.num_honest_heads = self.decoder_configs.configs.num_honest_heads

        model_base_name = self.model_configs.configs.model_name_or_path.split("/")[1]
        with open(
            os.path.join(
                self.decoder_configs.configs.honest_heads_dir,
                f"{model_base_name}.json",
            )
        ) as file:
            head_list = json.loads(file.readline())

        honest_heads = {}
        for feature_name in self.decoder_configs.configs.feature_names:
            if feature_name in head_list:
                for layer_head, value in head_list[feature_name].items():
                    if layer_head not in honest_heads:
                        honest_heads[layer_head] = []
                    honest_heads[layer_head].append(value)

        if self.decoder_configs.configs.aggregation_method == "mean":
            honest_heads = {k: np.mean(v) for k, v in honest_heads.items()}
        elif self.decoder_configs.configs.aggregation_method == "max":
            honest_heads = {k: np.max(v) for k, v in honest_heads.items()}
        else:
            raise ValueError(
                f"Unknown honest head aggregation method: {self.decoder_configs.configs.aggregation_method}"
            )

        honest_heads = sorted(honest_heads.items(), key=lambda x: x[1], reverse=True)
        self.honest_heads = [[int(ll) for ll in l[0].split("-")] for l in honest_heads][
            : self.num_honest_heads
        ]
        self.honest_head_scores = [l[1] for l in honest_heads][: self.num_honest_heads]

    def generate(
        self,
        inputs,
        return_attentions: bool = False,
    ) -> dict:
        return self._generate(
            inputs, return_attentions=return_attentions, block_list=self.honest_heads
        )

    def lm_score(
        self,
        prompt,
        answer,
    ):
        prompt = prompt["prompted_question"][0]
        with torch.no_grad():
            if type(prompt) == list:
                input_text = prompt + [answer]
            else:
                input_text = prompt + answer
            input_ids = self._verbalise_input(input_text).to(self.model.device)
            prefix_ids = self._verbalise_input(prompt).to(self.model.device)
            continue_ids = input_ids[0, prefix_ids.shape[-1] :]

            outputs = self.model(input_ids, block_list=self.honest_heads)[0]
            outputs = outputs.squeeze(0).log_softmax(-1)  # logits to log probs

            # skip tokens in the prompt -- we only care about the answer
            outputs = outputs[prefix_ids.shape[-1] - 1 : -1, :]

            # get logprobs for each token in the answer
            log_probs = outputs[range(outputs.shape[0]), continue_ids].sum().item()

        return log_probs
