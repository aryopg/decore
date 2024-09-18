from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn import functional as F

from src.configs import DecoderConfigs, ModelConfigs
from src.models.base_model import BaseModel

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


class ActivationDecoding(BaseModel):
    def __init__(
        self,
        model_configs: ModelConfigs,
        decoder_configs: DecoderConfigs,
    ):
        super().__init__(model_configs, decoder_configs)

        self.dola_layers = self.decoder_configs.configs.dola_layers
        self.alpha = self.decoder_configs.configs.alpha

        self.post_softmax = self.decoder_configs.configs.post_softmax

        self.num_layers = len(self.model.model.layers)
        mid_point = self.num_layers // 2

        if self.dola_layers == "low":
            self.candidate_premature_layers = list(range(0, mid_point, 2)) + [
                self.num_layers
            ]
        elif self.dola_layers == "high":
            self.candidate_premature_layers = list(
                range(mid_point, self.num_layers, 2)
            ) + [self.num_layers]
        else:
            self.candidate_premature_layers = [self.num_layers]

        self.mature_layer = self.candidate_premature_layers[-1]

        self.decoding_strategy = self.decoder_configs.configs.decoding_strategy
        self.decoding_mode = self.decoder_configs.configs.decoding_mode
        self.info_layer = self.decoder_configs.configs.info_layer
        self.relative_top = self.decoder_configs.configs.relative_top

    def _calculate_entropy(self, logits):
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)

        return entropy

    def relative_top_filter(
        self,
        scores: torch.FloatTensor,
        relative_top: float = 0.1,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ) -> torch.FloatTensor:
        scores_normalized = scores.log_softmax(dim=-1)
        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
        min_thresh = sorted_logits[..., min_tokens_to_keep - 1]
        probs_max = torch.max(scores_normalized, dim=-1).values
        probs_thresh = probs_max + np.log(relative_top)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        scores_normalized[scores_normalized < probs_thresh] = filter_value
        return scores_normalized

    def get_relative_top_filter(
        self,
        scores: torch.FloatTensor,
        relative_top: float = 0.1,
        min_tokens_to_keep: int = 1,
    ):
        scores_normalized = scores.log_softmax(dim=-1)
        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
        min_thresh = sorted_logits[..., min_tokens_to_keep - 1]
        probs_max = torch.max(scores_normalized, dim=-1).values
        probs_thresh = probs_max + np.log(relative_top)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        return scores_normalized < probs_thresh

    def generate(
        self,
        inputs,
        return_attentions: bool = False,
    ) -> dict:
        self.model.eval()

        prompt = inputs["prompted_question"][0]

        if len(inputs["verbalised_instruction"][0]):
            use_system_prompt = True
        else:
            use_system_prompt = False

        tokenised_inputs = self._verbalise_input(
            prompt, use_system_prompt=use_system_prompt
        ).to(self.model.device)

        # Predict
        with torch.inference_mode():
            before = ()
            generated_ids = []
            entropies = []

            input_ids = tokenised_inputs.clone()
            for _ in range(self.max_new_tokens):
                dict_outputs, outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=self.candidate_premature_layers
                    + [self.info_layer, self.mature_layer],
                )

                final_logits = dict_outputs[self.mature_layer][:, -1, :]
                if self.relative_top > 0.0:
                    final_logits = self.relative_top_filter(
                        final_logits, self.relative_top
                    )
                    mask = final_logits[0] < -1e3
                    index_nontop = torch.argwhere(mask).squeeze()

                logits = final_logits
                if len(before) == 0:  # the token is the first generated token
                    info_layer_score = dict_outputs[self.info_layer][
                        -1, :, :
                    ]  # [num_token_in_question, len_token_lib] -> e.g. [62, 32000]
                    before = (info_layer_score,)

                    # compute entropy of the info layer
                    info_layer_probs = F.softmax(
                        torch.t(info_layer_score), dim=1
                    ).unsqueeze(
                        0
                    )  # info_layer_score: [num_token_in_question, len_token_lib] -> e.g. [1, 250, 32000]
                    entropy = torch.distributions.Categorical(
                        probs=info_layer_probs, validate_args=False
                    ).entropy()  # [1,32000]
                elif len(before) >= 1:
                    info_layer_score = before[
                        0
                    ]  # [num_token_in_question, len_token_lib] -> e.g. [62, 32000]

                # we compute the adjust_score to calibrate the original score
                adjust_score = None

                if (
                    self.decoding_strategy == "entropy"
                    or self.decoding_mode == "activation_dola"
                ):
                    final_entropy = entropy.scatter(
                        1, index_nontop.unsqueeze(0), float("Inf")
                    )
                    if self.alpha != 0:
                        logits = logits + self.alpha * (-final_entropy)
                    else:
                        logits = logits

                    adjust_score = -final_entropy
                next_token_logits = logits.log_softmax(dim=-1)

                entropies += [adjust_score[0][0].item()]
                last_input_token = next_token_logits.argmax(dim=-1)
                generated_ids.append(last_input_token.item())
                input_ids = torch.cat([input_ids, last_input_token.unsqueeze(0)], dim=1)
                if last_input_token.item() == self.tokenizer.eos_token_id:
                    break
            decoded_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

        generation_output = {
            "decoded_text": decoded_text,
            "alphas": entropies,
            "attentions": {},
        }

        return generation_output

    def lm_score(
        self,
        prompt,
        answer,
    ):
        # Minimally adjusted from https://github.com/hkust-nlp/Activation_Decoding
        prompted_question = prompt["prompted_question"][0]

        if len(prompt["verbalised_instruction"][0]):
            use_system_prompt = True
        else:
            use_system_prompt = False

        with torch.no_grad():
            if type(prompted_question) == list:
                input_text = prompted_question + [answer]
            else:
                input_text = prompted_question + answer
            input_ids = self._verbalise_input(
                input_text,
                use_system_prompt=use_system_prompt,
                add_generation_prompt=False,
            ).to(self.model.device)
            prefix_ids = self._verbalise_input(
                prompted_question, use_system_prompt=use_system_prompt
            ).to(self.model.device)
            continue_ids = input_ids[0, prefix_ids.shape[-1] :]

            if self.decoding_mode == "activation":
                dict_outputs, _ = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=self.candidate_premature_layers
                    + [self.info_layer, self.mature_layer],
                )
                final_logits = dict_outputs[self.mature_layer][
                    0, prefix_ids.shape[-1] - 1 : -1
                ]
                # final_logits= self.model(input_ids)[0].squeeze(0)
                final_logits = final_logits.log_softmax(dim=-1)
                mask = final_logits[0] < -1e3

                if self.decoding_strategy == "entropy":
                    info_layer_score = dict_outputs[self.info_layer][-1, :, :]
                    index_nontop = torch.argwhere(mask).squeeze()
                    info_layer_probs = F.softmax(
                        torch.t(info_layer_score), dim=1
                    ).unsqueeze(
                        0
                    )  # info_layer_score: [num_token_in_question, len_token_lib] -> e.g. [250, 32000]
                    entropy = torch.distributions.Categorical(
                        probs=info_layer_probs, validate_args=False
                    ).entropy()

                    entropy = entropy.scatter(
                        1, index_nontop.unsqueeze(0), float("Inf")
                    )

                    if self.alpha != 0:
                        # entropy: the smaller the better
                        final_logits = final_logits + self.alpha * (-entropy)

                if self.decoding_strategy == "single_entropy":
                    info_layer_score = dict_outputs[self.info_layer][-1, :, :]

                    index_nontop = torch.argwhere(mask).squeeze()
                    info_layer_probs = F.softmax(
                        torch.t(info_layer_score), dim=1
                    ).unsqueeze(
                        0
                    )  # info_layer_score: [num_token_in_question, len_token_lib] -> e.g. [250, 32000]
                    entropy = torch.distributions.Categorical(
                        probs=info_layer_probs, validate_args=False
                    ).entropy()
                    # entropy = entropy.scatter(1, index_nontop.unsqueeze(0), float("Inf"))
                    final_logits = entropy
                log_probs = (
                    final_logits[range(final_logits.shape[0]), continue_ids]
                    .sum()
                    .item()
                )
            elif self.decoding_mode == "activation_dola":
                premature_layer_dist = {l: 0 for l in self.candidate_premature_layers}
                picked_logits = []
                result_dict = {}
                premature_layers = []

                dict_outputs, outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=self.candidate_premature_layers
                    + [self.mature_layer],
                )

                for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):
                    # Pick the less like layer to contrast with
                    # 1. Stacking all premature_layers into a new dimension
                    stacked_premature_layers = torch.stack(
                        [
                            dict_outputs[i][:, seq_i, :]
                            for i in self.candidate_premature_layers
                        ],
                        dim=0,
                    )

                    # 2. Calculate the softmax values for mature_layer and all premature_layers
                    softmax_mature_layer = F.softmax(
                        dict_outputs[self.mature_layer][:, seq_i, :], dim=-1
                    )  # shape: (batch_size, num_features)
                    softmax_premature_layers = F.softmax(
                        stacked_premature_layers, dim=-1
                    )  # shape: (num_premature_layers, batch_size, num_features)

                    # 3. Calculate M, the average distribution
                    M = 0.5 * (
                        softmax_mature_layer[None, :, :] + softmax_premature_layers
                    )  # shape: (num_premature_layers, batch_size, num_features)

                    # 4. Calculate log-softmax for the KL divergence
                    log_softmax_mature_layer = F.log_softmax(
                        dict_outputs[self.mature_layer][:, seq_i, :], dim=-1
                    )  # shape: (batch_size, num_features)
                    log_softmax_premature_layers = F.log_softmax(
                        stacked_premature_layers, dim=-1
                    )  # shape: (num_premature_layers, batch_size, num_features)

                    # 5. Calculate the KL divergences and then the JS divergences
                    kl1 = F.kl_div(
                        log_softmax_mature_layer[None, :, :], M, reduction="none"
                    ).mean(
                        -1
                    )  # shape: (num_premature_layers, batch_size)
                    kl2 = F.kl_div(
                        log_softmax_premature_layers, M, reduction="none"
                    ).mean(
                        -1
                    )  # shape: (num_premature_layers, batch_size)
                    js_divs = 0.5 * (
                        kl1 + kl2
                    )  # shape: (num_premature_layers, batch_size)

                    # 6. Reduce the batchmean
                    js_divs = js_divs.mean(-1)  # shape: (num_premature_layers,)
                    premature_layer = self.candidate_premature_layers[
                        int(js_divs.argmax().cpu().item())
                    ]
                    premature_layer_dist[premature_layer] += 1

                    premature_layers.append(premature_layer)

                base_logits = torch.zeros_like(
                    dict_outputs[self.mature_layer][0, prefix_ids.shape[-1] - 1 : -1]
                )
                for i, l in enumerate(premature_layers):
                    base_logits[i] = dict_outputs[l][0, prefix_ids.shape[-1] - 1 + i]
                final_logits = dict_outputs[self.mature_layer][
                    0, prefix_ids.shape[-1] - 1 : -1
                ]
                final_logits = final_logits.log_softmax(dim=-1)
                base_logits = base_logits.log_softmax(dim=-1)
                diff_logits = final_logits - base_logits

                if self.post_softmax:
                    diff_logits = diff_logits.log_softmax(dim=-1)

                info_layer_score = dict_outputs[self.info_layer][-1, :, :]
                mask = final_logits[0] < -1e3
                index_nontop = torch.argwhere(mask).squeeze()
                info_layer_probs = F.softmax(
                    torch.t(info_layer_score), dim=1
                ).unsqueeze(
                    0
                )  # info_layer_score: [num_token_in_question, len_token_lib] -> e.g. [250, 32000]
                entropy = torch.distributions.Categorical(
                    probs=info_layer_probs, validate_args=False
                ).entropy()

                entropy = entropy.scatter(1, index_nontop.unsqueeze(0), float("Inf"))

                if self.alpha != 0:
                    diff_logits = diff_logits + self.alpha * (-entropy)
                log_probs = (
                    diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()
                )

        return log_probs
