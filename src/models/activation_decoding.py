from typing import List, Optional, Tuple, Union

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

        self.mature_layer = self.candidate_premature_layers[-1]

        self.decoding_strategy = self.decoder_configs.configs.decoding_strategy
        self.decoding_mode = self.decoder_configs.configs.decoding_mode
        self.info_layer = self.decoder_configs.configs.info_layer

    def _calculate_entropy(self, logits):
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)

        return entropy

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
            input_logits = self.model(
                input_ids=tokenised_inputs[:, :-1], use_cache=True, return_dict=True
            )
            generated_ids = []
            entropies = []
            last_input_token = tokenised_inputs[:, -1]
            past_kv = input_logits.past_key_values
            for _ in range(self.max_new_tokens):
                last_input_token = last_input_token.view(1, 1)
                dict_outputs, outputs = self.model(
                    input_ids=tokenised_inputs,
                    past_key_values=past_kv,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=self.candidate_premature_layers
                    + [self.info_layer, self.mature_layer],
                )

                final_logits = dict_outputs[self.mature_layer][:, -1, :]
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
                    if self.alpha != 0:
                        logits = logits + self.alpha * (-entropy)
                    else:
                        logits = logits

                    adjust_score = -entropy

                entropies += [adjust_score]
                past_kv = outputs.past_key_values
                last_input_token = logits[0, -1].argmax()
                generated_ids.append(last_input_token.item())
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
