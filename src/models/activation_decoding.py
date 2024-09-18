from typing import List, Optional, Tuple, Union

import torch
from torch.nn import functional as F

from src.configs import DecoderConfigs, ModelConfigs

# Transformers submodule taken from https://github.com/hkust-nlp/Activation_Decoding
from transformers import AutoTokenizer
from transformers_ad.src.transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


class ActivationDecoding:
    def __init__(
        self,
        model_configs: ModelConfigs,
        decoder_configs: DecoderConfigs,
    ):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_configs.configs.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_configs.configs.model_name_or_path
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.max_seq_len = model_configs.configs.max_seq_len
        self.max_new_tokens = model_configs.configs.max_new_tokens

        self.model_configs = model_configs
        self.decoder_configs = decoder_configs

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

    def _verbalise_input(
        self,
        inputs: Union[list, str],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, None] = None,
        use_system_prompt: bool = True,
        add_generation_prompt: bool = True,
        use_chat_template: bool = True,
    ) -> torch.Tensor:
        if tokenizer is None:
            tokenizer = self.tokenizer

        if self.model_configs.model_type == "instruct":
            if use_chat_template:
                chat_inputs = []
                if type(inputs) == list:
                    if "mistral" in self.model_configs.name.lower():
                        if use_system_prompt:
                            system_prompt = inputs[0]
                            if type(system_prompt) in [tuple, list]:
                                system_prompt = system_prompt[0]
                            system_prompt = system_prompt + "\n"
                            inputs = inputs[1:]
                        else:
                            system_prompt = ""
                    for idx, input in enumerate(inputs):
                        if type(input) in [tuple, list]:
                            input = input[0]
                        # Mistral can't handle system prompt
                        if (
                            use_system_prompt
                            and "mistral" not in self.model_configs.name.lower()
                        ):
                            if idx == 0:
                                chat_inputs += [{"role": "system", "content": input}]
                            else:
                                if idx % 2 != 0:
                                    chat_inputs += [{"role": "user", "content": input}]
                                else:
                                    chat_inputs += [
                                        {"role": "assistant", "content": input}
                                    ]
                        else:
                            # Mistral can't handle system prompt
                            if "mistral" in self.model_configs.name.lower():
                                if idx % 2 == 0:
                                    chat_inputs += [
                                        {
                                            "role": "user",
                                            "content": system_prompt + input,
                                        }
                                    ]
                                else:
                                    chat_inputs += [
                                        {"role": "assistant", "content": input}
                                    ]
                            else:
                                if idx % 2 == 0:
                                    chat_inputs += [{"role": "user", "content": input}]
                                else:
                                    chat_inputs += [
                                        {"role": "assistant", "content": input}
                                    ]
                else:
                    if type(inputs) in [tuple, list]:
                        inputs = inputs[0]
                    chat_inputs += [{"role": "user", "content": inputs}]
                inputs = tokenizer.apply_chat_template(
                    chat_inputs,
                    add_generation_prompt=add_generation_prompt,
                    return_tensors="pt",
                    max_length=self.max_seq_len,
                )
            else:
                if type(inputs) in [tuple, list]:
                    inputs = inputs[0]
                inputs = tokenizer(
                    inputs, return_tensors="pt", max_length=self.max_seq_len
                ).input_ids

        elif self.model_configs.model_type == "base":
            inputs = tokenizer(
                inputs,
                return_tensors="pt",
                max_length=self.max_seq_len,
            ).input_ids
        else:
            raise ValueError(
                f"Unknown model type: {self.model_configs.model_type}. "
                "Terminate tokenisation process."
            )

        return inputs

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
        tokenised_inputs = self._verbalise_input(prompt).to(self.model.device)

        with torch.inference_mode():
            if self.decoding_mode == "activation":
                outputs = self.model.generate(
                    tokenised_inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_return_sequences=1,
                    output_hidden_states=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                    activation_decoding=True,
                    do_sample=False,
                    mature_layer=self.mature_layer,
                    premature_layer=None,
                    candidate_premature_layers=self.candidate_premature_layers,
                    return_adjust_scores=False,
                )
            elif self.decoding_mode == "activation_dola":
                outputs = self.model.generate(
                    tokenised_inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_return_sequences=1,
                    output_hidden_states=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                    activation_dola_decoding=True,
                    do_sample=False,
                    mature_layer=self.mature_layer,
                    premature_layer=None,
                    candidate_premature_layers=self.candidate_premature_layers,
                    return_adjust_scores=False,
                )
            decoded_text = self.tokenizer.decode(
                outputs.sequences[0, tokenised_inputs.size(1) :],
                skip_special_tokens=True,
            )
        logits = torch.stack(outputs.logits, dim=1)

        entropies = self._calculate_entropy(logits)
        entropies = entropies.cpu().numpy().tolist()

        return {"decoded_text": decoded_text, "alphas": entropies, "attentions": {}}

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
                    + [self.mature_layer],
                    info_layer=self.info_layer,
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
