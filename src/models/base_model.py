from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from src.configs import DecoderConfigs, ModelConfigs
from src.utils.modelling_llama import LlamaConfig, LlamaForCausalLM


class BaseModel(ABC):
    def __init__(
        self,
        model_configs: ModelConfigs,
        decoder_configs: DecoderConfigs,
    ):
        self.model = LlamaForCausalLM.from_pretrained(
            model_configs.configs.model_name_or_path,
            use_flash_attention_2="flash_attention_2",
            attn_implementation="flash_attention_2",
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
                    for idx, input in enumerate(inputs):
                        if type(input) in [tuple, list]:
                            input = input[0]
                        if idx == 0 and use_system_prompt:
                            chat_inputs += [{"role": "system", "content": input}]
                        else:
                            if idx % 2 != 0:
                                chat_inputs += [{"role": "user", "content": input}]
                            else:
                                chat_inputs += [{"role": "assistant", "content": input}]
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

    def _get_component_lengths(self, inputs, tokenised_inputs):
        if self.model_configs.model_type == "instruct":
            bos_length = 1
            # Skip BOS
            instruction_tokens = self._verbalise_input(
                inputs["verbalised_instruction"][0]
            )[:, 1:]
            instruction_length = instruction_tokens.shape[-1]
            # 5 is <|begin_of_text|><|start_header_id|>user<|end_header_id|> in llama3-8b-instruct tokenizer
            icl_demo_tokens = self._verbalise_input(
                inputs["verbalised_icl_demo"], use_system_prompt=False
            )[:, 5:]
            icl_demo_length = icl_demo_tokens.shape[-1]
            contexts_tokens = self._verbalise_input(
                inputs["verbalised_contexts"][0], use_chat_template=False
            )[:, 1:]
            contexts_length = contexts_tokens.shape[-1]
            question_tokens = self._verbalise_input(
                inputs["verbalised_question"][0], use_chat_template=False
            )[:, 1:]
            question_length = question_tokens.shape[-1]
            answer_prefix_tokens = self._verbalise_input(
                inputs["verbalised_answer_prefix"][0]
            )[:, 5:]
            answer_prefix_length = answer_prefix_tokens.shape[-1]
        else:
            bos_length = 1
            # Start from 1 to skip the BOS token
            instruction_length = self._verbalise_input(
                inputs["verbalised_instruction"]
            )[:, 1:].shape[-1]
            icl_demo_length = self._verbalise_input(inputs["verbalised_icl_demo"])[
                :, 1:
            ].shape[-1]
            contexts_length = self._verbalise_input(inputs["verbalised_contexts"])[
                :, 1:
            ].shape[-1]
            question_length = self._verbalise_input(inputs["verbalised_question"])[
                :, 1:
            ].shape[-1]
            answer_prefix_length = self._verbalise_input(
                inputs["verbalised_answer_prefix"]
            )[1:].shape[-1]

        sum_lengths = (
            bos_length
            + instruction_length
            + icl_demo_length
            + contexts_length
            + question_length
            + answer_prefix_length
        )
        try:
            assert sum_lengths == tokenised_inputs.size(1)
        except AssertionError:
            print(
                f"Tokenised inputs length does not match the sum of the lengths of the components"
            )
            print("instruction: ", instruction_tokens.cpu().numpy()[0].tolist())
            print("icl_demo: ", icl_demo_tokens.cpu().numpy()[0].tolist())
            print("contexts: ", contexts_tokens.cpu().numpy()[0].tolist())
            print("question: ", question_tokens.cpu().numpy()[0].tolist())
            print("answer_prefix: ", answer_prefix_tokens.cpu().numpy()[0].tolist())
            print("tokenised_inputs: ", tokenised_inputs.cpu().numpy()[0].tolist())
            print(f"bos_length: {bos_length}")
            print(f"instruction_length: {instruction_length}")
            print(f"icl_demo_length: {icl_demo_length}")
            print(f"contexts_length: {contexts_length}")
            print(f"question_length: {question_length}")
            print(f"answer_prefix_length: {answer_prefix_length}")
            print(f"Sum:{sum_lengths}")
            print(f"Tokenised inputs:{tokenised_inputs.size(1)}")
            raise AssertionError

        return {
            "bos": bos_length,
            "instruction": instruction_length,
            "icl_demo": icl_demo_length,
            "contexts": contexts_length,
            "question": question_length,
            "answer_prefix": answer_prefix_length,
        }

    def get_lookback_ratios(self, attentions, component_lengths, new_token_start_from):
        components = list(component_lengths.keys())
        # Define component order and initialize lookback ratio tensors
        num_layers = len(attentions[0])
        num_heads = attentions[0][0].shape[1]
        new_token_length = len(attentions)

        # Initialize lookback ratio tensors
        lookback_ratios = {
            comp: torch.zeros((num_layers, num_heads, new_token_length))
            for comp in components
        }
        lookback_ratios["new_tokens"] = torch.zeros(
            (num_layers, num_heads, new_token_length)
        )

        for i in range(new_token_length):
            for l in range(num_layers):
                curr_length = 0
                attn_sums = []

                # Calculate attention for each component
                for comp, length in component_lengths.items():
                    attn = attentions[i][l][
                        0, :, -1, curr_length : curr_length + length + 1
                    ].mean(-1)
                    lookback_ratios[comp][l, :, i] = attn
                    attn_sums.append(attn)
                    curr_length += length

                # Validate new token start
                assert (
                    new_token_start_from == curr_length
                ), "Mismatch in the length of the components"

                # Calculate attention for new tokens
                attn_new_tokens = attentions[i][l][
                    0, :, -1, new_token_start_from:
                ].mean(-1)
                lookback_ratios["new_tokens"][l, :, i] = attn_new_tokens
                attn_sums.append(attn_new_tokens)

                # Normalize ratios
                attn_sum = sum(attn_sums)
                attn_sum = attn_sum.cpu()
                for comp in lookback_ratios:
                    lookback_ratios[comp][l, :, i] /= attn_sum

        return lookback_ratios

    def _generate(
        self,
        inputs,
        return_attentions: bool = False,
        block_list: Optional[list] = None,
    ) -> dict:
        self.model.eval()

        prompt = inputs["prompted_question"][0]
        tokenised_inputs = self._verbalise_input(prompt).to(self.model.device)

        # Calculate the length of each component
        component_lengths = self._get_component_lengths(inputs, tokenised_inputs)

        # Predict
        with torch.inference_mode():
            input_logits = self.model(
                input_ids=tokenised_inputs[:, :-1], use_cache=True, return_dict=True
            )
            generated_ids = []
            attentions = []
            last_input_token = tokenised_inputs[:, -1]
            past_kv = input_logits.past_key_values
            for _ in range(self.max_new_tokens):
                last_input_token = last_input_token.view(1, 1)
                outputs = self.model(
                    input_ids=last_input_token,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_attentions=True,
                    attn_mode="torch",
                    block_list=block_list,
                )
                attentions += [outputs.attentions]
                past_kv = outputs.past_key_values
                last_input_token = outputs.logits[0, -1].argmax()
                generated_ids.append(last_input_token.item())
                if last_input_token.item() == self.tokenizer.eos_token_id:
                    break
            decoded_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

        generation_output = {"decoded_text": decoded_text, "attentions": {}}
        if return_attentions:
            generation_output["attentions"] = self.get_lookback_ratios(
                attentions, component_lengths, tokenised_inputs.size(1)
            )

        return generation_output

    @abstractmethod
    def generate(self, logits):
        pass

    @abstractmethod
    def lm_score(self, logits):
        pass
