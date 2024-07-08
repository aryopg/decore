from typing import Dict, Optional, Union

import torch
from abc import ABC, abstractmethod
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from src.configs import DecoderConfigs, ModelConfigs


class BaseModel(ABC):
    def __init__(
        self,
        model_configs: ModelConfigs,
        decoder_configs: DecoderConfigs,
    ):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_configs.configs.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
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
        inputs: str,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, None] = None,
    ) -> torch.Tensor:
        if tokenizer is None:
            tokenizer = self.tokenizer

        if self.model_configs.model_type == "instruct":
            # TODO: Consider adding system message, but now follow lm eval harness setup
            chat_inputs = [{"role": "user", "content": inputs}]

            inputs = tokenizer.apply_chat_template(
                chat_inputs,
                add_generation_prompt=True,
                return_tensors="pt",
                max_length=self.max_seq_len,
            )
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

    @abstractmethod
    def generate(self, logits):
        pass

    @abstractmethod
    def lm_score(self, logits):
        pass
