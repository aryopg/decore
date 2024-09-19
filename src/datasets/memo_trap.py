import gzip
import json
import math
import os
import ast
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedTokenizer
from datasets import load_dataset, load_from_disk

from src.configs import DataConfigs, DecoderConfigs
from src.datasets.base_dataset import BaseDataset


class MemoTrap(BaseDataset):
    def __init__(
        self,
        data_configs: DataConfigs,
        **kwargs,
    ):
        super().__init__(data_configs, **kwargs)

        self.data_filenames = {
            "proverb_ending": os.path.join(self.data_dir, "1-proverb-ending.csv"),
            "proverb_translation": os.path.join(
                self.data_dir, "2-proverb-translation.csv"
            ),
            "hate_speech_ending": os.path.join(
                self.data_dir, "3-hate-speech-ending.csv"
            ),
            "history_of_science_qa": os.path.join(
                self.data_dir, "4-history-of-science-qa.csv"
            ),
        }

        # Prepare data
        self.data = self.parse_data()

    def parse_data(self) -> List[dict]:
        # Open the gz file, and read the jsonl file
        data = []
        for split_name, data_filename in self.data_filenames.items():
            df = pd.read_csv(data_filename)

            for idx, instance in df.iterrows():
                data += [
                    {
                        "idx": f"{split_name}__{idx}",
                        "split": split_name,  # For the purpose of evaluation
                        "question": instance["prompt"],
                        "classes": ast.literal_eval(instance["classes"]),
                        "answer_index": instance["answer_index"],
                    }
                ]

        if self.num_samples > 0:
            data = data[: self.num_samples]

        return data

    def __getitem__(self, idx):
        sample = self.data[idx]

        true_idx = sample["answer_index"]
        false_idx = 0 if true_idx == 1 else 1

        # Attention analysis is challenging for MemoTrap as the dataset is very occluded
        sample["verbalised_instruction"] = ""
        sample["verbalised_icl_demo"] = ""
        sample["verbalised_contexts"] = ""
        sample["verbalised_question"] = sample["question"]
        sample["verbalised_answer_prefix"] = ""

        sample["prompted_question"] = (
            [[sample["question"]]]
            if self.kwargs["use_chat_template"]  # Use list for chat template
            else sample["question"]
        )
        sample["prompted_question_wo_context"] = ""
        sample["prompted_ref_true"] = sample["classes"][true_idx]
        sample["prompted_ref_false"] = sample["classes"][false_idx]

        return sample

    def __len__(self):
        return len(self.data)
