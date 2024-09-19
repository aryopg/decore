import gzip
import json
import math
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedTokenizer

from datasets import load_dataset, load_from_disk
from src.configs import DataConfigs, DecoderConfigs
from src.datasets.base_dataset import BaseDataset


class IFEval(BaseDataset):
    def __init__(
        self,
        data_configs: DataConfigs,
        **kwargs,
    ):
        super().__init__(data_configs, **kwargs)

        # Prepare data
        self.data = self.parse_data()

    def parse_data(self) -> List[dict]:
        data = []
        ds = load_from_disk(self.data_dir)

        for i in range(len(ds)):
            sample = {
                "idx": ds[i]["key"],
                "prompt": ds[i]["prompt"],
                "instruction_id_list": ds[i]["instruction_id_list"],
                "kwargs": [
                    {k: v for k, v in kwargs_.items() if v}
                    for kwargs_ in ds[i]["kwargs"]
                ],
            }
            for key, value in sample.items():
                if value is None:
                    print(f"Sample {sample['idx']} contains null value for key {key}")
            data += [sample]

        if self.num_samples > 0:
            data = data[: self.num_samples]

        return data

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Attention analysis is challenging for IFEval as the dataset is very occluded
        sample["verbalised_instruction"] = ""
        sample["verbalised_icl_demo"] = ""
        sample["verbalised_contexts"] = ""
        sample["verbalised_question"] = sample["prompt"]
        sample["verbalised_answer_prefix"] = ""

        sample["prompted_question"] = sample["prompt"]

        return sample

    def __len__(self):
        return len(self.data)
