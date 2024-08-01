import json
import os
from typing import List

from src.configs import DataConfigs
from src.datasets.base_dataset import BaseDataset


class XSum(BaseDataset):
    def __init__(
        self,
        data_configs: DataConfigs,
        **kwargs,
    ):
        super().__init__(data_configs, **kwargs)
        self.variation = data_configs.variation

        self.data_filename = os.path.join(self.data_dir, "xsum-1000.jsonl")

        # Prepare data
        self.data = self.parse_data()

    def parse_data(self) -> List[dict]:
        # Open the gz file, and read the jsonl file
        data = []

        with open(self.data_filename, "r") as f:
            for i, line in enumerate(f):
                instance = json.loads(line)
                data += [
                    {
                        "idx": instance["id"],
                        "document": instance["document"],
                        "summary": instance["summary"],
                    }
                ]

        if self.num_samples > 0:
            data = data[: self.num_samples]

        return data

    def build_prompt(self, context):
        instruction = [
            "Generate a summary comprising of 1 sentence for the given article."
        ]

        prompted_contexts = f"Article: {context}+\n"

        answer_prefix = "Summary: "
        if self.kwargs["use_chat_template"]:
            input_text_prompt = [instruction + [f"{prompted_contexts}{answer_prefix}"]]
        else:
            instruction = instruction[0] + "\n\n"
            input_text_prompt = instruction + (f"{prompted_contexts}{answer_prefix}")
        return {
            "verbalised_instruction": instruction,
            "verbalised_icl_demo": "",
            "verbalised_contexts": prompted_contexts,
            "verbalised_question": "",
            "verbalised_answer_prefix": answer_prefix,
            "prompted_question": input_text_prompt,
        }

    def __getitem__(self, idx):
        sample = self.data[idx]

        prompt = self.build_prompt(sample["document"])

        # For attention analysis
        sample["verbalised_instruction"] = prompt["verbalised_instruction"]
        sample["verbalised_icl_demo"] = prompt["verbalised_icl_demo"]
        sample["verbalised_contexts"] = prompt["verbalised_contexts"]
        sample["verbalised_question"] = prompt["verbalised_question"]
        sample["verbalised_answer_prefix"] = prompt["verbalised_answer_prefix"]

        sample["prompted_question"] = prompt["prompted_question"]

        return sample

    def __len__(self):
        return len(self.data)