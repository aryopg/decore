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


class TriviaQA(BaseDataset):
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
            data += [
                {
                    "idx": ds[i]["question_id"],
                    "question": ds[i]["question"],
                    "answers": ds[i]["answer"]["aliases"],
                }
            ]

        if self.num_samples > 0:
            data = data[: self.num_samples]

        return data

    def create_demo_text(self) -> List[str]:
        # First 8 train QA pairs
        questions, answers = [], []

        questions.append(
            "Which American-born Sinclair won the Nobel Prize for Literature in 1930?"
        )
        answers.append("sinclair lewis")

        questions.append("Where in England was Dame Judi Dench born?")
        answers.append("york")

        questions.append(
            "In which decade did Billboard magazine first publish and American hit chart?"
        )
        answers.append("30s")

        questions.append("From which country did Angola achieve independence in 1975?")
        answers.append("portugal")

        questions.append("Which city does David Soul come from?")
        answers.append("chicago")

        questions.append("Who won Super Bowl XX?")
        answers.append("chicago bears")

        questions.append(
            "Which was the first European country to abolish capital punishment?"
        )
        answers.append("norway")

        questions.append(
            "In which country did he widespread use of ISDN begin in 1988?"
        )
        answers.append("japan")

        demo_texts = []
        if self.kwargs["use_chat_template"]:
            for i in range(len(questions)):
                demo_texts += [
                    f"Question: {questions[i]}\nAnswer:",
                    answers[i],
                ]
        else:
            for i in range(len(questions)):
                demo_texts += [f"Question: {questions[i]}\nAnswer: {answers[i]}"]
        return demo_texts

    def build_prompt(self, question):
        instruction = ["Answer the given question."]

        icl_demo = self.create_demo_text()

        verbalised_question = f"Question: {question}\n"
        answer_prefix = "Answer:"
        if self.kwargs["use_chat_template"]:
            input_text_prompt = [
                instruction + icl_demo + [f"{verbalised_question}{answer_prefix}"]
            ]
        else:
            instruction = instruction[0]
            icl_demo = "\n\n".join(icl_demo) + "\n\n"
            input_text_prompt = (
                instruction
                + "\n\n"
                + icl_demo
                + (f"{verbalised_question}{answer_prefix}")
            )
        return {
            "verbalised_instruction": instruction,
            "verbalised_icl_demo": icl_demo,
            "verbalised_contexts": "",
            "verbalised_question": verbalised_question,
            "verbalised_answer_prefix": answer_prefix,
            "prompted_question": input_text_prompt,
            "prompted_question_wo_context": "",
        }

    def __getitem__(self, idx):
        sample = self.data[idx]

        # For attention analysis
        prompt = self.build_prompt(sample["question"])

        sample["verbalised_instruction"] = prompt["verbalised_instruction"]
        sample["verbalised_icl_demo"] = prompt["verbalised_icl_demo"]
        sample["verbalised_contexts"] = prompt["verbalised_contexts"]
        sample["verbalised_question"] = prompt["verbalised_question"]
        sample["verbalised_answer_prefix"] = prompt["verbalised_answer_prefix"]

        sample["prompted_question"] = prompt["prompted_question"]
        sample["prompted_question_wo_context"] = prompt["prompted_question_wo_context"]

        return sample

    def __len__(self):
        return len(self.data)
