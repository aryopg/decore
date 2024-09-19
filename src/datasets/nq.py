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


class NQ(BaseDataset):
    available_variations = {
        "oracle": "nq-open-oracle.jsonl.gz",
        "closed_book": "nq-open-oracle.jsonl.gz",
        "gold_at_0": "nq-open-10_total_documents_gold_at_0.jsonl.gz",
        "gold_at_4": "nq-open-10_total_documents_gold_at_4.jsonl.gz",
        "gold_at_9": "nq-open-10_total_documents_gold_at_9.jsonl.gz",
    }

    def __init__(
        self,
        data_configs: DataConfigs,
        **kwargs,
    ):
        super().__init__(data_configs, **kwargs)
        self.variation = data_configs.variation

        self.data_filename = os.path.join(
            self.data_dir, self.available_variations[data_configs.variation]
        )

        # Prepare data
        self.data = self.parse_data()

    @staticmethod
    def concat(title, content):
        return f"(Title: {title.strip()}) {content.strip()}"

    def parse_data(self) -> List[dict]:
        # Open the gz file, and read the jsonl file
        data = []

        with gzip.open(self.data_filename, "rb") as f:
            for i, line in enumerate(f):
                instance = json.loads(line)
                if self.variation == "closed_book":
                    contexts = []
                else:
                    contexts = [
                        self.concat(context["title"], context["text"])
                        for context in instance["ctxs"]
                    ]

                data += [
                    {
                        "idx": i,
                        "question": instance["question"],
                        "answers": instance["answers"],
                        "contexts": contexts,
                    }
                ]

        if self.num_samples > 0:
            data = data[: self.num_samples]

        return data

    def create_closed_book_demo_text(self) -> List[str]:
        questions, answers = [], []

        questions.append("where did they film hot tub time machine")
        answers.append("Fernie Alpine Resort")

        questions.append("who has the right of way in international waters")
        answers.append("Neither vessel")

        questions.append("who does annie work for attack on titan")
        answers.append("Marley")

        questions.append("when was the immigration reform and control act passed")
        answers.append("November\u00a06, 1986")

        questions.append("when was puerto rico added to the usa")
        answers.append("1950")

        questions.append(
            "who has been chosen for best supporting actress in 64 national filmfare award"
        )
        answers.append("Zaira Wasim")

        questions.append("which side of the white house is the front")
        answers.append("North")

        questions.append("who's hosting the super bowl in 2019")
        answers.append("Atlanta, Georgia")

        # Verbalise the demos
        if self.kwargs["use_chat_template"]:
            demo_text = []
            for question, answer in zip(questions, answers):
                demo_text += [f"Question: {question}\nAnswer: ", answer]
        else:
            demo_text = []
            for question, answer in zip(questions, answers):
                demo_text += [f"Question: {question}\nAnswer: {answer}"]

        return demo_text

    def create_open_book_demo_text(self) -> List[str]:
        contexts, questions, answers = [], [], []

        contexts.append("(Title: Example Title) Example Text")
        questions.append("example question")
        answers.append("march 2018")

        # Verbalise the demos
        if self.kwargs["use_chat_template"]:
            demo_text = []
            for i, (context, question, answer) in enumerate(
                zip(contexts, questions, answers)
            ):
                demo_text += [
                    f"Document [{i}]{context}\nQuestion: {question}\nAnswer: ",
                    answer,
                ]
        else:
            demo_text = []
            for i, (context, question, answer) in enumerate(
                zip(contexts, questions, answers)
            ):
                demo_text += [
                    f"Document [{i}]{context}\nQuestion: {question}\nAnswer: {answer}"
                ]

        return demo_text

    def build_prompt(self, contexts, question):
        if self.variation == "closed_book":
            instruction = [
                "Write a high-quality answer for the given question. Provide the answer in 5 words or less without any explanation."
            ]
            icl_demo = self.create_closed_book_demo_text()
        else:
            instruction = [
                "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant). Provide the answer in 5 words or less without any explanation."
            ]
            icl_demo = self.create_open_book_demo_text()

        prompted_contexts = "\n".join(
            [f"Document [{i+1}]{context}" for i, context in enumerate(contexts)]
        )
        if prompted_contexts:
            prompted_contexts += "\n\n"

        verbalised_question = f"Question: {question}\n"
        answer_prefix = "Answer: "
        if self.kwargs["use_chat_template"]:
            input_text_prompt = [
                instruction
                + icl_demo
                + [f"{prompted_contexts}{verbalised_question}{answer_prefix}"]
            ]
            prompted_question_wo_context = [
                instruction + icl_demo + [f"{verbalised_question}{answer_prefix}"]
            ]
        else:
            instruction = instruction[0]
            icl_demo = "\n\n".join(icl_demo)
            input_text_prompt = (
                instruction
                + "\n\n"
                + icl_demo
                + "\n\n"
                + (f"{prompted_contexts}{verbalised_question}{answer_prefix}")
            )
            prompted_question_wo_context = (
                instruction
                + "\n\n"
                + icl_demo
                + "\n\n"
                + (f"{verbalised_question}{answer_prefix}")
            )
        return {
            "verbalised_instruction": instruction,
            "verbalised_icl_demo": icl_demo,
            "verbalised_contexts": prompted_contexts,
            "verbalised_question": verbalised_question,
            "verbalised_answer_prefix": answer_prefix,
            "prompted_question": input_text_prompt,
            "prompted_question_wo_context": prompted_question_wo_context,
        }

    def __getitem__(self, idx):
        sample = self.data[idx]

        prompt = self.build_prompt(sample["contexts"], sample["question"])

        # For attention analysis
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
