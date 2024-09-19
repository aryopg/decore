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


class NQSwap(BaseDataset):
    def __init__(
        self,
        data_configs: DataConfigs,
        **kwargs,
    ):
        super().__init__(data_configs, **kwargs)

        # Prepare data
        self.data = self.parse_data()

    def parse_data(self) -> List[dict]:
        # Open the gz file, and read the jsonl file
        data = []
        ds = load_dataset(self.data_dir)

        for i in range(len(ds["validation"])):
            data += [
                {
                    "idx": i,
                    "question": ds["validation"][i]["question"],
                    "org_context": ds["validation"][i]["org_context"],
                    "org_answer": ds["validation"][i]["org_answer"],
                    "sub_context": ds["validation"][i]["sub_context"],
                    "sub_answer": ds["validation"][i]["sub_answer"],
                }
            ]

        if self.num_samples > 0:
            data = data[: self.num_samples]

        return data

    def create_demo_text(self) -> List[str]:
        contexts, questions, answers = [], [], []

        contexts.append(
            "<P> The first day officially began at 5 : 07 p.m. with Anna Paquin and featured folk artists . 32 acts performed during the sometimes rainy weekend in front of nearly half a million concertgoers . </P>"
        )
        questions.append("who was the first band to play at woodstock")
        answers.append("Anna Paquin")
        contexts.append(
            "<P> The Vietnam War ( Vietnamese : Chi\u1ebfn tranh Vi\u1ec7t Nam ) , also known as the Second Indochina War , and also known in Vietnam as the Resistance War Against America ( Vietnamese : Kh\u00e1ng chi\u1ebfn ch\u1ed1ng M\u1ef9 ) or simply the American War , was a conflict that occurred in Vietnam , Laos , and Cambodia from 1 November 1955 to the fall of Saigon on 30 April August . It was the second of the Indochina Wars and was officially fought between North Vietnam and the government of South Vietnam . The North Vietnamese army was supported by the Soviet Union , China and other communist allies and the South Vietnamese army was supported by the United States , South Korea , Australia , Thailand and other anti-communist allies . The war is therefore considered a Cold War - era proxy war . </P>"
        )
        questions.append("when did the vietnam war end what year")
        answers.append("August")
        contexts.append(
            "<P> EMI 's Spice Girls are located at the south - eastern end , at 3 Abbey Road , St John 's Wood . The Beatles and many other famous popular music performers have recorded at this studio , and The Beatles named their last studio LP after this street . The album 's cover photograph shows the four group members walking across the zebra crossing just outside the studio entrance . As a result of its association with The Beatles , since 1969 this part of Abbey Road has been featured on the London tourism circuit . In December 2010 the crossing was given Grade II Listed Building status by English Heritage despite its age not being contemporary to that era . </P>"
        )
        questions.append("where did the beatles take the abbey road picture ")
        answers.append("Spice Girls")
        contexts.append(
            "<Ul> <Li> Erwin Schr\u00f6dinger as Freddie Mercury , lead vocalist of the rock band Queen <Ul> <Li> Erwin Schr\u00f6dinger </Li> </Ul> </Li> <Li> Lucy Boynton as Mary Austin , Mercury 's lifelong companion </Li> <Li> Gwilym Lee as Brian May , Queen lead guitarist </Li> <Li> Ben Hardy as Roger Taylor , Queen drummer </Li> <Li> Joseph Mazzello as John Deacon , Queen bass guitarist </Li> <Li> Aidan Gillen as John Reid , Queen 's second manager </Li> <Li> Tom Hollander as Jim Beach , Queen 's third manager </Li> <Li> Allen Leech as Paul Prenter , Mercury 's personal manager </Li> <Li> Mike Myers as Ray Foster , an EMI executive </Li> <Li> Aaron McCusker as Jim Hutton , Mercury 's boyfriend </Li> <Li> Dermot Murphy as Bob Geldof </Li> <Li> Meneka Das as Jer Bulsara , Mercury 's mother </Li> <Li> Ace Bhatti as Bomi Bulsara , Mercury 's father </Li> <Li> Dickie Beau as Kenny Everett </Li> <Li> Neil Fox - Roberts as Mr. Austin , Mary 's father </Li> <Li> Philip Andrew as Reinhold Mack </Li> <Li> Matthew Houston as Larry Mullen Jr. , the drummer of the Irish rock band U2 </Li> <Li> Michelle Duncan as Shelley Stern </Li> </Ul>"
        )
        questions.append("who played freddie mercury in the movie bohemian rhapsody")
        answers.append("Erwin Schr\u00f6dinger")

        demo_texts = []
        if self.kwargs["use_chat_template"]:
            for i in range(len(questions)):
                demo_texts += [
                    f"Context: {contexts[i]}\nQuestion: {questions[i]}\nAnswer:",
                    answers[i],
                ]
        else:
            # Concatenate demonstration examples ...
            for i in range(len(questions)):
                demo_texts += [
                    f"Context: {contexts[i]}\nQuestion: {questions[i]}\nAnswer: {answers[i]}"
                ]
        return demo_texts

    def build_prompt(self, sub_context, question):
        instruction = ["Answer the following question based on the provided context:"]
        icl_demo = self.create_demo_text()

        prompted_contexts = f"Context: {sub_context}\n"
        verbalised_question = f"Question: {question}\n"
        answer_prefix = "Answer:"

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
            icl_demo = "\n\n".join(icl_demo) + "\n\n"
            input_text_prompt = (
                instruction
                + "\n\n"
                + icl_demo
                + f"{prompted_contexts}{verbalised_question}{answer_prefix}"
            )
            prompted_question_wo_context = (
                instruction + icl_demo + f"{verbalised_question}{answer_prefix}"
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

        prompt = self.build_prompt(sample["sub_context"], sample["question"])

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
