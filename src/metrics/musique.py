from typing import Dict, List

import numpy as np
import regex as re


class MuSiQue:
    def __init__(self):
        pass

    def compute_metrics(self, prediction: str, refs: List[str]):
        scores = {}
        scores["Subspan_EM"] = self.unnormalised_best_subspan_em(prediction, refs)

        return scores

    @staticmethod
    def answer_extractor(cot_answer: str) -> str:
        # Adapted from https://github.com/StonyBrookNLP/ircot/blob/main/evaluate.py

        cot_regex = re.compile(".* answer is:? (.*)\\.?")
        match = cot_regex.match(cot_answer)
        if match:
            output = match.group(1)
            if output.endswith("."):
                output = output[:-1]
        else:
            output = cot_answer

        return output

    @staticmethod
    def unnormalised_best_subspan_em(
        prediction: str, ground_truths: List[str]
    ) -> float:
        for ground_truth in ground_truths:
            if ground_truth.lower() in prediction.lower():
                return 1.0
        return 0.0

    def __call__(self, predictions) -> Dict[str, float]:
        subspan_em_scores = []
        for sample in predictions:
            refs = [
                ans[0] if type(ans) in [list, tuple] else ans
                for ans in sample["answers"]
            ]

            # Extract answer from the CoT reasonings
            prediction = self.answer_extractor(sample["predicted_answer"])

            scores = self.compute_metrics(prediction, refs)

            subspan_em_scores += [scores["Subspan_EM"]]

        metrics = {
            "Subspan_EM_scores": subspan_em_scores,
            "Subspan_EM": np.mean(subspan_em_scores),
        }
        return metrics
