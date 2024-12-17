from typing import Dict, List

import numpy as np
import regex as re

from src.metrics.utils import best_em, best_subspan_em


class NQSwap:
    def __init__(self):
        pass

    @staticmethod
    def compute_metrics(prediction: str, sub_ref: List[str], org_ref: List[str]):
        scores = {}
        scores["sub_EM"] = best_em(prediction, sub_ref)
        scores["sub_Subspan_EM"] = best_subspan_em(prediction, sub_ref)

        scores["org_EM"] = best_em(prediction, org_ref)
        scores["org_Subspan_EM"] = best_subspan_em(prediction, org_ref)

        return scores

    def __call__(self, predictions) -> Dict[str, float]:
        sub_em_scores = []
        sub_subspan_em_scores = []
        org_em_scores = []
        org_subspan_em_scores = []
        for sample in predictions:
            org_ref = [
                ans[0] if type(ans) in [list, tuple] else ans
                for ans in sample["org_answer"]
            ]
            sub_ref = [
                ans[0] if type(ans) in [list, tuple] else ans
                for ans in sample["sub_answer"]
            ]

            # Only consider until \n, ., or ,
            prediction = re.split("\n", sample["predicted_answer"])[0]

            scores = self.compute_metrics(prediction, sub_ref, org_ref)

            sub_em_scores += [scores["sub_EM"]]
            sub_subspan_em_scores += [scores["sub_Subspan_EM"]]

            org_em_scores += [scores["org_EM"]]
            org_subspan_em_scores += [scores["org_Subspan_EM"]]

        metrics = {
            "sub_EM": np.mean(sub_em_scores),
            "sub_Subspan_EM": np.mean(sub_subspan_em_scores),
            "org_EM": np.mean(org_em_scores),
            "org_Subspan_EM": np.mean(org_subspan_em_scores),
        }
        return metrics
