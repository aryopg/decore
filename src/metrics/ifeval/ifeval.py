from typing import Dict, List

import regex as re
import numpy as np

from src.metrics.ifeval.utils import process_results


class IFEval:
    def __init__(self):
        pass

    def __call__(self, predictions) -> Dict[str, float]:
        prompt_level_strict_accs = []
        inst_level_strict_accs = []
        prompt_level_loose_accs = []
        inst_level_loose_accs = []
        for prediction in predictions:
            doc = {
                "key": prediction["key"],
                "instruction_id_list": prediction["instruction_id_list"],
                "prompt": prediction["prompt"],
                "kwargs": prediction["kwargs"],
            }
            metric = process_results(doc, prediction["predicted_answer"])

            prompt_level_strict_accs.append(metric["prompt_level_strict_accs"])
            inst_level_strict_accs.append(metric["inst_level_strict_accs"])
            prompt_level_loose_accs.append(metric["prompt_level_loose_accs"])
            inst_level_loose_accs.append(metric["inst_level_loose_accs"])

        metrics = {
            "prompt_level_strict_acc": np.mean(prompt_level_strict_accs),
            "inst_level_strict_acc": np.mean(inst_level_strict_accs),
            "prompt_level_loose_acc": np.mean(prompt_level_loose_accs),
            "inst_level_loose_acc": np.mean(inst_level_loose_accs),
        }
        return metrics
