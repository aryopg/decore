from typing import Dict, List

import regex as re
import numpy as np

from src.metrics.ifeval.utils import process_results, agg_inst_level_acc


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
                "key": prediction["idx"],
                "instruction_id_list": [
                    (
                        instruction_id[0]
                        if type(instruction_id) in [list, tuple]
                        else instruction_id
                    )
                    for instruction_id in prediction["instruction_id_list"]
                ],
                "prompt": prediction["prompt"],
                "kwargs": [
                    {
                        k: v[0] if type(v) in [list, tuple] else v
                        for k, v in kwargs_.items()
                    }
                    for kwargs_ in prediction["kwargs"]
                ],
            }
            metric = process_results(doc, prediction["predicted_answer"])

            prompt_level_strict_accs.append(metric["prompt_level_strict_acc"])
            inst_level_strict_accs.append(metric["inst_level_strict_acc"])
            prompt_level_loose_accs.append(metric["prompt_level_loose_acc"])
            inst_level_loose_accs.append(metric["inst_level_loose_acc"])

        metrics = {
            "prompt_level_strict_acc": np.mean(prompt_level_strict_accs),
            "inst_level_strict_acc": agg_inst_level_acc(inst_level_strict_accs),
            "prompt_level_loose_acc": np.mean(prompt_level_loose_accs),
            "inst_level_loose_acc": agg_inst_level_acc(inst_level_loose_accs),
        }
        return metrics


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_filepath", type=str)
    args = parser.parse_args()

    print(f"Reading predictions: {args.predictions_filepath}")

    with open(args.predictions_filepath, "r") as f:
        predictions = [json.loads(line) for line in f]

    metric = IFEval()
    metrics = metric(predictions)
    print(metrics)
