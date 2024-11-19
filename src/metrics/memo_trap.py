from typing import Dict, List, Union

import numpy as np


class MemoTrap:
    def __init__(self):
        pass

    def __call__(self, predictions: dict) -> Dict[str, Union[float, List[float]]]:
        """
        Compute macro and micro accuracy for MemoTrap.

        Args:
            predictions (dict): Predictions from the model.

        Returns:
            Dict[str, float]: Macro and micro accuracy.
        """
        scores = {
            "all": [],
            "proverb_ending": [],
            "proverb_translation": [],
            "hate_speech_ending": [],
            "history_of_science_qa": [],
        }
        for sample in predictions:
            scores_true = sample["scores_true"][0]
            scores_false = sample["scores_false"][0]
            split = sample["split"][0]
            score = 1.0 if scores_true > scores_false else 0.0
            scores[split] += [score]
            scores["all"] += [score]

        metrics = {f"{split}_acc": np.mean(scores[split]) for split in scores.keys()}
        metrics["macro_avg_acc"] = np.mean(
            [
                metrics[key]
                for key in metrics.keys()
                if key.endswith("_acc") and not key.startswith("all")
            ]
        )
        metrics["micro_avg_acc"] = np.mean(scores["all"])
        metrics["all_scores"] = scores["all"]
        metrics["proverb_ending_scores"] = scores["proverb_ending"]
        metrics["proverb_translation_scores"] = scores["proverb_ending"]
        metrics["hate_speech_ending_scores"] = scores["proverb_ending"]
        metrics["history_of_science_qa_scores"] = scores["proverb_ending"]

        return metrics
