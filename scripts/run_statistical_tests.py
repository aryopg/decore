import logging
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv(".env")

import argparse
import json

import numpy as np
from scipy.stats import bootstrap
from statsmodels.stats.contingency_tables import mcnemar

from src import metrics


def perform_mcnemars_test(model_a_results, model_b_results):
    """
    Performs McNemar's Test on binary paired data.
    """
    # Ensure inputs are numpy arrays
    model_a_results = np.array(model_a_results)
    model_b_results = np.array(model_b_results)

    # Contingency table counts
    n_11 = np.sum((model_a_results == 1) & (model_b_results == 1))
    n_10 = np.sum((model_a_results == 1) & (model_b_results == 0))
    n_01 = np.sum((model_a_results == 0) & (model_b_results == 1))
    n_00 = np.sum((model_a_results == 0) & (model_b_results == 0))

    # Build contingency table
    table = [[n_11, n_10], [n_01, n_00]]

    # Perform McNemar's Test
    result = mcnemar(table, exact=False, correction=True)

    if result.pvalue < 0.0001:
        significance_text = "^{\\ast\\ast\\ast\\ast}"
    elif result.pvalue < 0.001:
        significance_text = "^{\\ast\\ast\\ast}"
    elif result.pvalue < 0.01:
        significance_text = "^{\\ast\\ast}"
    elif result.pvalue < 0.05:
        significance_text = "^{\\ast}"
    else:
        significance_text = ""

    print(f"Contingency Table: {table}")
    print(f"McNemar's Test Statistic: {result.statistic:.2f}{significance_text}")
    print(f"P-value: {result.pvalue:.4f}")


def perform_bootstrap_test(model_a_scores, model_b_scores, n_bootstraps=10000):
    """
    Performs bootstrap resampling to test statistical significance for continuous data.
    """
    # Ensure inputs are numpy arrays
    model_a_scores = np.array(model_a_scores)
    model_b_scores = np.array(model_b_scores)

    # Compute the observed mean difference
    observed_diff = np.mean(model_b_scores - model_a_scores)

    # Compute per-instance differences
    score_differences = model_b_scores - model_a_scores

    # Perform bootstrap resampling
    res = bootstrap(
        (score_differences,),
        np.mean,
        n_resamples=n_bootstraps,
        method="percentile",
        confidence_level=0.95,
    )
    confidence_interval = res.confidence_interval
    print(
        f"95% Confidence Interval: [{confidence_interval.low:.4f}, {confidence_interval.high:.4f}]"
    )

    # Calculate p-value (two-tailed test)
    bootstrap_means = res.bootstrap_distribution
    p_value = np.sum(
        np.abs(bootstrap_means - observed_diff) >= np.abs(observed_diff)
    ) / len(bootstrap_means)
    print(f"P-value: {p_value:.4f}")

    if p_value < 0.0001:
        significance_text = "^{\\ast\\ast\\ast\\ast}"
    elif p_value < 0.001:
        significance_text = "^{\\ast\\ast\\ast}"
    elif p_value < 0.01:
        significance_text = "^{\\ast\\ast}"
    elif p_value < 0.05:
        significance_text = "^{\\ast}"
    else:
        significance_text = ""

    print(f"Observed Mean Difference: {-observed_diff:.2f}{significance_text}")


def main():
    parser = argparse.ArgumentParser(
        description="Perform statistical significance testing."
    )
    parser.add_argument(
        "--model_a_results_filepath",
        type=str,
        required=True,
        help="Path to Model A results",
    )
    parser.add_argument(
        "--model_b_results_filepath",
        type=str,
        required=True,
        help="Path to Model B results",
    )
    parser.add_argument(
        "--task_metrics",
        type=str,
        required=True,
        help="The metric to use",
        choices=[
            "IFEval",
            "MemoTrap",
            "MuSiQue",
            "NQ",
            "NQSwap",
            "PopQA",
            "TriviaQA",
            "TruthfulQA",
            "XSum",
        ],
    )
    parser.add_argument(
        "--is_binary",
        action="store_true",
        help="Indicates whether the evaluation metric is binary.",
    )
    args = parser.parse_args()

    print(f"Model A Results: {args.model_a_results_filepath}")
    print(f"Model B Results: {args.model_b_results_filepath}")

    # Load model results
    print("Loading model results...")
    model_a_results = []
    with open(args.model_a_results_filepath, "r") as f:
        for line in f:
            model_a_results += [json.loads(line)]
    model_b_results = []
    with open(args.model_b_results_filepath, "r") as f:
        for line in f:
            model_b_results += [json.loads(line)]

    # Initialise metrics class
    metric_obj = getattr(metrics, args.task_metrics)()

    # Compute metrics
    print("Computing metrics for model A...")
    metrics_a = metric_obj(model_a_results)
    print("Computing metrics for model B...")
    metrics_b = metric_obj(model_b_results)

    # print(metrics_a)

    metrics_keys = [m for m in list(metrics_a.keys()) if m.endswith("_scores")]

    for metric_key in metrics_keys:
        print(metric_key)
        if not args.is_binary or "MC2" in metric_key or "MC3" in metric_key:
            perform_bootstrap_test(metrics_a[metric_key], metrics_b[metric_key])
        else:
            perform_mcnemars_test(metrics_a[metric_key], metrics_b[metric_key])


if __name__ == "__main__":
    main()
