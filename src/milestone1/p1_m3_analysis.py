import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)


import scipy.io
from sklearn.metrics import RocCurveDisplay
from datasets.path import DATASET_PATH
from src.milestone1.p1_m2_z_score import Zscorer
from src.milestone1.p1_m1_roc_auc import auc_roc
from src.milestone1.p1_m1_roc_auc import roc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    PrecisionRecallDisplay,
    precision_recall_curve,
    average_precision_score,
)
import random


def inject_uniform_in_range(min, max, samples, labels, number_injected):
    """
    -------------------------------------------------------------------------------------------------------------------------------
    Args:
        min: Minum value a randomly generated sample can have
        max: Maximum value a randomly generated sample can have
        samples: Original samples to be extended
        labels: Original labels (corresponding to `samples`) to be extended
        number_injected: Number of injected anomalies

    Returns:
        Tuple of extended samples and labels. All injected samples are considered anomalous.
    -------------------------------------------------------------------------------------------------------------------------------
    """

    rng = np.random.default_rng(666)
    synthetic = rng.uniform(low=min, high=max, size=number_injected)
    extended_samples = np.append(synthetic, samples)
    extended_labels = np.append(np.ones(number_injected), labels)

    return (extended_samples, extended_labels)


def summary_plot(samples, predictions, labels, title):
    """
    -------------------------------------------------------------------------------------------------------------------------------
    Args:
        samples: samples that were analyzed
        predictions: anomaly score predictions corresponding to the samples
        labels: actual labels corresponding to samples, 1 denotes an anomalous instance.
        title: Title for the generated plot

    Returns:
        Plot that shows:
            - distribution of the samples as a whole
            - distribution of the samples per inlier/outlier category
            - ROC curve with corresponding AUC
            - Precision recall curve with corresponding average precision
    -------------------------------------------------------------------------------------------------------------------------------
    """

    fig, axs = plt.subplots(4)
    fig.suptitle(title)

    ### FEATURE DISTRIBUTION HISTOGRAMS ###

    # PER CATEGORY
    outliers = samples[labels == 1]
    inliers = samples[~(labels == 1)]

    n, bins, patches = axs[0].hist(
        [outliers, inliers],
        80,
        density=True,
        color=["red", "blue"],
        alpha=0.5,
        histtype="step",
        label=["outliers", "inliers"],
    )

    axs[0].legend(prop={"size": 10})
    axs[0].set_title("Feature distribution per category")
    axs[0].set(xlabel="Value", ylabel="Density")

    # ALL DATA
    n, bins, patches = axs[1].hist(
        samples,
        80,
        density=True,
        color="black",
        alpha=0.45,
        histtype="stepfilled",
        label="all data",
    )
    axs[1].legend(prop={"size": 10})
    axs[1].set_title("Feature distribution all data")
    axs[1].set(xlabel="Value", ylabel="Density")

    ### ROC-AUC CURVE ###
    fpr, tpr = roc(labels, predictions)
    roc_auc = auc_roc(labels, predictions)

    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot(ax=axs[2])
    axs[2].set_title("ROC Curve")
    axs[2].legend(prop={"size": 10})
    axs[2].set_title("ROC Curve")

    ### PRECISION-RECALL CURVE ###
    precision, recall, _ = precision_recall_curve(labels, predictions)
    avg_precision = average_precision_score(labels, predictions)
    PrecisionRecallDisplay(
        precision=precision,
        recall=recall,
        average_precision=avg_precision,
        estimator_name="Example Model",
    ).plot(ax=axs[3])
    axs[3].set_title("Precision-Recall Curve")

    return axs


if __name__ == "__main__":
    data_set_path = DATASET_PATH + "/forestCover.mat"
    data_set = scipy.io.loadmat(data_set_path)
    all_features = data_set["X"]
    labels = data_set["y"].flatten()

    number_observations = len(all_features)
    number_outliers = np.count_nonzero(labels == 1)
    print(
        f"Number of observations: {number_observations}, number of outliers: {number_outliers}, percentage of outliers: {100.0*number_outliers/number_observations}"
    )

    feature_number = 0
    features = all_features[:, feature_number].flatten()

    ### VANILLA DATASET ###
    scorer = Zscorer(features)
    anomaly_scores = scorer.score(features)
    summary_plot(features, anomaly_scores, labels, "Summary")

    ### TEST WITH SYNTHETIC ANOMALIES ###
    features_injected, labels_injected = inject_uniform_in_range(
        2200, 2800, features, labels, 1000
    )
    anomaly_scores_injected = scorer.score(features_injected)
    summary_plot(
        features_injected,
        anomaly_scores_injected,
        labels_injected,
        "Summary (injected anomalies)",
    )

    plt.show()
