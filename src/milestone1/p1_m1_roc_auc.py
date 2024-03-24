import numpy as np

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def roc(true_labels, predictions):
    """
    -------------------------------------------------------------------------------------------------------------------------------
    Args:
        true_labels: numpy array of true labels, positive class is signaled by a value of 1, any other value is assumed as negative class

        predictions: model class predictions. Higher values denote more likelihood of sample belonging to the positive class

    Returns:
        false positive rate and corresponding true positive rate arrays.
    -------------------------------------------------------------------------------------------------------------------------------
    """

    fpr_array = []
    tpr_array = []

    thresholds = [np.inf, 0.7, 0.3, 0.25, 0]

    for t in thresholds:
        tp = 0  # Initialize counters
        fp = 0
        fn = 0
        tn = 0

        for true_label, prediction in zip(true_labels, predictions):
            if true_label == 1 and prediction >= t:
                tp += 1
            elif true_label == 0 and prediction >= t:
                fp += 1
            elif true_label == 1 and prediction < t:
                fn += 1
            else:  # true_label == 0 and prediction < t
                tn += 1

        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)
        fpr_array.append(fpr)
        tpr_array.append(tpr)

    return np.array(fpr_array), np.array(tpr_array)


def roc_conceptual(true_labels, predictions, number_of_thresholds=10):
    """
    -------------------------------------------------------------------------------------------------------------------------------
    Args:
        true_labels: numpy array of true labels, positive class is signaled by a value of 1, any other value is assumed as negative class

        predictions: model class predictions. Higher values denote more likelihood of sample belonging to the positive class

        number_of_thresholds: number of abnormality thresholds?

    Returns:
        false positive rate and corresponding true positive rate arrays.
    -------------------------------------------------------------------------------------------------------------------------------
    """
    # Sort predictions and corresponding true labels based on the prediction scores in descending order
    sorted_data = sorted(
        zip(predictions, true_labels), key=lambda x: x[0], reverse=True
    )

    # Initialize counters and arrays
    tp = fp = 0
    fpr_array = [0]  # Start with 0 to represent the starting point of the ROC curve
    tpr_array = [0]
    total_positives = sum(true_labels)
    total_negatives = len(true_labels) - total_positives

    # Iterate through sorted predictions, accumulating TP and FP values for each unique prediction level
    prev_prediction = None
    for prediction, true_label in sorted_data:
        # Check if we have moved to a new prediction value
        if prediction != prev_prediction and prev_prediction is not None:
            # Calculate FPR and TPR for the previous group of the same predictions and add them to their respective arrays
            fpr = fp / total_negatives if total_negatives else 0
            tpr = tp / total_positives if total_positives else 0
            fpr_array.append(fpr)
            tpr_array.append(tpr)

        # Update TP or FP based on the current label
        if true_label == 1:
            tp += 1
        else:
            fp += 1

        prev_prediction = prediction

    # After the loop, add the final point (1,1) to FPR and TPR arrays
    fpr_array.append(1)
    tpr_array.append(1)

    return np.array(fpr_array), np.array(tpr_array)


def auc_roc(true_labels, predictions):
    """
    -------------------------------------------------------------------------------------------------------------------------------
    Args:
        true_labels: numpy array of true labels, positive class is signaled by a value of 1, any other value is assumed as negative class

        predictions: model class predictions. Higher values denote more likelihood of sample belonging to the positive class

    Returns:
        AUC Area under de ROC curve.
    -------------------------------------------------------------------------------------------------------------------------------
    """
    fpr, tpr = roc_conceptual(true_labels, predictions)
    return np.trapz(tpr, x=fpr)


if __name__ == "__main__":

    # Set up a random classification problem

    X, y = make_classification(random_state=87817)  # 8787917 #1231
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = SVC(random_state=0).fit(X_train, y_train)
    y_pred = clf.decision_function(X_test)

    # plot and display the sklearn ROC and AUC results

    RocCurveDisplay.from_predictions(y_test, y_pred, name="sklearn")

    # calls your roc and auc implementations

    fpr, tpr = roc(y_test, y_pred)
    roc_auc_aduh = auc_roc(y_test, y_pred)
    impl = RocCurveDisplay(
        fpr=fpr, tpr=tpr, roc_auc=roc_auc_aduh, estimator_name="Lerner"
    )
    impl.plot()

    plt.show()
