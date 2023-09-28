"""Evaluation helper functions"""
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score


def generate_metrics(y_true, y_pred):
    """Generate the task specified evaluation metrics

    ARGS:
        y_true: An array of the true y labels
        y_pred: An array of the predict y labels

    RETURNS:
        f1: The F1 score metric
        accuracy: Percentage of correct predictions
        recall: The recall metric
        roc_auc: The ROC AUC metric
    """
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    return f1, accuracy, recall, roc_auc
