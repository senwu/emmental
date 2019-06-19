from emmental.metrics.accuracy import accuracy_scorer
from emmental.metrics.accuracy_f1 import accuracy_f1_scorer
from emmental.metrics.fbeta import f1_scorer, fbeta_scorer
from emmental.metrics.matthews_correlation import (
    matthews_correlation_coefficient_scorer,
)
from emmental.metrics.mean_squared_error import mean_squared_error_scorer
from emmental.metrics.pearson_correlation import pearson_correlation_scorer
from emmental.metrics.pearson_spearman import pearson_spearman_scorer
from emmental.metrics.precision import precision_scorer
from emmental.metrics.recall import recall_scorer
from emmental.metrics.roc_auc import roc_auc_scorer
from emmental.metrics.spearman_correlation import spearman_correlation_scorer

METRICS = {
    "accuracy": accuracy_scorer,
    "accuracy_f1": accuracy_f1_scorer,
    "precision": precision_scorer,
    "recall": recall_scorer,
    "f1": f1_scorer,
    "fbeta": fbeta_scorer,
    "matthews_correlation": matthews_correlation_coefficient_scorer,
    "mean_squared_error": mean_squared_error_scorer,
    "pearson_correlation": pearson_correlation_scorer,
    "pearson_spearman": pearson_spearman_scorer,
    "spearman_correlation": spearman_correlation_scorer,
    "roc_auc": roc_auc_scorer,
}

__all__ = [
    "accuracy_scorer",
    "accuracy_f1_scorer",
    "f1_scorer",
    "fbeta_scorer",
    "matthews_correlation_coefficient_scorer",
    "mean_squared_error_scorer",
    "pearson_correlation_scorer",
    "pearson_spearman_scorer",
    "precision_scorer",
    "recall_scorer",
    "roc_auc_scorer",
    "spearman_correlation_scorer",
]
