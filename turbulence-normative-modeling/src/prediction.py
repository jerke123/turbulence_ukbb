"""
Phase 2, part two: depressive symptom prediction.

The question is whether normative deviation scores carry more information about
depressive symptoms than the raw, uncorrected TD features. Both feature sets are
put through the same classifier and cross-validation scheme, and the difference
in performance is tested against a null built by permuting the labels.

The classifier is a logistic regression with an elastic net penalty (L1 and L2
weighted equally), optimised with the SAGA solver and a balanced class weighting,
evaluated by ten-fold cross-validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from . import config


def make_classifier(random_state: int = config.RANDOM_SEED) -> LogisticRegression:
    """The elastic net logistic regression used for every comparison."""
    return LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=config.ELASTIC_NET_L1_RATIO,
        C=config.ELASTIC_NET_C,
        class_weight="balanced",
        max_iter=config.MAX_ITER,
        random_state=random_state,
    )


def make_cv(
    n_splits: int = config.CV_FOLDS, random_state: int = config.RANDOM_SEED
) -> StratifiedKFold:
    """Stratified k-fold, so both classes appear in every fold."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


@dataclass
class CVResult:
    """Per-fold scores plus the pooled out-of-fold predictions for plotting."""

    auc: List[float]
    balanced_accuracy: List[float]
    f1: List[float]
    y_true: np.ndarray
    y_score: np.ndarray

    def means(self) -> Dict[str, float]:
        return {
            "auc": float(np.mean(self.auc)),
            "balanced_accuracy": float(np.mean(self.balanced_accuracy)),
            "f1": float(np.mean(self.f1)),
        }


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    model: LogisticRegression | None = None,
    cv=None,
) -> CVResult:
    """
    Run cross-validation, fitting the imputer and scaler inside each fold.

    Keeping preprocessing inside the fold matters: fitting a scaler or imputer on
    the whole dataset first would leak test-fold information into training.
    """
    model = model or make_classifier()
    cv = cv or make_cv()

    aucs, accuracies, f1s = [], [], []
    all_true, all_score = [], []

    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        aucs.append(roc_auc_score(y_test, probabilities))
        accuracies.append(balanced_accuracy_score(y_test, predictions))
        f1s.append(f1_score(y_test, predictions))

        all_true.append(y_test)
        all_score.append(probabilities)

    return CVResult(
        auc=aucs,
        balanced_accuracy=accuracies,
        f1=f1s,
        y_true=np.concatenate(all_true),
        y_score=np.concatenate(all_score),
    )


def permutation_test(
    X_deviation: np.ndarray,
    X_raw: np.ndarray,
    y: np.ndarray,
    observed_difference: float,
    n_permutations: int = config.N_PERMUTATIONS,
    metric: str = "auc",
    random_state: int = config.RANDOM_SEED,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Test whether the deviation scores outperform the raw features.

    The null distribution is built by shuffling the labels and re-running the
    full cross-validation for both feature sets, so that any advantage arising
    from the feature sets' differing dimensionality or scale is preserved under
    the null. The p-value is one-sided, counting permuted differences at least as
    large as the observed one, with the observed case included in both the
    numerator and denominator.
    """
    rng = np.random.default_rng(random_state)
    model = make_classifier()
    differences = np.empty(n_permutations)

    for i in range(n_permutations):
        y_permuted = rng.permutation(y)
        cv = make_cv(random_state=random_state + i)

        deviation = np.mean(getattr(cross_validate(X_deviation, y_permuted, model, cv), metric))
        raw = np.mean(getattr(cross_validate(X_raw, y_permuted, model, cv), metric))
        differences[i] = deviation - raw

        if verbose and (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_permutations} permutations")

    n_extreme = int(np.sum(differences >= observed_difference))
    p_value = (n_extreme + 1) / (n_permutations + 1)

    return {
        "observed_difference": float(observed_difference),
        "p_value": float(p_value),
        "null_mean": float(np.mean(differences)),
        "null_sd": float(np.std(differences)),
        "n_permutations": n_permutations,
    }


def compare_feature_sets(
    df: pd.DataFrame,
    raw_columns: Sequence[str],
    deviation_columns: Sequence[str],
    label_column: str,
    model: LogisticRegression | None = None,
    cv=None,
) -> Tuple[CVResult, CVResult, pd.DataFrame]:
    """
    Cross-validate the raw and deviation feature sets on the same participants.

    Returns both results and a tidy comparison table of the mean scores.
    """
    X_raw = df[list(raw_columns)].to_numpy(dtype=float)
    X_deviation = df[list(deviation_columns)].to_numpy(dtype=float)
    y = df[label_column].to_numpy(dtype=int)

    model = model or make_classifier()
    cv = cv or make_cv()

    deviation_result = cross_validate(X_deviation, y, model, cv)
    raw_result = cross_validate(X_raw, y, model, cv)

    deviation_means = deviation_result.means()
    raw_means = raw_result.means()

    table = pd.DataFrame(
        {
            "metric": ["AUC-ROC", "Balanced accuracy", "F1"],
            "deviation_scores": [
                deviation_means["auc"],
                deviation_means["balanced_accuracy"],
                deviation_means["f1"],
            ],
            "raw_features": [
                raw_means["auc"],
                raw_means["balanced_accuracy"],
                raw_means["f1"],
            ],
        }
    )
    table["difference"] = table["deviation_scores"] - table["raw_features"]

    return deviation_result, raw_result, table


def roc_points(result: CVResult) -> Tuple[np.ndarray, np.ndarray]:
    """Pooled out-of-fold false and true positive rates, for plotting."""
    fpr, tpr, _ = roc_curve(result.y_true, result.y_score)
    return fpr, tpr
