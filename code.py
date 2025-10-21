"""
train.py — End-to-end scikit-learn pipeline on a synthetic binary classification dataset.

It will:
1) Generate a reproducible synthetic dataset (no external files needed)
2) Build preprocessing & modeling pipelines
3) Compare Logistic Regression vs Random Forest via GridSearchCV
4) Evaluate on a held-out test set
5) Save metrics, plots, and the best model to artifacts/
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


@dataclass
class Paths:
    base: str = "artifacts"
    models: str = os.path.join("artifacts", "models")
    metrics: str = os.path.join("artifacts", "metrics")
    plots: str = os.path.join("artifacts", "plots")


def ensure_dirs(paths: Paths) -> None:
    for p in (paths.base, paths.models, paths.metrics, paths.plots):
        os.makedirs(p, exist_ok=True)


def generate_synthetic_data(n_samples: int = 4000) -> pd.DataFrame:
    """
    Generate a tabular dataset with:
      - 8 numeric features from make_classification
      - 2 categorical features generated separately
      - binary target 'churn'
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        n_repeated=0,
        n_clusters_per_class=2,
        weights=[0.6, 0.4],
        flip_y=0.01,
        class_sep=1.0,
        random_state=RANDOM_STATE,
    )

    df_num = pd.DataFrame(
        X,
        columns=[f"num_feat_{i}" for i in range(X.shape[1])]
    )

    # Two categorical columns with moderate cardinality
    seg_values = ["basic", "plus", "pro"]
    region_values = ["north", "south", "east", "west"]

    df_cat = pd.DataFrame({
        "segment": np.random.choice(seg_values, size=n_samples, p=[0.5, 0.3, 0.2]),
        "region": np.random.choice(region_values, size=n_samples)
    })

    df = pd.concat([df_num, df_cat], axis=1)
    df["churn"] = y.astype(int)
    return df


def split_data(df: pd.DataFrame, target: str = "churn") -> Tuple[pd.DataFrame, ...]:
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipe = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop"
    )
    return preprocessor


def build_models(preprocessor: ColumnTransformer) -> dict:
    log_reg = Pipeline(steps=[
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=500, n_jobs=None, random_state=RANDOM_STATE))
    ])

    rf = Pipeline(steps=[
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE))
    ])

    param_grids = {
        "log_reg": {
            "clf__C": [0.1, 1.0, 5.0],
            "clf__solver": ["liblinear", "lbfgs"],
        },
        "rf": {
            "clf__n_estimators": [150, 300],
            "clf__max_depth": [None, 8, 14],
            "clf__min_samples_split": [2, 5],
        },
    }

    return {
        "log_reg": (log_reg, param_grids["log_reg"]),
        "rf": (rf, param_grids["rf"]),
    }


def evaluate_and_plot(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    paths: Paths
) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    roc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    cm = confusion_matrix(y_test, preds).tolist()

    # ROC Curve
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve")
    plt.savefig(os.path.join(paths.plots, "roc_curve.png"), bbox_inches="tight")
    plt.close()

    # Precision-Recall Curve
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    plt.title("Precision-Recall Curve")
    plt.savefig(os.path.join(paths.plots, "pr_curve.png"), bbox_inches="tight")
    plt.close()

    # Confusion Matrix (simple matplotlib)
    fig, ax = plt.subplots()
    mat = np.array(cm)
    im = ax.imshow(mat, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, str(v), ha="center", va="center")
    fig.colorbar(im)
    plt.savefig(os.path.join(paths.plots, "confusion_matrix.png"), bbox_inches="tight")
    plt.close()

    metrics = {
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm,
    }
    return metrics


def main():
    paths = Paths()
    ensure_dirs(paths)

    # 1) Data
    df = generate_synthetic_data(n_samples=4000)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df)

    # 2) Preprocessor & Models
    pre = build_preprocessor(X_train)
    models = build_models(pre)

    # 3) Grid search across models using the same validation data via CV
    results = []
    best_model = None
    best_score = -np.inf
    best_name = ""

    for name, (pipe, grid) in models.items():
        gs = GridSearchCV(
            pipe,
            param_grid=grid,
            scoring="roc_auc",
            cv=5,
            n_jobs=-1,
            verbose=0,
        )
        # Use train+valid for CV by concatenation (keeps a final untouched test set)
        X_cv = pd.concat([X_train, X_valid], axis=0)
        y_cv = pd.concat([y_train, y_valid], axis=0)

        gs.fit(X_cv, y_cv)
        score = gs.best_score_
        results.append({"model": name, "cv_roc_auc": float(score), "best_params": gs.best_params_})

        if score > best_score:
            best_score = score
            best_model = gs.best_estimator_
            best_name = name

    # 4) Final evaluation on the held-out test set
    metrics = evaluate_and_plot(best_model, X_test, y_test, paths)
    metrics["selected_model"] = best_name
    metrics["cv_roc_auc"] = float(best_score)

    # 5) Save artifacts
    joblib.dump(best_model, os.path.join(paths.models, "best_model.joblib"))
    with open(os.path.join(paths.metrics, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("===== Training complete =====")
    print(f"Selected model: {best_name}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
