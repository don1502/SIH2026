"""Train the suspect-prediction model on the cleaned dataset.

Node-classification task: predict whether a person is a suspect (Prime Suspect /
Accused / Co-conspirator) vs a non-suspect (Witness), using multi-hop graph
features. Ground-truth labels come from case involvement and are never used as
features. Artifacts are written to backend/models/.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.config import settings
from app.data_access import load_cleaned_tables
from app.ml.features import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_labels,
    build_person_features,
)

MODEL_PATH = "suspect_model.joblib"
METRICS_PATH = "metrics.json"


def build_pipeline() -> Pipeline:
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    pre = ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("clf", model)])


def _feature_importances(pipe: Pipeline) -> list[dict]:
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    names = list(NUMERIC_FEATURES)
    ohe = pre.named_transformers_["cat"].named_steps["onehot"]
    names += list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    importances = clf.feature_importances_
    pairs = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)
    return [{"feature": n, "importance": round(float(v), 4)} for n, v in pairs[:15]]


def train() -> dict:
    tables = load_cleaned_tables()
    features = build_person_features(tables)
    labels = build_labels(tables)

    labeled = features.join(labels.rename("y"), how="inner").dropna(subset=["y"])
    X = labeled[ALL_FEATURES]
    y = labeled["y"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    cv_auc = cross_val_score(build_pipeline(), X, y, cv=5, scoring="roc_auc", n_jobs=-1)

    metrics = {
        "task": "suspect_prediction",
        "label": "involvement in {Prime Suspect, Accused, Co-conspirator} vs Witness",
        "n_labeled": int(len(labeled)),
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
        "test": {
            "accuracy": round(float((pred == y_test).mean()), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
            "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        },
        "cv_roc_auc_mean": round(float(cv_auc.mean()), 4),
        "cv_roc_auc_std": round(float(cv_auc.std()), 4),
        "feature_importances": _feature_importances(pipe),
    }

    # Retrain on all labeled data for the deployed model.
    final = build_pipeline()
    final.fit(X, y)

    models_dir = Path(settings.models_path)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, models_dir / MODEL_PATH)

    # Population feature stats used for per-person indicator explanations.
    pop_stats = {
        col: {"mean": float(X[col].mean()), "std": float(X[col].std() or 1.0)}
        for col in NUMERIC_FEATURES
    }
    metrics["population_stats"] = pop_stats
    with open(models_dir / METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    result = train()
    print(json.dumps({k: v for k, v in result.items() if k != "population_stats"}, indent=2))
