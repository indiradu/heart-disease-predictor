import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier, StackingClassifier
from xgboost import XGBClassifier

from preprocess import load_data, get_feature_lists, build_preprocessor

DATA_PATH = "data/heart_cleaned.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "heart_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

def evaluate_model(name,pipeline, X_test, y_test):
    y_pred= pipeline.predict(X_test)
    y_prob= pipeline.predict_proba(X_test)[:,1]

    metrics = {
        "model_name": name, 
        "accuracy": round(accuracy_score(y_test, y_pred),4),
        "precision": round(precision_score(y_test, y_pred),4),
        "recall": round(recall_score(y_test, y_pred),4),
        "f1_score": round(f1_score(y_test, y_pred),4),
        "roc_auc": round(roc_auc_score(y_test, y_prob),4),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }
    return metrics

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df=load_data(DATA_PATH)
    if "target" not in df.columns:
        raise ValueError("Target column not found. Expected 'target' or source column 'num'.")

    X=df.drop(columns=["target"])
    y=df["target"]

    numeric_features, categorical_features = get_feature_lists(df)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state =42, stratify = y)
    ensemble_model = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )),
        ("xgb", XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42
        ))
    ],
    voting="soft"
    )
    stacking_model = StackingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
        )),
        ("xgb", XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )),
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    stack_method="predict_proba",
    cv=5,
    passthrough=False
    )
    models = {
        "LogisticRegression": LogisticRegression(max_iter = 1000, random_state=42),
        "RandomForest" : RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,),
        "XGBoost":XGBClassifier(
             n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        ),
        "Ensemble": ensemble_model,
        "Stacking": stacking_model,
    }

    best_score= -1
    best_pipeline = None
    best_metrics = None 
    all_metrics ={}

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ])

        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(name, pipeline, X_test, y_test)
        all_metrics[name]=metrics

        print(f"\n{name}")
        print(f"Accuracy:  {metrics['accuracy']}")
        print(f"Precision: {metrics['precision']}")
        print(f"Recall:    {metrics['recall']}")
        print(f"F1 Score:  {metrics['f1_score']}")
        print(f"ROC AUC:   {metrics['roc_auc']}")

        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_pipeline = pipeline
            best_metrics = metrics

    joblib.dump(best_pipeline, MODEL_PATH)

    output_metrics = {
        "best_model": best_metrics["model_name"],
        "best_model_metrics": best_metrics,
        "all_models": all_metrics,
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(output_metrics, f, indent=4)


    print(f"\nBest model: {best_metrics['model_name']}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")

if __name__ == "__main__":
    main()

    


