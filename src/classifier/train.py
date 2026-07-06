import argparse
import os
import json
import pickle
import mlflow
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--max-features", type=int, default=500)
    parser.add_argument("--C", type=float, default=1.0)
    args = parser.parse_args()

    with open(args.data_path) as f:
        data = json.load(f)

    texts = [d["text"] for d in data]
    labels = [d["category"] for d in data]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer(max_features=args.max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Manual MLflow logging (compatible with Azure ML)
    mlflow.log_param("max_features", args.max_features)
    mlflow.log_param("C", args.C)
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    mlflow.log_param("num_categories", len(set(labels)))

    model = LogisticRegression(C=args.C, max_iter=200, random_state=42)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("macro_f1", report["macro avg"]["f1-score"])
    mlflow.log_metric("macro_precision", report["macro avg"]["precision"])
    mlflow.log_metric("macro_recall", report["macro avg"]["recall"])

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
    print(classification_report(y_test, y_pred))

    # Save model artifacts to outputs/ (auto-uploaded by Azure ML)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("outputs/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    # Log artifacts to MLflow
    mlflow.log_artifact("outputs/model.pkl")
    mlflow.log_artifact("outputs/vectorizer.pkl")

    print("Model and vectorizer saved to outputs/")

if __name__ == "__main__":
    main()
