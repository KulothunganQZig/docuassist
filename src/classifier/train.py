import argparse
import os
import json
import mlflow
import mlflow.sklearn
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

    # Load training data
    with open(args.data_path) as f:
        data = json.load(f)

    texts = [d["text"] for d in data]
    labels = [d["category"] for d in data]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Featurize
    vectorizer = TfidfVectorizer(max_features=args.max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train
    mlflow.autolog()
    model = LogisticRegression(C=args.C, max_iter=200, random_state=42)
    model.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("macro_f1", report["macro avg"]["f1-score"])
    mlflow.log_param("max_features", args.max_features)
    mlflow.log_param("C", args.C)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
    print(classification_report(y_test, y_pred))

    # Save artifacts
    os.makedirs("outputs", exist_ok=True)
    mlflow.sklearn.log_model(model, "query-router-model")

    import pickle
    with open("outputs/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    mlflow.log_artifact("outputs/vectorizer.pkl")

if __name__ == "__main__":
    main()
