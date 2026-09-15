"""Train and evaluate ML sentiment classifiers for Sentiment140-style data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from preprocessing import clean_tweet, label_name


EXPECTED_COLUMNS = ["target", "id", "date", "query", "user", "text"]


def load_dataset(path: Path, sample_size: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    if "target" not in df.columns or "text" not in df.columns:
        df = pd.read_csv(path, encoding="latin-1", header=None, names=EXPECTED_COLUMNS)
    df = df[["target", "text"]].dropna()
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
    df["clean_text"] = df["text"].map(clean_tweet)
    df["sentiment"] = df["target"].map(label_name)
    df = df[df["clean_text"].str.len() > 0]
    return df


def build_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression_tfidf": Pipeline(
            [
                ("features", TfidfVectorizer(max_features=100000, ngram_range=(1, 2), min_df=2)),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)),
            ]
        ),
        "naive_bayes_count": Pipeline(
            [
                ("features", CountVectorizer(max_features=80000, ngram_range=(1, 2), min_df=2)),
                ("model", MultinomialNB()),
            ]
        ),
        "random_forest_tfidf": Pipeline(
            [
                ("features", TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2)),
                ("model", RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1, class_weight="balanced")),
            ]
        ),
    }


def save_confusion_matrix(y_true: pd.Series, y_pred: list[str], labels: list[str], out_path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Sentiment140 sentiment classifiers.")
    parser.add_argument("--data", type=Path, default=Path("data/sample_tweets.csv"))
    parser.add_argument("--sample-size", type=int, default=None, help="Use a random sample for faster experiments.")
    parser.add_argument("--model-dir", type=Path, default=Path("models/sentiment_model"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    args.model_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "figures").mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data, args.sample_size)
    labels = sorted(df["sentiment"].unique())
    if len(labels) < 2:
        raise ValueError("At least two sentiment classes are required for model training.")

    x_train, x_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["sentiment"],
        test_size=args.test_size,
        random_state=42,
        stratify=df["sentiment"] if df["sentiment"].value_counts().min() > 1 else None,
    )

    results: dict[str, dict[str, object]] = {}
    best_name = ""
    best_score = -1.0
    best_model: Pipeline | None = None
    best_pred: list[str] | None = None

    for name, model in build_models().items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        score = accuracy_score(y_test, pred)
        report = classification_report(y_test, pred, output_dict=True, zero_division=0)
        results[name] = {"accuracy": score, "classification_report": report}
        if score > best_score:
            best_name = name
            best_score = score
            best_model = model
            best_pred = pred.tolist()

    assert best_model is not None and best_pred is not None
    joblib.dump(best_model, args.model_dir / "sentiment_model.joblib")
    with open(args.report_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)

    save_confusion_matrix(y_test, best_pred, labels, args.report_dir / "figures" / "confusion_matrix.png")
    print(f"Best model: {best_name} | accuracy: {best_score:.4f}")
    print(f"Saved model to {args.model_dir / 'sentiment_model.joblib'}")


if __name__ == "__main__":
    main()
