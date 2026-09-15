"""Exploratory analysis for Sentiment140-style CSV files."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from wordcloud import WordCloud
except ImportError:  # Word clouds are useful, but the rest of EDA should still run.
    WordCloud = None

from preprocessing import clean_tweet, label_name


EXPECTED_COLUMNS = ["target", "id", "date", "query", "user", "text"]


def load_dataset(path: Path, sample_size: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    if "target" not in df.columns or "text" not in df.columns:
        df = pd.read_csv(path, encoding="latin-1", header=None, names=EXPECTED_COLUMNS)
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
    df = df[["target", "text"]].dropna()
    df["clean_text"] = df["text"].map(clean_tweet)
    df["sentiment"] = df["target"].map(label_name)
    return df


def plot_label_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(7, 4))
    order = ["negative", "neutral", "positive"]
    sns.countplot(data=df, x="sentiment", order=[x for x in order if x in set(df["sentiment"])])
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Tweet count")
    plt.tight_layout()
    plt.savefig(out_dir / "sentiment_distribution.png", dpi=160)
    plt.close()


def plot_top_words(df: pd.DataFrame, out_dir: Path, n_words: int = 20) -> None:
    stop_words = {
        "the", "a", "an", "and", "or", "is", "are", "was", "were", "to", "of", "in",
        "for", "on", "with", "this", "that", "it", "my", "i", "you", "me", "at",
    }
    for sentiment, group in df.groupby("sentiment"):
        words = " ".join(group["clean_text"]).split()
        counts = Counter(w for w in words if w not in stop_words)
        common = counts.most_common(n_words)
        if not common:
            continue
        word_df = pd.DataFrame(common, columns=["word", "count"])
        plt.figure(figsize=(8, 5))
        sns.barplot(data=word_df, x="count", y="word")
        plt.title(f"Top Words in {sentiment.title()} Tweets")
        plt.xlabel("Frequency")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(out_dir / f"top_words_{sentiment}.png", dpi=160)
        plt.close()


def make_wordclouds(df: pd.DataFrame, out_dir: Path) -> None:
    if WordCloud is None:
        print("Skipping word clouds because the optional wordcloud package is not installed.")
        return

    for sentiment, group in df.groupby("sentiment"):
        text = " ".join(group["clean_text"])
        if not text.strip():
            continue
        cloud = WordCloud(width=1000, height=600, background_color="white", collocations=False).generate(text)
        plt.figure(figsize=(10, 6))
        plt.imshow(cloud, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"wordcloud_{sentiment}.png", dpi=160)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDA for Sentiment140 data.")
    parser.add_argument("--data", type=Path, default=Path("data/sample_tweets.csv"))
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(args.data, args.sample_size)
    plot_label_distribution(df, args.out_dir)
    plot_top_words(df, args.out_dir)
    make_wordclouds(df, args.out_dir)
    print(f"Saved EDA figures to {args.out_dir}")


if __name__ == "__main__":
    main()
