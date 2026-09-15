from __future__ import annotations

from pathlib import Path

import joblib

from src.preprocessing import clean_tweet


DEFAULT_MODEL_PATH = Path("models/sentiment_model/sentiment_model.joblib")


def load_sentiment_model(model_path: Path = DEFAULT_MODEL_PATH):
    return joblib.load(model_path)


def predict_tweet_sentiment(tweet: str, model_path: Path = DEFAULT_MODEL_PATH) -> str:
    model = load_sentiment_model(model_path)
    return str(model.predict([clean_tweet(tweet)])[0])
