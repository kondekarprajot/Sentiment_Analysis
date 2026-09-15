"""Streamlit app for real-time tweet sentiment prediction."""

from __future__ import annotations

from pathlib import Path

import joblib
import streamlit as st

from src.preprocessing import clean_tweet


MODEL_PATH = Path("models/sentiment_model/sentiment_model.joblib")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Tweet Sentiment Analyzer", page_icon=":speech_balloon:", layout="centered")
st.title("Tweet Sentiment Analyzer")

if not MODEL_PATH.exists():
    st.warning("Train the model first with: python src/train.py --data data/training.1600000.processed.noemoticon.csv")
    st.stop()

model = load_model()
tweet = st.text_area("Enter a tweet", height=140, placeholder="Type or paste a tweet here...")

if st.button("Predict sentiment", type="primary"):
    cleaned = clean_tweet(tweet)
    if not cleaned:
        st.error("Please enter a tweet with meaningful text.")
    else:
        prediction = model.predict([cleaned])[0]
        probabilities = getattr(model, "predict_proba", None)
        st.subheader(prediction.title())
        st.caption(f"Cleaned text: {cleaned}")
        if probabilities:
            proba = model.predict_proba([cleaned])[0]
            labels = model.classes_
            st.bar_chart({label: float(score) for label, score in zip(labels, proba)})
