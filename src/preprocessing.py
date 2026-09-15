"""Text preprocessing utilities for Sentiment140-style tweets."""

from __future__ import annotations

import html
import re
import string


URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
MULTISPACE_RE = re.compile(r"\s+")


NEGATION_MAP = {
    "can't": "can not",
    "cannot": "can not",
    "won't": "will not",
    "n't": " not",
    "i'm": "i am",
    "it's": "it is",
    "that's": "that is",
    "what's": "what is",
    "there's": "there is",
    "you're": "you are",
    "they're": "they are",
    "we're": "we are",
    "i've": "i have",
    "don't": "do not",
    "didn't": "did not",
}


def clean_tweet(text: object) -> str:
    """Normalize noisy tweet text into a model-friendly lowercase string."""
    if text is None:
        return ""

    value = html.unescape(str(text)).lower()
    value = URL_RE.sub(" ", value)
    value = MENTION_RE.sub(" ", value)
    value = HASHTAG_RE.sub(r"\1", value)

    for src, dst in NEGATION_MAP.items():
        value = value.replace(src, dst)

    value = value.translate(str.maketrans("", "", string.punctuation))
    value = NON_ALPHA_RE.sub(" ", value)
    value = MULTISPACE_RE.sub(" ", value).strip()
    return value


def label_name(label: int | str) -> str:
    """Map Sentiment140 labels to human-readable names."""
    mapping = {0: "negative", 2: "neutral", 4: "positive", "0": "negative", "2": "neutral", "4": "positive"}
    return mapping.get(label, str(label))
