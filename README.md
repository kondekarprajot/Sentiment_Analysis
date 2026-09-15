# Sentiment Analysis of Tweets using the Sentiment140 Dataset

This project builds a machine learning pipeline for classifying tweet sentiment as negative `0`, neutral `2`, or positive `4`.

Important dataset note: the most common Sentiment140 download contains 1.6 million tweets labelled only as `0` negative and `4` positive. If your file has no `2` labels, the scripts will train a binary negative/positive model. To train three classes, use a Sentiment140 variant or added labelled data that includes neutral `2` examples.

## Project Structure

```text
.ipynb_checkpoints/
data/
  sample_tweets.csv
models/
  sentiment_model/
    sentiment_model.joblib
reports/
  metrics.json
  figures/
src/
  preprocessing.py
  eda.py
  train.py
tests/
  test_preprocessing.py
tools/
  project_commands.md
utils/
  prediction_helper.py
.gitignore
app.py
Sentiment140_Tweet_Sentiment_Analysis.ipynb
requirements-advanced.txt
requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Place the Sentiment140 CSV in the `data` folder. The original file is commonly named:

```text
training.1600000.processed.noemoticon.csv
```

The scripts accept either a headered CSV with `target` and `text` columns or the original six-column Sentiment140 format:

```text
target, id, date, query, user, text
```

## Run EDA

```bash
python src/eda.py --data data/training.1600000.processed.noemoticon.csv --sample-size 100000
```

Outputs are saved to `reports/figures`:

- Sentiment distribution
- Top words by sentiment
- Word clouds by sentiment

## Train Models

```bash
python src/train.py --data data/training.1600000.processed.noemoticon.csv --sample-size 200000
```

The training script compares:

- Logistic Regression with TF-IDF
- Multinomial Naive Bayes with CountVectorizer
- Random Forest with TF-IDF

It saves:

- Best trained model: `models/sentiment_model/sentiment_model.joblib`
- Metrics: `reports/metrics.json`
- Confusion matrix: `reports/figures/confusion_matrix.png`

For a full training run, omit `--sample-size`. Logistic Regression or Naive Bayes is usually a better first choice than Random Forest for high-dimensional sparse text.

## Run the Streamlit App

```bash
streamlit run app.py
```

Enter a tweet and the app will show the predicted sentiment and class probabilities when available.

## Recommended Workflow

1. Run EDA on a sample to understand class balance and common words.
2. Train the baseline models on a sample, then increase sample size.
3. Compare accuracy, precision, recall, F1-score, and confusion matrix.
4. Use the saved model in the Streamlit app for real-time predictions.

## Optional Deep Learning Extension

For an advanced version, compare the classical models with an LSTM or transformer model. A strong modern baseline would be a fine-tuned BERT/RoBERTa sentiment classifier, but it requires more compute and a separate training pipeline.
