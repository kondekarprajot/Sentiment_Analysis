# Project Report: Sentiment Analysis of Tweets

## Background

Twitter data is a rich source of public opinion about products, events, brands, and social topics. This project uses Sentiment140-style labelled tweets to build supervised sentiment classifiers that can predict whether unseen tweet text is negative, neutral, or positive.

## Objective

Develop a machine learning model that classifies tweets into:

- `0`: negative
- `2`: neutral
- `4`: positive

The standard Sentiment140 dataset is often distributed with only `0` and `4` labels. The implementation supports `2` labels when they are present, but otherwise trains a binary model.

## Methodology

### Data Preprocessing

Tweets are cleaned by:

- Converting text to lowercase
- Removing URLs
- Removing mentions
- Preserving hashtag words while removing the `#` symbol
- Removing punctuation, numbers, and special characters
- Normalizing whitespace
- Expanding common negations such as `don't` to `do not`

### Exploratory Data Analysis

The EDA script generates:

- Sentiment distribution chart
- Top frequent words for each sentiment class
- Word clouds for each sentiment class

These outputs help identify class imbalance, noisy tokens, and repeated linguistic patterns.

### Feature Engineering

The project supports two common sparse text representations:

- TF-IDF vectors with unigram and bigram features
- Count vectors with unigram and bigram features

### Model Development

The training script compares:

- Logistic Regression
- Multinomial Naive Bayes
- Random Forest

The best model by accuracy is saved for later inference.

### Evaluation

Evaluation includes:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Metrics are saved in `reports/metrics.json`, and the confusion matrix is saved as an image.

## Expected Insights

After running on the full dataset, the project should reveal:

- Whether the dataset is balanced across sentiment labels
- Which words are most strongly associated with positive and negative tweets
- Which sentiment classes are most commonly confused
- Whether simple linear models outperform heavier models on sparse tweet text

## Deployment

The included Streamlit app loads the trained model and predicts sentiment for user-entered tweets in real time.
