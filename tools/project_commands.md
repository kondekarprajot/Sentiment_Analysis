# Useful Project Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook Sentiment140_Tweet_Sentiment_Analysis.ipynb
```

Train the model:

```bash
python src/train.py --data data/training.1600000.processed.noemoticon.csv --sample-size 200000
```

Run EDA:

```bash
python src/eda.py --data data/training.1600000.processed.noemoticon.csv --sample-size 100000
```

Run the Streamlit app:

```bash
streamlit run app.py
```

Run tests:

```bash
pytest
```
