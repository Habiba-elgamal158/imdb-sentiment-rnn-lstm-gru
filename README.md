# 🎬 IMDB Sentiment Analysis: RNN vs LSTM vs GRU

A comparison of recurrent neural networks for binary sentiment classification of movie reviews, benchmarked against a classical **TF-IDF + Logistic Regression** baseline.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Habiba-elgamal158/imdb-sentiment-rnn-lstm-gru/blob/main/sentiment_analysis_rnn_lstm_gru.ipynb)

🔗 **Live demo:** [Movie Review Sentiment](https://imdb-sentiment-rnn-lstm-gru-dnncvysym2hddwreougzme.streamlit.app/)
## Problem

Classify an IMDB movie review as **positive** or **negative**. The goal is to compare how Simple RNN, LSTM, GRU and Bidirectional LSTM handle long text, and to check whether deep models actually beat a simple linear baseline.

## Dataset

[IMDB Movie Reviews](https://keras.io/api/datasets/imdb/) (built into Keras): 25,000 training and 25,000 test reviews, perfectly balanced (50% positive / 50% negative). Reviews come pre-encoded as word IDs.

## Approach

1. **Exploration:** class balance, review length distribution, decoding IDs back to text.
2. **Preprocessing:** vocabulary limited to the 10,000 most frequent words, every review padded/truncated to 200 tokens (`pre` padding and truncating).
3. **Models** (same Embedding(128) → recurrent layer (64 units) → Dropout(0.5) → Dense(sigmoid), same optimizer and callbacks for a fair comparison):
   - Simple RNN
   - LSTM
   - GRU
   - Bidirectional LSTM
4. **Baseline:** TF-IDF (unigrams + bigrams) with Logistic Regression.
5. **Training:** 20% of the training set used for validation, EarlyStopping on `val_loss` with `restore_best_weights`.
6. **Evaluation:** Accuracy, Precision, Recall, F1, confusion matrices, learning curves and training time, all on the untouched test set.
7. **Error analysis:** inspection of reviews the best deep model gets wrong.
8. **Deployment:** the TF-IDF + Logistic Regression model, the best performer, is served through a Streamlit app.

## Results (test set, 25,000 reviews)

| Model | Accuracy | Precision | Recall | F1 | Training time (s) |
|---|---|---|---|---|---|
| Simple RNN | 0.8199 | 0.7772 | 0.8970 | 0.8328 | 60 |
| LSTM | 0.8605 | 0.8925 | 0.8198 | 0.8546 | 41 |
| GRU | 0.8607 | 0.8347 | 0.8995 | 0.8659 | 35 |
| BiLSTM | 0.8630 | 0.8409 | 0.8953 | 0.8673 | 65 |
| **TF-IDF + Logistic Regression** | **0.9080** | **0.9053** | **0.9112** | **0.9083** | **30** |

_Training times are wall-clock on Google Colab and depend on the hardware (an earlier run took 200-850 s per neural model), so they are only indicative. Neural-network scores vary slightly between runs; the TF-IDF baseline is deterministic._

## Key findings

- **A simple TF-IDF + Logistic Regression baseline beat every recurrent model** (0.908 accuracy vs 0.863 for the best, BiLSTM) at an equal or lower training cost. Deep learning was not worth its complexity on this task.
- **BiLSTM, GRU and LSTM are practically tied** (accuracy 0.861-0.863); Simple RNN is clearly weaker (0.820). In an earlier run of the same code the ranking among the three differed (LSTM was on top), so differences of one or two points are within run-to-run noise.
- **All deep models overfit quickly:** the best validation loss appeared in epoch 2-3 while training accuracy rose to 96-98%.
- **Negation is fragile:** "The movie was not good and I hated it" was classified as positive by Simple RNN and BiLSTM; LSTM and GRU got it right with only 52% and 60% confidence.
- **The TF-IDF model handles negation better than the networks:** it classifies "The movie was not good and I hated it" as negative (84% confidence), because the two-word feature *not good* pushes the score toward negative; Simple RNN and BiLSTM missed this sentence.
- **Error analysis:** the BiLSTM misclassified 3,426 of 25,000 test reviews (13.7%), mostly long reviews that mix praise and criticism, or contain many unknown (`<UNK>`) words.

## Lessons learned

- **Padding matters for RNNs.** An earlier run with `post` padding and truncating left Simple RNN and GRU at about 51% accuracy (chance level), because the recurrent state had to cross many padding tokens before the prediction. Switching to `pre` padding/truncating, a longer EarlyStopping patience and a fresh callback per model fixed the training.
- **Always run a strong classical baseline** before claiming a deep model is better.
- **Single runs are noisy.** Setting seeds is not enough for fully reproducible GPU training, so close neural models should be compared over several runs.
- **Overfitting appears fast with embeddings learned from scratch;** regularization or pretrained embeddings would be the next step.

## Web app

Because the TF-IDF + Logistic Regression model beat all neural networks, it is the one deployed: it is accurate (0.908 test accuracy), fast, and light enough to run on a free host without TensorFlow.

The app lets you type or paste a movie review and returns:
- the predicted sentiment (positive, negative, or *uncertain* when the model is close to 50/50) with probabilities;
- the **words and two-word phrases that pushed the prediction** toward each class (tf-idf value × model coefficient), so the decision is explainable;
- a note when some words fall outside the model's 10,000-word vocabulary.

New text is encoded exactly like the training data (`<START>` token, word IDs shifted by +3, unknown words mapped to `<UNK>`).

## Limitations

- Sarcasm fools the model: "Oh great, another two hours of my life I will never get back. Truly a masterpiece of boredom." is classified as positive (about 69%), because the praise words outweigh the rest.
- Word-level IDs with a 10,000-word vocabulary; rare words become `<UNK>`.
- Embeddings are learned from scratch (no pretrained GloVe or transformer embeddings).
- Only the first or last 200 tokens of long reviews are used.

## Run it

**The notebook** (needs TensorFlow; a GPU such as Google Colab speeds it up):

```bash
git clone https://github.com/Habiba-elgamal158/imdb-sentiment-rnn-lstm-gru.git
cd imdb-sentiment-rnn-lstm-gru
pip install -r requirements-notebook.txt
jupyter notebook sentiment_analysis_rnn_lstm_gru.ipynb
```

The dataset downloads automatically through Keras.

**The web app:**

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Project structure

```
├── sentiment_analysis_rnn_lstm_gru.ipynb   # analysis, training and evaluation
├── app.py                                  # Streamlit app
├── tfidf_vectorizer.pkl                    # fitted TF-IDF vectorizer
├── sentiment_model.pkl                     # trained Logistic Regression model
├── word_index.json                         # word -> index map (10,000-word vocabulary)
├── requirements.txt                        # dependencies of the web app
├── requirements-notebook.txt               # dependencies of the notebook
├── .streamlit/config.toml                  # app theme
└── README.md
```

## Tech stack

Python · TensorFlow / Keras · scikit-learn · pandas · NumPy · Matplotlib · Streamlit
