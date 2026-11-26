🎬 IMDB Sentiment Analysis: RNN vs LSTM vs GRU

A comparison of recurrent neural networks for binary sentiment classification of movie reviews, benchmarked against a classical TF-IDF + Logistic Regression baseline.

Open in Colab

Problem

Classify an IMDB movie review as positive or negative. The goal is to compare how Simple RNN, LSTM, GRU and Bidirectional LSTM handle long text, and to check whether deep models actually beat a simple linear baseline.

Dataset

IMDB Movie Reviews (built into Keras): 25,000 training and 25,000 test reviews, perfectly balanced (50% positive / 50% negative). Reviews come pre-encoded as word IDs.

Approach
Exploration: class balance, review length distribution, decoding IDs back to text.
Preprocessing: vocabulary limited to the 10,000 most frequent words, every review padded/truncated to 200 tokens (pre padding and truncating).
Models (same Embedding(128) → recurrent layer (64 units) → Dropout(0.5) → Dense(sigmoid), same optimizer and callbacks for a fair comparison):
Simple RNN
LSTM
GRU
Bidirectional LSTM
Baseline: TF-IDF (unigrams + bigrams) with Logistic Regression.
Training: 20% of the training set used for validation, EarlyStopping on val_loss with restore_best_weights.
Evaluation: Accuracy, Precision, Recall, F1, confusion matrices, learning curves and training time, all on the untouched test set.
Error analysis: inspection of reviews the best deep model gets wrong.
Results (test set, 25,000 reviews)
Model	Accuracy	Precision	Recall	F1	Training time (s)
Simple RNN	0.805	0.786	0.839	0.811	206
LSTM	0.866	0.854	0.884	0.869	551
GRU	0.846	0.825	0.879	0.851	632
BiLSTM	0.863	0.879	0.842	0.860	851
TF-IDF + Logistic Regression	0.908	0.905	0.911	0.908	37

Training times are wall-clock on Google Colab and include different numbers of epochs (EarlyStopping), so they are only indicative.

Key findings
A simple TF-IDF + Logistic Regression baseline beat every recurrent model (0.908 accuracy vs 0.866 for the best, LSTM) and trained in about 37 seconds instead of 5-14 minutes. Deep learning was not worth its cost on this task.
LSTM was the best deep model (F1 0.869), followed closely by BiLSTM (0.860) and GRU (0.851); Simple RNN was clearly weaker (0.811). The gaps among the top three are small and come from a single run.
All deep models overfit quickly: the best validation loss appeared in epoch 1-2 while training accuracy rose to 95-99%.
Negation is a weak spot: "The movie was not good and I hated it" was classified as positive by Simple RNN, GRU and BiLSTM; only the LSTM got it right.
Error analysis: the LSTM misclassified 3,345 of 25,000 test reviews (13.4%), mostly long reviews that mix praise and criticism, or contain many unknown (<UNK>) words.
Lessons learned
Padding matters for RNNs. An earlier run with post padding and truncating left Simple RNN and GRU at about 51% accuracy (chance level), because the recurrent state had to cross many padding tokens before the prediction. Switching to pre padding/truncating, a longer EarlyStopping patience and a fresh callback per model fixed the training.
Always run a strong classical baseline before claiming a deep model is better.
Overfitting appears fast with embeddings learned from scratch; regularization or pretrained embeddings would be the next step.
Limitations
Word-level IDs with a 10,000-word vocabulary; rare words become <UNK>.
Embeddings are learned from scratch (no pretrained GloVe or transformer embeddings).
Only the first or last 200 tokens of long reviews are used.
Run it
bash
git clone https://github.com/Habiba-elgamal158/imdb-sentiment-rnn-lstm-gru.git
cd imdb-sentiment-rnn-lstm-gru
pip install -r requirements.txt
jupyter notebook sentiment_analysis_rnn_lstm_gru.ipynb

The dataset downloads automatically through Keras. A GPU (e.g. Google Colab) speeds up training.

Project structure
├── sentiment_analysis_rnn_lstm_gru.ipynb   # full analysis, training and evaluation
├── requirements.txt
└── README.md
Tech stack

Python · TensorFlow / Keras · scikit-learn · pandas · NumPy · Matplotlib
