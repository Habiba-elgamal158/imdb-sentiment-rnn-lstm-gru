import json
import re

import joblib
import streamlit as st

st.set_page_config(page_title="Movie Review Sentiment", page_icon="🎬", layout="centered")

TEST_ACCURACY = "90.8%"      # test-set accuracy of the saved model (from the notebook)
UNCERTAIN_BELOW = 0.60       # below this confidence the review is shown as "uncertain"

EXAMPLES = {
    "😍 Positive": "This movie was absolutely amazing. The acting was brilliant and I loved every minute of it.",
    "😡 Negative": "A boring, predictable film. The plot made no sense and the acting was terrible.",
    "🤔 Mixed": "The visuals were beautiful but the story was weak and the ending was disappointing.",
    "🙅 Negation": "The movie was not good and I hated it.",
}


# ----------------------------------------------------------------- model
@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    model = joblib.load("sentiment_model.pkl")
    with open("word_index.json", "r", encoding="utf-8") as f:
        word_index = json.load(f)                       # word -> Keras index (without the +3 shift)
    id_to_word = {idx + 3: w for w, idx in word_index.items()}
    feature_names = vectorizer.get_feature_names_out()
    return vectorizer, model, word_index, id_to_word, feature_names


vectorizer, model, word_index, id_to_word, feature_names = load_artifacts()


def encode(text):
    """Encode text exactly like imdb.load_data: <START>=1 first, ids shifted by +3, unknown -> <UNK>=2."""
    words = re.findall(r"[a-z']+", text.lower())
    ids = [1] + [word_index[w] + 3 if w in word_index else 2 for w in words]
    unknown = [w for w in words if w not in word_index]
    return " ".join(map(str, ids)), len(words), len(unknown)


def explain(X, top_n=6):
    """Contribution of each word/bigram to the prediction (tf-idf value x coefficient)."""
    coefs = model.coef_[0]
    row = X.tocoo()
    scores = {}
    for j, value in zip(row.col, row.data):
        tokens = feature_names[j].split()
        if any(int(t) < 3 for t in tokens):            # skip <PAD>, <START>, <UNK>
            continue
        phrase = " ".join(id_to_word.get(int(t), "?") for t in tokens)
        scores[phrase] = scores.get(phrase, 0.0) + float(value * coefs[j])
    pos = sorted([(p, s) for p, s in scores.items() if s > 0], key=lambda x: -x[1])[:top_n]
    neg = sorted([(p, s) for p, s in scores.items() if s < 0], key=lambda x: x[1])[:top_n]
    return pos, neg


def contrib_html(items, css_class, max_value):
    if not items:
        return '<div class="empty">Nothing notable.</div>'
    html = ""
    for phrase, score in items:
        width = max(6, int(abs(score) / max_value * 100))
        html += (f'<div class="contrib"><div class="w">{phrase}</div>'
                 f'<div class="cbar"><div class="cfill {css_class}" style="width:{width}%"></div></div></div>')
    return html


# ----------------------------------------------------------------- style
st.markdown("""
<style>
.stApp { background: #f3f5f9; }
header[data-testid="stHeader"] { background: #0b1220; }
.block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 760px; }

/* sidebar */
section[data-testid="stSidebar"] > div { background: linear-gradient(180deg, #0b1a33 0%, #12274a 100%); }
section[data-testid="stSidebar"] * { color: #e8edf6; }
.brand { font-size: 1.05rem; font-weight: 700; letter-spacing: .5px; margin-bottom: 1.6rem; }
.side-title { font-size: 1rem; font-weight: 700; margin: .4rem 0 .6rem 0; }
.side-text { font-size: .85rem; line-height: 1.6; color: #cbd5e6; }
.side-label { font-size: .85rem; font-weight: 700; margin-top: .7rem; }
.side-value { font-size: .85rem; color: #cbd5e6; margin-bottom: .2rem; }
.side-hr { border: none; border-top: 1px solid rgba(255,255,255,.12); margin: 1.6rem 0; }
.side-foot { font-size: .72rem; color: #9fb0cc; margin-top: 2.5rem; }

/* hero */
.hero { background: linear-gradient(135deg, #0f2342 0%, #1d4f66 65%, #2a7a78 100%);
        border-radius: 18px; padding: 2.2rem 2.2rem 2rem 2.2rem; margin-bottom: 2.2rem;
        box-shadow: 0 6px 18px rgba(15,35,66,.18); }
.hero h1 { color: #fff; font-size: 1.9rem; margin: 0 0 1.2rem 0; padding: 0; }
.hero p { color: #dbe6f2; font-size: .92rem; line-height: 1.6; margin: 0; }

.page-h2 { font-size: 1.8rem; font-weight: 600; color: #16213a; margin: 1rem 0 1rem 0; }
.section { font-size: 1.15rem; font-weight: 700; color: #16213a; margin: 2rem 0 .8rem 0; }
label p, div[data-testid="stWidgetLabel"] p { color: #4b5568 !important; font-size: .8rem !important; }
textarea { border-radius: 10px !important; }

/* buttons */
div.stButton > button { background: #2b7a78; color: #fff; border: none; border-radius: 8px;
                        padding: .55rem 1.1rem; font-weight: 600; }
div.stButton > button:hover { background: #236b69; color: #fff; }

/* result */
.result-card { background: #fff; border-radius: 12px; padding: 2rem 1.5rem; text-align: center;
               box-shadow: 0 2px 10px rgba(15,35,66,.08); border-left: 6px solid #2e9e6a; }
.result-card.neg { border-left-color: #d9534f; }
.result-card.unsure { border-left-color: #e0a030; }
.result-card h3 { margin: 0 0 .8rem 0; font-size: 1.25rem; font-weight: 600; color: #333; }
.result-card p { margin: 0; color: #4b5568; font-size: .85rem; }
.metric-card { background: #fff; border-radius: 12px; padding: 1.1rem 1.2rem;
               box-shadow: 0 2px 10px rgba(15,35,66,.08); }
.metric-card .lbl { font-size: .75rem; color: #5b6578; }
.metric-card .val { font-size: 1.7rem; font-weight: 700; color: #16213a; margin-top: .2rem; }
.bar-label { font-size: .85rem; font-weight: 700; color: #16213a; margin: 1.2rem 0 .5rem 0; }
.bar-track { background: #1f2430; border-radius: 4px; height: 6px; width: 100%; }
.bar-fill { background: #1f78d4; height: 6px; border-radius: 4px; }

/* explanation */
.explain-card { background: #fff; border-radius: 12px; padding: 1rem 1.2rem;
                box-shadow: 0 2px 10px rgba(15,35,66,.08); }
.explain-card .t { font-size: .85rem; font-weight: 700; color: #16213a; margin-bottom: .6rem; }
.contrib { display: flex; align-items: center; gap: .6rem; margin: .35rem 0; }
.contrib .w { width: 38%; font-size: .85rem; color: #16213a; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.cbar { flex: 1; background: #eef1f6; border-radius: 4px; height: 8px; }
.cfill { height: 8px; border-radius: 4px; }
.cfill.pos { background: #2e9e6a; }
.cfill.neg { background: #d9534f; }
.empty { font-size: .8rem; color: #7a8497; }
.note { font-size: .78rem; color: #7a8497; margin-top: 1rem; }
.footer { text-align: center; color: #7a8497; font-size: .75rem; margin-top: 3rem; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(f"""
    <div class="brand">MOVIE SENTIMENT AI</div>
    <hr class="side-hr">
    <div class="side-title">About the Project</div>
    <div class="side-text">Classifies a movie review as positive or negative and shows
    which words influenced the decision.</div>
    <hr class="side-hr">
    <div class="side-title">Model Information</div>
    <div class="side-label">Algorithm</div><div class="side-value">TF-IDF + Logistic Regression</div>
    <div class="side-label">Task</div><div class="side-value">Binary Sentiment Classification</div>
    <div class="side-label">Training Data</div><div class="side-value">IMDB (25,000 reviews)</div>
    <div class="side-label">Test Accuracy</div><div class="side-value">{TEST_ACCURACY}</div>
    <hr class="side-hr">
    <div class="side-foot">IMDB Sentiment Analysis Project</div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------ hero
st.markdown("""
<div class="hero">
  <h1>Movie Review Sentiment</h1>
  <p>Write a movie review in English and the model will predict whether it is positive or negative.
  This simple TF-IDF model outperformed LSTM, GRU and BiLSTM networks in the project's comparison.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------- input
if "review" not in st.session_state:
    st.session_state["review"] = ""


def set_example(text):
    st.session_state["review"] = text


st.markdown('<div class="section">Try an example</div>', unsafe_allow_html=True)
cols = st.columns(len(EXAMPLES))
for col, (label, text) in zip(cols, EXAMPLES.items()):
    col.button(label, on_click=set_example, args=(text,), use_container_width=True)

st.markdown('<div class="section">Your review</div>', unsafe_allow_html=True)
review = st.text_area("Movie review", key="review", height=150,
                      placeholder="Type or paste a movie review here...")
analyze = st.button("Analyze Review")

# ---------------------------------------------------------------- result
if analyze:
    if not review.strip():
        st.warning("Please write a review first.")
    else:
        doc, n_words, n_unknown = encode(review)
        X = vectorizer.transform([doc])
        pos_p = float(model.predict_proba(X)[0, 1])
        neg_p = 1 - pos_p
        confidence = max(pos_p, neg_p)

        st.markdown('<div class="page-h2" style="margin-top:2.5rem">Prediction Result</div>',
                    unsafe_allow_html=True)
        if confidence < UNCERTAIN_BELOW:
            st.markdown("""<div class="result-card unsure"><h3>Uncertain</h3>
            <p>The model is not confident: the review looks mixed or neutral.</p></div>""",
                        unsafe_allow_html=True)
        elif pos_p >= 0.5:
            st.markdown(f"""<div class="result-card"><h3>Positive Review 😊</h3>
            <p>The model is {pos_p:.0%} confident that this review is positive.</p></div>""",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="result-card neg"><h3>Negative Review 😞</h3>
            <p>The model is {neg_p:.0%} confident that this review is negative.</p></div>""",
                        unsafe_allow_html=True)

        m1, m2 = st.columns(2, gap="medium")
        m1.markdown(f'<div class="metric-card"><div class="lbl">Positive Probability</div>'
                    f'<div class="val">{pos_p:.2%}</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="lbl">Negative Probability</div>'
                    f'<div class="val">{neg_p:.2%}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bar-label">Positive Probability</div>'
                    f'<div class="bar-track"><div class="bar-fill" style="width:{pos_p*100:.1f}%"></div></div>',
                    unsafe_allow_html=True)

        # which words influenced the decision
        pos_items, neg_items = explain(X)
        max_value = max([s for _, s in pos_items] + [abs(s) for _, s in neg_items] + [1e-9])
        st.markdown('<div class="section">What influenced the prediction</div>', unsafe_allow_html=True)
        e1, e2 = st.columns(2, gap="medium")
        e1.markdown(f'<div class="explain-card"><div class="t">Pushes toward positive</div>'
                    f'{contrib_html(pos_items, "pos", max_value)}</div>', unsafe_allow_html=True)
        e2.markdown(f'<div class="explain-card"><div class="t">Pushes toward negative</div>'
                    f'{contrib_html(neg_items, "neg", max_value)}</div>', unsafe_allow_html=True)

        note = ("Bars show each word's (or two-word phrase's) contribution to the model's score. ")
        if n_unknown:
            note += (f"Of {n_words} words, {n_unknown} fell outside the model's 10,000-word vocabulary "
                     "and were treated as unknown. ")
        note += "The model was trained on English movie reviews and may miss sarcasm."
        st.markdown(f'<div class="note">{note}</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">IMDB Sentiment Analysis Project</div>', unsafe_allow_html=True)
