# fake_news_app.py
import streamlit as st
import pickle
import re
import string
import base64
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# --------------------------------------------------------
# 1️⃣ PAGE CONFIG
# --------------------------------------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# --------------------------------------------------------
# 2️⃣ BACKGROUND IMAGE
# --------------------------------------------------------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded_str = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_str}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}

        .center-box {{
            max-width: 700px;
            margin: auto;
            padding: 20px;
        }}

        .red-button > button {{
            background-color: #cc0000 !important;
            color: white !important;
            font-weight: bold !important;
            border-radius: 8px !important;
            height: 50px !important;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("bg.jpg")

# --------------------------------------------------------
# 3️⃣ LOAD MODEL & VECTORIZER
# --------------------------------------------------------
@st.cache_resource
def load_model():
    return pickle.load(open("svm_model.pkl", "rb"))

@st.cache_resource
def load_vectorizer():
    return pickle.load(open("tfidf_vectorizer.pkl", "rb"))

svm_model = load_model()
vectorization = load_vectorizer()

# --------------------------------------------------------
# 4️⃣ TEXT CLEANING (NO NLTK, NO SPACY — Streamlit Safe)
# --------------------------------------------------------
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
stop_words = ENGLISH_STOP_WORDS

def wordopt(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = []
    for w in text.split():
        if w not in stop_words:
            words.append(w)

    return " ".join(words)

# --------------------------------------------------------
# 5️⃣ HEADER
# --------------------------------------------------------
st.markdown(
    """
    <div class="center-box">
    <h1 style="text-align:center; color:white; 
        text-shadow: 0 0 10px black; font-size:42px;">
        📰 Fake News Detection
    </h1>
    <p style="text-align:center; color:white; font-size:20px;">
        Enter any news text and let AI detect if it's Fake or Real.
    </p>
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------------
# 6️⃣ INPUT BOX
# --------------------------------------------------------
with st.container():
    st.markdown("<div class='center-box'>", unsafe_allow_html=True)

    news_input = st.text_area(
        "Paste your news article text:",
        height=180,
        placeholder="Enter news text here..."
    )

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------
# 7️⃣ RED PREDICT BUTTON
# --------------------------------------------------------
st.markdown("<div class='red-button'>", unsafe_allow_html=True)
predict_clicked = st.button("Predict")
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------
# 8️⃣ PREDICTION
# --------------------------------------------------------
if predict_clicked:

    if news_input.strip() == "":
        st.warning("⚠️ Please enter some news text!")
    else:

        with st.spinner("Analyzing the article... ⏳"):

            cleaned = wordopt(news_input)
            vect = vectorization.transform([cleaned])

            proba = svm_model.predict_proba(vect)[0]
            pred = np.argmax(proba)
            conf = proba[pred] * 100

        # Determine label + color
        if pred == 0:
            color = "#ff1a1a"
            label = "Fake News ❌"
        else:
            color = "#28a745"
            label = "Real News ✅"

        # RESULT CARD
        html_result = f"""
        <div style="
            background: rgba(255,255,255,0.92);
            padding:25px; 
            border-radius:15px;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.50);
            text-align:center;
            width: 80%;
            margin: auto;
        ">

            <h2 style="
                color:{color};
                font-size:40px;
                font-weight:800;
                margin-bottom:10px;
            ">
                {label}
            </h2>

            <h3 style="
                color:#000000;
                font-size:30px;
                font-weight:800;
                margin-top:5px;
            ">
                Confidence: {conf:.2f}%
            </h3>

        </div>
        """

        components.html(html_result, height=220)

        # BAR GRAPH
        fig, ax = plt.subplots(figsize=(3.5, 3))
        labels = ["Fake News", "Real News"]
        values = [proba[0] * 100, proba[1] * 100]
        colors = ['#ff4d4d', '#4CAF50']

        ax.bar(labels, values, color=colors)
        ax.set_ylim([0, 100])
        ax.set_ylabel("Probability (%)")
        ax.set_title("Confidence Chart")

        st.pyplot(fig)
