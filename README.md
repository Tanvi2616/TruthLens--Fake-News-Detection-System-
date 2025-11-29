## 📰 TruthLens – Fake News Detection System

TruthLens is an advanced Fake News Detection System that classifies news articles as Real or Fake using Machine Learning, NLP, and text vectorization techniques.
It extracts linguistic patterns from the text and uses classification models—especially Calibrated LinearSVC—to detect misinformation with high accuracy.

## 🚀 Features

✔ Classifies news articles as Real or Fake

✔ Uses TF-IDF vectorization + NLP preprocessing

✔ Multiple ML models compared (Logistic Regression, Naive Bayes, SVM, PAC, LinearSVC)

✔ Calibrated LinearSVC chosen as best model

✔ EDA with:

Fake vs Real distribution

Text length analysis

Word clouds

Histograms & box plots

✔ Accuracy, Precision, Recall, F1-score evaluation

✔ Clean workflow with modular code

✔ Optional UI using Streamlit/Flask

## 📂 Dataset Used

Fake and Real News Dataset – Kaggle

Link: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Contains ~44,900 news articles

Files included:

- Fake.csv

- True.csv

## 🛠️ Tech Stack

- Python

- Libraries: Pandas ,NumPy ,Scikit-learn, NLTK, Matplotlib, Seaborn

- Modeling: TF-IDF, LinearSVC, Logistic Regression, Naive Bayes

- UI: Streamlit 

## 🧠 How It Works

- Loads Fake & Real news datasets

- Labels and merges them into a single dataframe

- Performs NLP preprocessing:

Lowercasing

Stopwords removal

Punctuation removal

Lemmatization

- Converts text to vectors using TF-IDF

- Trains multiple ML models

- Selects the best-performing model (Calibrated LinearSVC)

- Saves the model as best_model.pkl

- Predicts news authenticity on new input

## 📊 Modeling & EDA
- Preprocessing

Tokenization

Lemmatization

Stopwords removal

TF-IDF vectorization

- EDA Insights

Real & Fake news counts are almost balanced

Text length distribution overlaps for both classes

Word clouds show:

Fake news → more political/sensational words

Real news → more factual terms

- Evaluation Metrics

Accuracy

Precision

Recall

F1-score

Confusion Matrix

## 📷 UI Preview

Visit the live app to try it out:

🔗 https://truthlens-fake-news-detection-system.streamlit.app/

## 🌐 Deployment

Hosted on Streamlit Cloud

## 🔮 Future Scope

- Add Deep Learning (LSTM / GRU / BERT)

- Real-time URL & article classification

- Chrome extension for instant fact-checking

- Multi-language news detection

- Add credibility scoring system

- API for external apps


## 📜 License

Licensed under the MIT License — free to use and modify.

## 👩‍💻 Author

Tanvi Yedvi

If you like this project, please give a ⭐ on GitHub — it motivates future improvements!
