AG News Text Classification

This project implements text classification on the AG News dataset
 using Naive Bayes and Logistic Regression with TF-IDF and Count Vectorizer.

📌 Dataset

AG News dataset: A collection of over 120,000 news articles categorized into 4 classes:

🌍 World

🏅 Sports

💼 Business

💻 Sci/Tech

⚙️ Preprocessing

Convert text to lowercase

Remove punctuation and special characters

Apply TF-IDF and Count Vectorizer (max 3000 features, unigrams + bigrams)

🧠 Models Trained

Multinomial Naive Bayes (NB)

Logistic Regression (LR)

Each model was trained with both Count Vectorizer and TF-IDF.

📊 Evaluation Metrics

For each model, the following metrics are calculated:

Accuracy

Precision (weighted)

Recall (weighted)

F1 Score (weighted)

ROC-AUC (One-vs-Rest)

Confusion matrices are also plotted for all models.

✅ Results (Sample Output)
Model	Accuracy	F1 Score	Precision	Recall	ROC-AUC
Naive Bayes (Count)	0.87	0.87	0.87	0.87	0.95
Naive Bayes (TFIDF)	0.88	0.88	0.88	0.88	0.96
Logistic Regression (Count)	0.91	0.91	0.91	0.91	0.97
Logistic Regression (TFIDF)	0.92	0.92	0.92	0.92	0.98

(Logistic Regression with TF-IDF performed best ✅)

💾 Saving Models

Trained models and vectorizers are saved with joblib for later use:

agnews_model.pkl

agnews_tfidf.pkl

agnews_cv.pkl

🚀 How to Run

Install dependencies:

pip install -r requirements.txt


Open the notebook:

jupyter notebook ag_news_nb_lr.ipynb
with open("README.md", "a", encoding="utf-8") as f:
    f.write("""

## 🔍 Usage Example

You can test the saved model on your own text:

```python
import joblib

# Load model and vectorizer
model = joblib.load("agnews_model.pkl")
tfidf = joblib.load("agnews_tfidf.pkl")

# Example input
text = ["The stock market is looking strong today with major gains in tech."]

# Transform and predict
X = tfidf.transform(text)
prediction = model.predict(X)

# Class labels
labels = ['World', 'Sports', 'Business', 'Sci/Tech']
print("Predicted class:", labels[prediction[0]])

