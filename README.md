# AG News Classification: Naive Bayes & Logistic Regression

This project implements **text classification** on the AG News dataset using **Naive Bayes** and **Logistic Regression** with both **CountVectorizer** and **TF-IDF** features.  
It includes model evaluation metrics, confusion matrices, and saved models for reuse.

---

## ✅ Results

| Model | Accuracy | F1 Score | Precision | Recall | ROC-AUC |
|-------|----------|----------|-----------|--------|---------|
| MultinomialNB_counts | 0.87 | 0.87 | 0.87 | 0.87 | 0.95 |
| MultinomialNB_tfidf | 0.88 | 0.88 | 0.88 | 0.88 | 0.96 |
| LogisticRegression_counts | 0.91 | 0.91 | 0.91 | 0.91 | 0.97 |
| LogisticRegression_tfidf | 0.92 | 0.92 | 0.92 | 0.92 | 0.98 |

### JSON Snippet of Results

```json
[
  {
    "model": "MultinomialNB_counts",
    "accuracy": 0.87,
    "f1_score": 0.87,
    "precision": 0.87,
    "recall": 0.87,
    "roc_auc_score": 0.95
  },
  {
    "model": "MultinomialNB_tfidf",
    "accuracy": 0.88,
    "f1_score": 0.88,
    "precision": 0.88,
    "recall": 0.88,
    "roc_auc_score": 0.96
  },
  {
    "model": "LogisticRegression_counts",
    "accuracy": 0.91,
    "f1_score": 0.91,
    "precision": 0.91,
    "recall": 0.91,
    "roc_auc_score": 0.97
  },
  {
    "model": "LogisticRegression_tfidf",
    "accuracy": 0.92,
    "f1_score": 0.92,
    "precision": 0.92,
    "recall": 0.92,
    "roc_auc_score": 0.98
  }
]
🔍 Usage Example
You can test the saved model on your own text:

python
Copy code
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
Expected Output:

cpp
Copy code
Predicted class: Business
📦 Dataset
The AG News dataset consists of 120,000 training samples and 7,600 test samples across 4 categories:

World

Sports

Business

Sci/Tech

📝 Preprocessing
Convert text to lowercase

Remove punctuation

Vectorize text using CountVectorizer and TF-IDF

Consider unigram and bigram features

🛠 Models Trained
Multinomial Naive Bayes (Counts & TF-IDF)

Logistic Regression (Counts & TF-IDF)

📂 Saved Models
agnews_model.pkl → trained classifier

agnews_tfidf.pkl → TF-IDF vectorizer

agnews_cv.pkl → CountVectorizer

🎯 Evaluation
Models are evaluated using:

Accuracy

F1 Score

Precision

Recall

ROC-AUC Score

Confusion Matrix visualization

✅ Notes
High-level performance observed for Logistic Regression with TF-IDF.

Models are ready to be deployed or reused in other projects.
