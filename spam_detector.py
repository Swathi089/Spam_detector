# spam_detector.py

import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK data (only once)
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    stemmer = PorterStemmer()
    clean_words = [stemmer.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(clean_words)

df['clean_text'] = df['text'].apply(preprocess_text)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['label_num']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# Predict new email
def predict_spam(email):
    email_clean = preprocess_text(email)
    email_vector = vectorizer.transform([email_clean])
    prediction = model.predict(email_vector)[0]
    return "Spam" if prediction == 1 else "Not Spam"

# Try it!
print("\n📩 New Email Prediction Test:")
sample_email = "Congratulations! You've won a free ticket. Click here to claim it now!"
print("Message:", sample_email)
print("Prediction:", predict_spam(sample_email))

import joblib

# Save model and vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# spam_detector.py (updated for live use)

import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# Load pre-trained model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    stemmer = PorterStemmer()
    clean_words = [stemmer.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(clean_words)

def predict_spam(email):
    email_clean = preprocess_text(email)
    email_vector = vectorizer.transform([email_clean])
    prediction = model.predict(email_vector)[0]
    return "Spam" if prediction == 1 else "Not Spam"
