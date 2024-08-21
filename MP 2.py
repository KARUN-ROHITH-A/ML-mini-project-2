import pandas as pd
import numpy as np 


df = pd.read_csv("C:\\Users\\karun\\Downloads\\Corona_NLP_test.csv")


print(df.head())


print(df.isnull().sum())


print(df['Sentiment'].value_counts())

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()  
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  
    text = re.sub(r'\@w+|\#','', text)  
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['CleanedTweet'] = df['OriginalTweet'].apply(preprocess_text)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df['SentimentEncoded'] = label_encoder.fit_transform(df['Sentiment'])


print(label_encoder.classes_)

from sklearn.model_selection import train_test_split

X = df['CleanedTweet']
y = df['SentimentEncoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

from gensim.models import Word2Vec

X_train_tokens = X_train.apply(lambda x: x.split())
X_test_tokens = X_test.apply(lambda x: x.split())

w2v_model = Word2Vec(sentences=X_train_tokens, vector_size=100, window=5, min_count=5, workers=4)

def get_average_word2vec(tokens_list, model, vector_size):
    vec = np.zeros(vector_size).reshape((1, vector_size))
    count = 0
    for word in tokens_list:
        try:
            vec += model.wv[word].reshape((1, vector_size))
            count += 1
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

X_train_w2v = np.concatenate([get_average_word2vec(tokens, w2v_model, 100) for tokens in X_train_tokens])
X_test_w2v = np.concatenate([get_average_word2vec(tokens, w2v_model, 100) for tokens in X_test_tokens])

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

lr_tfidf = LogisticRegression(max_iter=100)
lr_tfidf.fit(X_train_tfidf, y_train)

y_pred_tfidf = lr_tfidf.predict(X_test_tfidf)
print("TF-IDF Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_tfidf))
print(classification_report(y_test, y_pred_tfidf, target_names=label_encoder.classes_, zero_division=0))

lr_w2v = LogisticRegression(max_iter=100)
lr_w2v.fit(X_train_w2v, y_train)

y_pred_w2v = lr_w2v.predict(X_test_w2v)
print("Word2Vec Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_w2v))
print(classification_report(y_test, y_pred_w2v, target_names=label_encoder.classes_, zero_division=0))

results = {
    "TF-IDF": accuracy_score(y_test, y_pred_tfidf),
    "Word2Vec": accuracy_score(y_test, y_pred_w2v)
}

print("Comparison of Model Accuracies:")
for key, value in results.items():
    print(f"{key}: {value:.4f}")



