import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('C:/Users/lgspa/Downloads/IMDB Dataset.csv')

# Convert sentiment labels to numeric
label_encoder = LabelEncoder()
df['sentiment'] = label_encoder.fit_transform(df['sentiment'])

# Split the dataset into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42
)

# Convert the text data into a bag-of-words representation using CountVectorizer
vectorizer = CountVectorizer()
train_data_vectorized = vectorizer.fit_transform(train_data)
test_data_vectorized = vectorizer.transform(test_data)

# Function to train and evaluate a classifier
def train_and_evaluate_classifier(classifier, train_data, test_data, train_labels, test_labels):
    classifier.fit(train_data, train_labels)
    predictions = classifier.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions)
    return accuracy, report

# Function to predict sentiment based on user input
def predict_sentiment(review):
    review_vectorized = vectorizer.transform([review])
    prediction = nb_classifier.predict(review_vectorized)
    return prediction[0]

# Title
st.title('Movie Review Sentiment Analysis')

# Sidebar
user_review = st.text_area('Enter a movie review:')
submitted = st.button('Submit')

if submitted:
    # Naive Bayes Classifier
    nb_classifier = MultinomialNB()
    nb_accuracy, nb_report = train_and_evaluate_classifier(nb_classifier, train_data_vectorized, test_data_vectorized, train_labels, test_labels)

    # Predict sentiment
    prediction = predict_sentiment(user_review)

    # Display result
    if prediction == 1:
        st.write("The sentiment is POSITIVE.")
    else:
        st.write("The sentiment is NEGATIVE.")
