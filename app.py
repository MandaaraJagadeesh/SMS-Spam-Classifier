import streamlit as st
import pandas as pd
import joblib
import os
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Download NLTK resources
nltk.download('stopwords')

# Function to train the model
def train_model():
    # Load the dataset with explicit encoding
    df = pd.read_csv('spam.csv', encoding='latin1')

    # Drop unnecessary columns if present
    df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')

    # Rename columns to a standard format
    df.columns = ['label', 'message']

    # Encode the labels
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # Text preprocessing function
    def preprocess_text(text):
        ps = PorterStemmer()
        stop_words = set(stopwords.words('english'))
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        words = [ps.stem(word) for word in words if word.lower() not in stop_words]
        return ' '.join(words)

    # Apply text preprocessing
    df['message'] = df['message'].apply(preprocess_text)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

    # Create a pipeline that combines the CountVectorizer and SVM
    model = make_pipeline(CountVectorizer(), SVC(probability=True))

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(classification_report(y_test, y_pred))

    # Save the model
    joblib.dump(model, 'sms_spam_classifier.pkl')
    print("Model saved successfully.")

# Check if the model file exists, if not train the model
if not os.path.exists('sms_spam_classifier.pkl'):
    train_model()

# Load the trained model
model = joblib.load('sms_spam_classifier.pkl')

# Text preprocessing function
def preprocess_text(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [ps.stem(word) for word in words if word.lower() not in stop_words]
    return ' '.join(words)

# Streamlit app
def main():
    st.title('SMS Spam Classifier')
    st.write("Enter an SMS message to check if it's spam or not.")

    # Input field for SMS message
    sms_message = st.text_area('SMS Message')

    # Predict button
    if st.button('Predict'):
        # Preprocess the input message
        preprocessed_message = preprocess_text(sms_message)

        # Predict using the model
        prediction = model.predict([preprocessed_message])[0]
        prediction_proba = model.predict_proba([preprocessed_message])[0]

        # Display the result
        if prediction == 1:
            st.error(f'This message is classified as Spam with a probability of {prediction_proba[1]:.2f}')
        else:
            st.success(f'This message is classified as Not Spam with a probability of {prediction_proba[0]:.2f}')

if __name__ == '__main__':
    main()
