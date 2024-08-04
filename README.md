# SMS-Spam-Classifier
A machine learning application for classifying SMS messages as spam or not spam using Streamlit for the user interface. The classifier is built with Scikit-learn and natural language processing (NLP) techniques.

Features:

  Text Preprocessing: Cleans and preprocesses SMS text using stemming and stopword removal.
  Model Training: Trains a Support Vector Machine (SVM) classifier using a dataset of SMS messages.
  Prediction: Classifies new SMS messages as spam or not spam with probability estimates.
  Streamlit App: Provides an interactive web interface to input SMS messages and get predictions.

Installation Prerequisites:

  Python 3.x
  pip

Project Structure:

  app.py: The Streamlit application script.
  train_model.py: Script for training and saving the machine learning model.
  spam.csv: The dataset used for training the model (should be in the project directory).
  sms_spam_classifier.pkl: Saved model file (created after training).
  requirements.txt: List of required Python packages.



