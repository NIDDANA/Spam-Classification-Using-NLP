# -*- coding: utf-8 -*-
"""Spam Classification using NLP.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rGQw-b66vHxh6iDBl6TNTsK4Cf64NUgH

## **Step-by-Step Implementation:**
Import Necessary Libraries

Load and Preprocess Data

Feature Extraction

Model Training

Model Evaluation

Saving the Model
"""



#importing the required libraries
import os
import nltk
import string
import joblib
import pandas as pd
from sklearn.svm import SVC
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#connecting drive to colab
from google.colab import drive
drive.mount('/content/drive')

#  ----------------------------- Download NLTK stopwords ---------------------------------

nltk.download('stopwords')

# loading the dataset
dataset = pd.read_csv("/content/drive/MyDrive/spam.csv", encoding='latin-1')
dataset.head()

dataset.info

#getting the columns
dataset.columns

# getting information of columns and dataset
dataset = dataset[['v1', 'v2']]
dataset.columns = ['label', 'text']



def process(text):

    # Remove punctuation
    nopunc = ''.join([char for char in text if char not in string.punctuation])
    # Tokenize and remove stopwords
    tokens = nopunc.split()
    clean_tokens = [word for word in tokens if word.lower() not in stopwords.words('english')]
    return clean_tokens

def load_data(file_path):

    dataset = pd.read_csv(file_path, encoding='latin-1')
    dataset = dataset[['v1', 'v2']]
    dataset.columns = ['label', 'text']

    vectorizer = CountVectorizer(analyzer=process)
    X = vectorizer.fit_transform(dataset['text'])
    y = pd.get_dummies(dataset['label'], drop_first=True)

    return X, y, vectorizer

def perform_grid_search(model, param_grid, X_train, y_train):
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train.values.ravel())
    return grid_search.best_estimator_

def train_and_evaluate_models(X_train, X_test, y_train, y_test):

    param_grid_lr = {'C': [0.1, 1, 10, 100]}
    param_grid_svc = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
    param_grid_rf = {'n_estimators': [100, 200, 500], 'max_features': ['auto', 'sqrt', 'log2']}
    param_grid_gb = {'n_estimators': [100, 200, 500], 'learning_rate': [0.01, 0.1, 0.5, 1]}

    best_lr = perform_grid_search(LogisticRegression(), param_grid_lr, X_train, y_train)
    best_svc = perform_grid_search(SVC(), param_grid_svc, X_train, y_train)
    best_rf = perform_grid_search(RandomForestClassifier(), param_grid_rf, X_train, y_train)
    best_gb = perform_grid_search(GradientBoostingClassifier(), param_grid_gb, X_train, y_train)

    models = {
        'Logistic Regression': best_lr,
        'Support Vector Classifier': best_svc,
        'Random Forest': best_rf,
        'Gradient Boosting': best_gb
    }

    for name, model in models.items():
        predictions = model.predict(X_test)
        print(f"Results for {name}:")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        print("\nAccuracy Score:")
        print(accuracy_score(y_test, predictions))
        print("\n" + "="*60 + "\n")

    return models

def save_model_and_vectorizer(model, vectorizer, model_path='best_spam_detection_model.pkl', vectorizer_path='vectorizer.pkl'):
    # Save the best model
    joblib.dump(model, model_path)
    print("Best model saved successfully.")
    joblib.dump(vectorizer, vectorizer_path)
    print("Vectorizer saved successfully.")

def main():
    # Load and preprocess the data
    X, y, vectorizer = load_data("/content/drive/MyDrive/spam.csv")

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    models = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Save the best model
    best_model = models['Gradient Boosting']
    save_model_and_vectorizer(best_model, vectorizer)

# main
if __name__ == "__main__":
    main()



"""# Explanation:
Import Necessary Libraries: Import all required libraries and download NLTK stopwords.
## Load the Dataset:
Load your dataset. In this example, we're assuming the dataset is a CSV file with columns v1 (labels) and v2 (texts).
## Text Preprocessing:
 Define a process function to remove punctuation, tokenize, and remove stopwords from the text.
## Feature Extraction:
 Use CountVectorizer with the custom analyzer to transform the text data into a matrix of token counts.
## Encode Labels:
Encode the labels into binary values.
## Train-Test Split: **
split the dataset into training and testing sets.
## **Train the Model: **
Train a Naive Bayes classifier on the training data
## **Make Predictions:**
 Predict the labels for the test data.
## **Evaluate the Model:**
 Evaluate the model using confusion matrix, classification report, and accuracy score.
## **Save the Model:**
 Save the trained model and vectorizer using joblib.
"""

























