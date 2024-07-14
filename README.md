# Spam-Classification-Using-NLP

Project Overview
This project involves building a spam classification model using Natural Language Processing (NLP) techniques. The primary objective is to classify emails or text messages as spam or not spam. The project includes data preprocessing, feature extraction, model training, evaluation, and deployment using a web application.

Features

Text Preprocessing: Removal of punctuation, tokenization, and removal of stopwords.

Feature Extraction: Using CountVectorizer to convert text data into numerical features.

Model Training: Training multiple machine learning models including Logistic Regression, Support Vector Classifier, Random Forest, and Gradient Boosting.

Hyperparameter Tuning: Using GridSearchCV to find the best parameters for each model.

Model Evaluation: Evaluating models using confusion matrix, classification report, and accuracy score.

Model Deployment: Deploying the best model using Streamlit for real-time spam detection

# **Installation**
Clone the repository


git clone https://github.com/yourusername/spam-classification-nlp.git

cd spam-classification-nlp

Create a virtual environment and activate it


python -m venv venv

source venv/bin/activate  # On Windows: venv\Scripts\activate

Install the required packages

Download NLTK stopwords


python -m nltk.downloader stopwords

Dataset
The dataset used for this project is a public dataset of spam and ham messages. It consists of labeled text data with columns for the message and its corresponding label (spam or ham).

Usage

Training the Model

To train the model, run the following script:


python train_model.py

This script performs the following tasks:

Loads and preprocesses the dataset.

Extracts features using CountVectorizer.

Splits the data into training and test sets.

Trains multiple models and performs hyperparameter tuning using GridSearchCV.

Evaluates the models and saves the best model and vectorizer using joblib.

Running the Streamlit App

To deploy the model and check for spam text using Streamlit, run:



streamlit run Spam_streamlit_main.py

Streamlit Application Features

Text Input: Enter text to check if it's spam or not.

Prediction: The app predicts and displays whether the entered text is spam or not.

Visualizations: The app includes visualizations like model accuracies and confusion matrix heatmaps.


Files

train_model.py: Script to train and evaluate models.

Spam_streamlit_main.py: Streamlit app script for real-time spam detection.

best_spam_detection_model.pkl: Saved best model.

vectorizer.pkl: Saved CountVectorizer.

requirements.txt: List of required packages.

# Results
The project achieved high accuracy in spam detection using Gradient Boosting as the best model. The model evaluation includes detailed metrics such as confusion matrix and classification report to ensure robust performance.
