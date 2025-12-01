import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import sys
import os

# Add parent directory to path to import data modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.preprocess import clean_text

def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found. Please run data/download_data.py first.")
        sys.exit(1)
    return pd.read_csv(filepath)

def train_baseline(data_path, model_type='lr'):
    print(f"Training {model_type.upper()} baseline...")
    df = load_data(data_path)
    
    df['cleaned_text'] = df['sentence'].apply(clean_text)
    
    X = df['cleaned_text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == 'lr':
        clf = LogisticRegression(max_iter=1000)
    elif model_type == 'svm':
        clf = SVC(kernel='linear')
    else:
        raise ValueError("Invalid model_type. Choose 'lr' or 'svm'.")
        
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', clf)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return pipeline

if __name__ == "__main__":
    data_path = 'data/financial_phrasebank.csv'
    train_baseline(data_path, model_type='lr')
