import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import os
import sys

                              
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.preprocess import clean_text

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_transformer(data_path, model_name='distilbert-base-uncased', epochs=3, batch_size=16):
    print(f"Fine-tuning {model_name}...")
    
    df = pd.read_csv(data_path)
                                
    df['label'] = df['label'].astype(int)
    
                
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
                    
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples['sentence'], padding="max_length", truncation=True)
    
                           
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
                
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    print("Starting training...")
    trainer.train()
    
    print("\nEvaluating...")
    eval_result = trainer.evaluate()
    print(f"Evaluation results: {eval_result}")
    
                
    model_path = "models/saved_transformer"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    data_path = 'data/financial_phrasebank.csv'
    if os.path.exists(data_path):
        train_transformer(data_path)
    else:
        print("Data file not found.")
