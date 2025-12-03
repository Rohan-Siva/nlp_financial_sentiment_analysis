import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys
import os
from tqdm import tqdm

                              
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.preprocess import preprocess_pipeline
from data.vocabulary import Vocabulary

class FinancialDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=50):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]
        
        tokens = preprocess_pipeline(text)
        token_ids = self.vocab.convert_tokens_to_ids(tokens)
        
                              
        if len(token_ids) < self.max_len:
            token_ids = token_ids + [0] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]
            
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                            bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
                                      
        embedded = self.dropout(self.embedding(text))
                                                   
        
        output, (hidden, cell) = self.lstm(embedded)
                                                                
        
                                                             
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
                                           
        
        return self.fc(hidden)

def train_lstm(data_path, epochs=5, batch_size=64, embedding_dim=100, hidden_dim=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    df = pd.read_csv(data_path)
    X = df['sentence']
    y = df['label']
    
                      
    print("Building vocabulary...")
    vocab = Vocabulary(min_freq=2)
    tokenized_texts = [preprocess_pipeline(text) for text in X]
    vocab.build_vocabulary(tokenized_texts)
    print(f"Vocabulary size: {len(vocab)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_dataset = FinancialDataset(X_train, y_train, vocab)
    test_dataset = FinancialDataset(X_test, y_test, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    model = SentimentLSTM(len(vocab), embedding_dim, hidden_dim, 3, 2, 0.5).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for texts, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            texts, labels = texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(texts)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")
        
                
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            predictions = model(texts)
            preds = torch.argmax(predictions, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print("\nLSTM Model Performance:")
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(classification_report(all_labels, all_preds))

if __name__ == "__main__":
    data_path = 'data/financial_phrasebank.csv'
    if os.path.exists(data_path):
        train_lstm(data_path)
    else:
        print("Data file not found.")
