import os
from datasets import load_dataset
import pandas as pd

def download_data(output_dir='data'):
    print("Downloading Financial PhraseBank dataset...")
    dataset = load_dataset("financial_phrasebank", "sentences_allagree")
    
    df = pd.DataFrame(dataset['train'])
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'financial_phrasebank.csv')
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(f"Dataset shape: {df.shape}")
    print(df.head())

if __name__ == "__main__":
    download_data()
