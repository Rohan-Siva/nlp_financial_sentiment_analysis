# Financial Sentiment Analysis

## Overview
This project implements a comprehensive Natural Language Processing (NLP) pipeline for Financial Sentiment Analysis. It utilizes the Financial PhraseBank dataset to train and evaluate various models, ranging from traditional machine learning baselines to state-of-the-art deep learning architectures. The goal is to accurately classify financial texts into positive, negative, or neutral sentiments.

## Tools Used
- **Deep Learning**: PyTorch, Hugging Face Transformers
- **Machine Learning**: Scikit-learn
- **Data Processing**: Pandas, NumPy, NLTK, Datasets
- **Visualization**: Matplotlib, Seaborn

## Structure
- `data/`: Contains scripts for downloading and preprocessing the Financial PhraseBank dataset.
- `models/`: Includes implementations of three distinct model architectures:
  - **Baseline**: TF-IDF vectorization with Logistic Regression.
  - **LSTM**: Long Short-Term Memory network with word embeddings.
  - **Transformer**: Fine-tuned DistilBERT model for high-performance classification.
- `evaluation/`: Modules for calculating performance metrics and performing error analysis.
- `notebooks/`: Jupyter notebooks for data exploration, model training, and comparative analysis.

## Setup

### Prerequisites
- Python 3.8+

### Installation
1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Data Preparation
Download and preprocess the dataset:
```bash
python data/download_data.py
```

## How It Works
1. **Data Loading**: The `data` scripts load the Financial PhraseBank dataset and prepare it for training.
2. **Model Training**: You can train different models using the scripts in `models/` or explore the process in `notebooks/`.
   - The baseline model establishes a performance benchmark.
   - The LSTM model captures sequential dependencies in the text.
   - The Transformer model (DistilBERT) leverages pre-trained contextual embeddings for superior accuracy.
3. **Evaluation**: The `evaluation` module provides detailed metrics (accuracy, F1-score, confusion matrix) to compare model performance.

## Contact
For collaborations or questions, please reach out to rohansiva123@gmail.com.
