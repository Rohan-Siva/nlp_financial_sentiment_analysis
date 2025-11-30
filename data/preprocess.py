import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def clean_text(text):

    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_text(text):

    return word_tokenize(text)

def remove_stopwords(tokens):

    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

def preprocess_pipeline(text, remove_stop=True):

    text = clean_text(text)
    tokens = tokenize_text(text)
    if remove_stop:
        tokens = remove_stopwords(tokens)
    return tokens
