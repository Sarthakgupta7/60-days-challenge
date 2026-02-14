import re
import string
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


class SpamPreprocessingPipeline:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()
        self.vectorizer = TfidfVectorizer()

    # Cleaning
    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"\d+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # Tokenization
    def tokenize(self, text: str) -> List[str]:
        return word_tokenize(text)

    # Stopword removal
    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t not in self.stop_words]

    # Stemming
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        return [self.stemmer.stem(t) for t in tokens]

    # Full preprocessing
    def preprocess(self, text: str) -> str:
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stop_words(tokens)
        tokens = self.stem_tokens(tokens)
        return " ".join(tokens)


# ===============================
# Demo
# ===============================

if __name__ == "__main__":

    messages = [
        "WIN a FREE iPhone now!!! Click here: www.spamlink.com",
        "Hey, are we still meeting tomorrow?",
        "Congratulations! You have won $1000. Claim now!!!"
    ]

    pipeline = SpamPreprocessingPipeline()

    print("========== BEFORE PREPROCESSING ==========\n")
    for msg in messages:
        print(msg)

    print("\n========== AFTER PREPROCESSING ==========\n")
    processed_messages = [pipeline.preprocess(msg) for msg in messages]

    for msg in processed_messages:
        print(msg)

    # Vectorization
    tfidf_matrix = pipeline.vectorizer.fit_transform(processed_messages)

    print("\n========== TF-IDF VECTOR DETAILS ==========")
    print("Shape:", tfidf_matrix.shape)

    print("\nVocabulary:")
    print(pipeline.vectorizer.get_feature_names_out())

    print("\nTF-IDF Matrix:")
    print(tfidf_matrix.toarray())
