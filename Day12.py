import re
import string
from typing import List, Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
class NLPPipeline:
    def __init__(
        self,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        stem: bool = False,
        vectorize: bool = False
    ):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stem = stem
        self.vectorize = vectorize

        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

        self.vectorizer = TfidfVectorizer() if vectorize else None

    # ---------------------------
    # Core Cleaning Steps
    # ---------------------------

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"\d+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        return word_tokenize(text)

    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t not in self.stop_words]

    def apply_lemmatization(self, tokens: List[str]) -> List[str]:
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    def apply_stemming(self, tokens: List[str]) -> List[str]:
        return [self.stemmer.stem(t) for t in tokens]

    # ---------------------------
    # Main Pipeline Function
    # ---------------------------

    def preprocess(self, text: str) -> str:
        if not text or not text.strip():
            return ""

        text = self.clean_text(text)
        tokens = self.tokenize(text)

        if self.remove_stopwords:
            tokens = self.remove_stop_words(tokens)

        if self.lemmatize:
            tokens = self.apply_lemmatization(tokens)

        if self.stem:
            tokens = self.apply_stemming(tokens)

        return " ".join(tokens)

    # ---------------------------
    # Optional Vectorization
    # ---------------------------

    def transform(self, texts: List[str]):
        processed = [self.preprocess(t) for t in texts]

        if self.vectorize:
            return self.vectorizer.fit_transform(processed)

        return processed
if __name__ == "__main__":
    documents = [
        "Natural Language Processing is amazing!",
        "NLP pipelines should be reusable and modular."
    ]

    pipeline = NLPPipeline(
        remove_stopwords=True,
        lemmatize=True,
        vectorize=True
    )

    vectors = pipeline.transform(documents)

    print("TF-IDF Shape:", vectors.shape)
