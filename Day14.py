import re
import string
from collections import Counter
from typing import List

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# Sample Text
text = """
Natural language processing is amazing. NLP helps machines understand human language!
It is widely used in search engines, chatbots, and spam detection systems.
"""


# ---------------------------
# Sentence Tokenization
# ---------------------------
sentences = sent_tokenize(text)

print("===== Sentence Tokenization =====")
for s in sentences:
    print(s)

# ---------------------------
# Word Tokenization
# ---------------------------
words = word_tokenize(text)

print("\n===== Word Tokenization =====")
print(words)

# ---------------------------
# Remove Punctuation
# ---------------------------
words_no_punct = [w for w in words if w not in string.punctuation]

# ---------------------------
# Remove Stopwords
# ---------------------------
stop_words = set(stopwords.words("english"))
cleaned_tokens = [w.lower() for w in words_no_punct if w.lower() not in stop_words]

print("\n===== Cleaned Tokens =====")
print(cleaned_tokens)
# ---------------------------
# Word Frequency Distribution
# ---------------------------
freq_dist = Counter(cleaned_tokens)

print("\n===== Word Frequency =====")
print(freq_dist)

# Top 20 words
print("\n===== Top 20 Words =====")
print(freq_dist.most_common(20))


documents = sentences  # Use sentence list as documents


# ---------------------------
# Bag-of-Words
# ---------------------------
bow_vectorizer = CountVectorizer(stop_words='english')
bow_matrix = bow_vectorizer.fit_transform(documents)

print("\n===== Bag-of-Words Vocabulary =====")
print(bow_vectorizer.get_feature_names_out())

print("\n===== BoW Matrix =====")
print(bow_matrix.toarray())


# ---------------------------
# TF-IDF
# ---------------------------
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print("\n===== TF-IDF Vocabulary =====")
print(tfidf_vectorizer.get_feature_names_out())

print("\n===== TF-IDF Matrix =====")
print(tfidf_matrix.toarray())


# ---------------------------
# Comparison Insight
# ---------------------------
print("\n===== Comparison =====")
print("BoW counts frequency.")
print("TF-IDF weighs importance by reducing common words impact.")
class NLPPipeline:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"\d+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        return word_tokenize(text)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t not in self.stop_words]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    def preprocess(self, text: str) -> str:
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return " ".join(tokens)
