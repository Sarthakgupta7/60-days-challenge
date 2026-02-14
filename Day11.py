import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text
doc1 = "Natural language processing enables machines to understand human language."
doc2 = "Machines use natural language processing techniques to understand text."
doc1_clean = preprocess(doc1)
doc2_clean = preprocess(doc2)

documents = [doc1_clean, doc2_clean]
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(documents)

print("BoW Vectors:")
print(bow_matrix.toarray())

similarity_bow = cosine_similarity(bow_matrix)[0][1]

print("\nCosine Similarity (BoW):", similarity_bow)
