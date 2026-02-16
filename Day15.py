import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


documents = [
    "Natural language processing enables machines to understand text.",
    "Machine learning models rely on numerical representations.",
    "Text embeddings convert words into vectors.",
    "Deep learning improves NLP performance."
]


vectorizer = TfidfVectorizer(stop_words='english')
embedding_matrix = vectorizer.fit_transform(documents)

embedding_matrix = embedding_matrix.toarray()

print("Embedding Matrix Shape:", embedding_matrix.shape)
dot_product_matrix = np.dot(embedding_matrix, embedding_matrix.T)

print("\nDot Product Matrix Shape:", dot_product_matrix.shape)
print(dot_product_matrix)

normalized_matrix = normalize(embedding_matrix)


cosine_similarity_matrix = np.dot(normalized_matrix, normalized_matrix.T)

print("\nCosine Similarity Matrix Shape:", cosine_similarity_matrix.shape)
print(cosine_similarity_matrix)
mean_embedding = np.mean(embedding_matrix, axis=0)

print("\nMean Embedding Shape:", mean_embedding.shape)
