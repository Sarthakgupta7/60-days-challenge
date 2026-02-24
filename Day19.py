import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(42)

# Simulate 5 embeddings of dimension 6
embeddings = np.random.randn(5, 6)

# Add a near-zero vector to test stability
embeddings[0] = np.zeros(6)

print("Embedding Matrix Shape:", embeddings.shape)
print(embeddings)
similarity_before = embeddings @ embeddings.T

print("\nDot Product Similarity (Before Normalization):")
print(similarity_before)
def l2_normalize(vectors, epsilon=1e-10):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + epsilon)

normalized_embeddings = l2_normalize(embeddings)

print("\nNormalized Embeddings:")
print(normalized_embeddings)
similarity_after = normalized_embeddings @ normalized_embeddings.T

print("\nCosine Similarity (After Normalization):")
print(similarity_after)