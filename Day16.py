import numpy as np

# Example sentence embeddings (4-dimensional)
v1 = np.array([0.2, 0.5, 0.1, 0.7])
v2 = np.array([0.3, 0.4, 0.2, 0.6])
v3 = np.array([0.9, 0.1, 0.3, 0.2])

print("Vector Shape:", v1.shape)
dot_product = np.dot(v1, v2)
print("Dot Product (v1 · v2):", dot_product)
cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
print("Cosine Similarity:", cos_sim)
embedding_matrix = np.vstack([v1, v2, v3])

print("Embedding Matrix Shape:", embedding_matrix.shape)
print(embedding_matrix)
similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T)

print("Similarity Matrix Shape:", similarity_matrix.shape)
print(similarity_matrix)
