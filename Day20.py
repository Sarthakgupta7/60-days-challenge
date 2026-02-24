import numpy as np


class EmbeddingSimilarityCalculator:

    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon

    def _validate_vectors(self, v1, v2):
        if v1.shape != v2.shape:
            raise ValueError("Vectors must have the same shape.")
        if v1.ndim != 1:
            raise ValueError("Vectors must be 1-dimensional.")


    def _normalize(self, v):
        norm = np.linalg.norm(v)
        return v / (norm + self.epsilon)

    def cosine_similarity(self, v1, v2, normalize=True):
        self._validate_vectors(v1, v2)

        if normalize:
            v1 = self._normalize(v1)
            v2 = self._normalize(v2)

        return np.dot(v1, v2)

    # -----------------------------
    # Dot Product
    # -----------------------------
    def dot_product(self, v1, v2):
        self._validate_vectors(v1, v2)
        return np.dot(v1, v2)

    # -----------------------------
    # Euclidean Distance
    # -----------------------------
    def euclidean_distance(self, v1, v2):
        self._validate_vectors(v1, v2)
        return np.linalg.norm(v1 - v2)


# -----------------------------
# Example Usage
# -----------------------------

if __name__ == "__main__":

    # Example 5-dimensional embeddings
    v1 = np.array([0.2, 0.5, 0.1, 0.7, 0.3])
    v2 = np.array([0.3, 0.4, 0.2, 0.6, 0.2])

    calculator = EmbeddingSimilarityCalculator()

    cos_sim = calculator.cosine_similarity(v1, v2)
    dot = calculator.dot_product(v1, v2)
    euclid = calculator.euclidean_distance(v1, v2)

    print("Cosine Similarity:", cos_sim)
    print("Dot Product:", dot)
    print("Euclidean Distance:", euclid)
