import numpy as np
from scipy.optimize import minimize
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(42)

# Simulated embeddings
X = np.random.rand(5, 4)  # 5 samples, 4-dimensional embeddings
true_W = np.random.rand(4, 4)

# Create target embeddings using true transformation
Y = X @ true_W
def cost_function(W_flat, X, Y):
    W = W_flat.reshape(4, 4)
    prediction = X @ W
    error = prediction - Y
    return np.sum(error ** 2)
W_init = np.random.rand(4, 4)
W_init_flat = W_init.flatten()
result = minimize(
    cost_function,
    W_init_flat,
    args=(X, Y),
    method="BFGS"
)

W_optimized = result.x.reshape(4, 4)
initial_similarity = cosine_similarity(X @ W_init, Y).mean()
print("Initial Similarity:", initial_similarity)
optimized_similarity = cosine_similarity(X @ W_optimized, Y).mean()
print("Optimized Similarity:", optimized_similarity)
