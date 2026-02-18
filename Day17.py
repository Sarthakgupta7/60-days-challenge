import re
import numpy as np
from collections import Counter
from scipy.stats import poisson, binom
text = """
Natural language processing enables machines to understand language.
Language models process text and generate language.
Probability helps us model word frequency in language data.
"""
# Clean + tokenize
tokens = re.findall(r'\b[a-z]+\b', text.lower())

print("Tokens:")
print(tokens)
freq_dist = Counter(tokens)

print("\nWord Frequency:")
print(freq_dist)
total_words = len(tokens)

word_probabilities = {
    word: count / total_words
    for word, count in freq_dist.items()
}

print("\nWord Probabilities:")
print(word_probabilities)
lambda_value = np.mean(list(freq_dist.values()))

print("\nLambda (Mean Frequency):", lambda_value)

# Probability of a word appearing 3 times
prob_poisson = poisson.pmf(3, lambda_value)

print("Poisson P(X=3):", prob_poisson)
n = total_words
p = word_probabilities["language"]

# Probability language appears exactly 3 times
prob_binomial = binom.pmf(3, n, p)

print("\nBinomial P(X=3) for 'language':", prob_binomial)
