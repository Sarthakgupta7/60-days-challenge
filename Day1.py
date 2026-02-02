import re
from collections import Counter

def tokenize(text):
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())

def remove_stopwords(tokens):
    stopwords = {
        "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
        "in", "on", "at", "to", "for", "of", "with", "by", "as", "from"
    }
    return [word for word in tokens if word not in stopwords]

def word_frequency(tokens):
    return Counter(tokens)
  
print("enter the sentence")
text = input().strip()

tokens = tokenize(text)
filtered_tokens = remove_stopwords(tokens)
frequency = word_frequency(filtered_tokens)

print("Tokens:", tokens)
print("After removing stopwords:", filtered_tokens)
print("Word frequency:", frequency)
