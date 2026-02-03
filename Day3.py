import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return word_tokenize(text)
stemmer = PorterStemmer()

def apply_stemming(tokens):
    return [stemmer.stem(word) for word in tokens]
lemmatizer = WordNetLemmatizer()

reviews = [
    "The product is running very smoothly and works perfectly.",
    "I was disappointed with the quality of the product.",
    "The delivery was fast and the packaging was excellent."
]

def apply_lemmatization(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]
for review in reviews:
    tokens = clean_text(review)
    
    stemmed = apply_stemming(tokens)
    lemmatized = apply_lemmatization(tokens)

    print("\nOriginal Review:")
    print(review)

    print("Stemmed Output:")
    print(stemmed)

    print("Lemmatized Output:")
    print(lemmatized)
