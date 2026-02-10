documents = [
    "Natural language processing is interesting",
    "Machine learning models learn from data",
    "Natural language processing and machine learning are related",
    "Text data needs preprocessing before modeling"
]
from sklearn.feature_extraction.text import CountVectorizer

# Initialize BoW vectorizer
bow_vectorizer = CountVectorizer(stop_words='english')


bow_matrix = bow_vectorizer.fit_transform(documents)


print("BoW Vocabulary:")
print(bow_vectorizer.get_feature_names_out())


print("\nBoW Feature Matrix:")
print(bow_matrix.toarray())
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print("\nTF-IDF Vocabulary:")
print(tfidf_vectorizer.get_feature_names_out())
print("\nTF-IDF Feature Matrix:")
print(tfidf_matrix.toarray())
