import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


data = {
    "text": [
        "I absolutely loved this movie",
        "The product quality is amazing",
        "This service is terrible",
        "I am very disappointed",
        "It was an average experience",
        "The staff was extremely helpful",
        "Worst purchase ever",
        "I am happy with the results",
        "Not worth the money",
        "Highly recommend this product",
        "It was okay, nothing special",
        "Very bad experience overall"
    ],
    "sentiment": [
        "positive","positive","negative","negative","neutral",
        "positive","negative","positive","negative","positive",
        "neutral","negative"
    ]
}

df = pd.DataFrame(data)

# Encode labels
df["sentiment"] = df["sentiment"].map({
    "negative": 0,
    "neutral": 1,
    "positive": 2
})


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

df["clean_text"] = df["text"].apply(clean_text)


vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["clean_text"])
y = df["sentiment"]

print("TF-IDF Shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall:", recall_score(y_test, y_pred, average="weighted"))
print("F1-score:", f1_score(y_test, y_pred, average="weighted"))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()