from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

app = FastAPI(title="NLP Model Serving API")

texts = [
    "Win a free iPhone now",
    "Limited time offer",
    "Claim your reward now",
    "Hello how are you",
    "Let's meet tomorrow",
    "Are you available for meeting"
]

labels = [1, 1, 1, 0, 0, 0]  # 1 = spam, 0 = ham

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

class InputText(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "NLP Model API is running 🚀"}

@app.post("/predict")
def predict(data: InputText):

    text = data.text
    X_input = vectorizer.transform([text])
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0]

    confidence = np.max(prob)

    result = "spam" if pred == 1 else "ham"

    return {
        "input_text": text,
        "prediction": result,
        "confidence": round(float(confidence), 3)
    }