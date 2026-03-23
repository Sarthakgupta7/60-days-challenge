from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


app = FastAPI(title="AI Prediction API")


texts = [
    "Win a free iPhone now",
    "Limited time offer",
    "Hello how are you",
    "Let's meet tomorrow",
    "Claim your reward now",
    "Are you available for meeting"
]

labels = [1, 1, 0, 0, 1, 0]  

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

class TextInput(BaseModel):
    message: str


@app.get("/")
def home():
    return {"message": "AI Prediction API is running 🚀"}


@app.post("/predict")
def predict(data: TextInput):

    text = data.message

    X_input = vectorizer.transform([text])

    prediction = model.predict(X_input)[0]

    result = "spam" if prediction == 1 else "ham"

    return {
        "input": text,
        "prediction": result
    }