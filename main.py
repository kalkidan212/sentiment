from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import os

app = FastAPI(title="Sentiment Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JSONL ፋይልን ማንበብ
DATASET_PATH = "sentiment_analysis (1).jsonl"
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"{DATASET_PATH} not found in backend directory.")

data = []
with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

df = pd.DataFrame(data)

text_col = next((col for col in ['text', 'comment', 'review', 'content'] if col in df.columns), df.columns[0])
sentiment_col = next((col for col in ['sentiment', 'label', 'class', 'target'] if col in df.columns), df.columns[1])

X = df[text_col]
y = df[sentiment_col]

# Train NLP Pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=200))
])
model.fit(X, y)

# Databases
users_db = {}
history_db = []

class AuthRequest(BaseModel):
    email: str
    password: str

class PredictRequest(BaseModel):
    email: str
    category: str
    text: str

@app.post("/api/signup")
def signup(req: AuthRequest):
    if req.email in users_db:
        raise HTTPException(status_code=400, detail="Email already registered!")
    users_db[req.email] = req.password
    return {"message": "Signup successful"}

@app.post("/api/login")
def login(req: AuthRequest):
    if req.email not in users_db:
        raise HTTPException(status_code=404, detail="Email not found. Please Sign Up first!")
    if users_db[req.email] != req.password:
        raise HTTPException(status_code=401, detail="Incorrect password!")
    return {"message": "Login successful"}

@app.post("/api/predict")
def predict_sentiment(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    prediction = model.predict([req.text])[0]
    
    record = {
        "email": req.email,
        "category": req.category,
        "text": req.text,
        "sentiment": prediction
    }
    history_db.append(record)
    
    return {"sentiment": prediction, "record": record}

@app.get("/api/history/{email}")
def get_history(email: str):
    user_history = [h for h in history_db if h["email"] == email]
    return {"history": user_history}