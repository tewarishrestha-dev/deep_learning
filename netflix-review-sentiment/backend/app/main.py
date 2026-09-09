from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
from pathlib import Path

# Add ml/src to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_SRC = PROJECT_ROOT / "ml" / "src"

sys.path.append(str(ML_SRC))

from predict import predict_sentiment


app = FastAPI(
    title="Netflix Review Sentiment API",
    description="LSTM-powered movie review sentiment analysis API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReviewRequest(BaseModel):
    review: str


@app.get("/")
def root():
    return {
        "message": "Netflix Review Sentiment API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(request: ReviewRequest):

    result = predict_sentiment(
        request.review
    )

    return result