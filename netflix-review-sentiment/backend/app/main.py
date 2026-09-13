
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io

import sys
from pathlib import Path

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
        "https://deep-learning-1-uhb0.onrender.com"
    ],
    allow_credentials=False,
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

    if not request.review.strip():
        raise HTTPException(
            status_code=400,
            detail="Review cannot be empty."
        )

    return predict_sentiment(request.review)


@app.post("/batch-predict")
async def batch_predict(
    file: UploadFile = File(...)
):

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported."
        )

    contents = await file.read()

    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read the CSV file."
        )

    if "review" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="CSV must contain a 'review' column."
        )

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="CSV file is empty."
        )

    if len(df) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Maximum 1000 reviews allowed per upload."
        )

    results = []

    for review in df["review"]:

        if pd.isna(review) or not str(review).strip():

            results.append({
                "review": "",
                "sentiment": "unknown",
                "confidence": 0.0,
                "positive_probability": 0.0,
                "negative_probability": 0.0
            })

            continue

        result = predict_sentiment(str(review))

        results.append({
            "review": str(review),
            **result
        })

    positive_count = sum(
        result["sentiment"] == "positive"
        for result in results
    )

    negative_count = sum(
        result["sentiment"] == "negative"
        for result in results
    )

    valid_results = [
        result
        for result in results
        if result["sentiment"] != "unknown"
    ]

    average_confidence = (
        sum(result["confidence"] for result in valid_results)
        / len(valid_results)
        if valid_results
        else 0.0
    )

    return {
        "total_reviews": len(results),
        "positive_reviews": positive_count,
        "negative_reviews": negative_count,
        "positive_percentage": (
            positive_count / len(results) * 100
            if results else 0
        ),
        "negative_percentage": (
            negative_count / len(results) * 100
            if results else 0
        ),
        "average_confidence": average_confidence,
        "results": results
    }

