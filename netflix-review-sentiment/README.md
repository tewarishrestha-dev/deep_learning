# Netflix Review Sentiment Analysis

A movie review sentiment analysis app built using **PyTorch, LSTM, FastAPI, and React**.

The app takes a movie review and predicts whether it is **Positive** or **Negative**, along with a confidence score.

## Features

- Positive / Negative sentiment prediction
- Confidence score
- Token visualization
- Prediction history
- Batch review analysis
- React frontend
- FastAPI backend
- PyTorch LSTM model
- Deployed online

## Tech Stack

- Python
- PyTorch
- LSTM
- FastAPI
- React
- Vite
- JavaScript
- Render

## Project Structure

netflix-review-sentiment/
├── backend/
├── frontend/
├── ml/
├── models/
└── README.md

## Live Demo

https://deep-learning-1-uhb0.onrender.com/

## Example

Review:

> This movie was amazing. I loved every minute of it.

Prediction:

Positive  
Confidence: 98%

## How It Works

1. User enters a movie review.
2. The review is sent to the FastAPI backend.
3. The text is tokenized and processed.
4. The trained LSTM model predicts the sentiment.
5. The result and confidence score are returned to the React frontend.

## Note

The dataset and unnecessary large files are not included in the repository.

Built as a project to learn **Deep Learning, NLP, model deployment, and full-stack AI applications**.