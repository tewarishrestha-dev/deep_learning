import { useState } from "react";
import "./App.css";

function App() {
  const [review, setReview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const analyzeReview = async () => {
    if (!review.trim()) return;

    setLoading(true);
    setResult(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          review: review,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to analyze review");
      }

      const data = await response.json();

      setResult(data);
    } catch (error) {
      console.error(error);
      alert("Could not connect to the sentiment API.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <nav className="navbar">
        <div className="logo">
          <span className="logo-mark">N</span>
          <span>REVIEW AI</span>
        </div>

        <div className="nav-badge">
          LSTM SENTIMENT ENGINE
        </div>
      </nav>

      <main className="hero">
        <section className="hero-content">

          <p className="eyebrow">
            AI-POWERED MOVIE REVIEW ANALYSIS
          </p>

          <h1>
            What does your
            <span> review </span>
            really say?
          </h1>

          <p className="subtitle">
            Analyze the sentiment of any movie review using
            a trained LSTM neural network.
          </p>

          <div className="review-card">

            <textarea
              value={review}
              onChange={(e) => setReview(e.target.value)}
              placeholder="Write or paste a movie review..."
              rows="7"
            />

            <div className="input-footer">
              <span>
                {review.length} characters
              </span>

              <button
                onClick={analyzeReview}
                disabled={loading || !review.trim()}
              >
                {loading ? "ANALYZING..." : "ANALYZE REVIEW →"}
              </button>
            </div>

          </div>

          {result && (
            <div className="result-card">

              <div className="result-header">
                <span>AI ANALYSIS</span>

                <span className="confidence">
                  {(result.confidence * 100).toFixed(1)}% CONFIDENCE
                </span>
              </div>

              <div className="sentiment">
                {result.sentiment.toUpperCase()}
              </div>

              <div className="probability-section">

                <div className="probability-row">
                  <span>Positive</span>
                  <span>
                    {(result.positive_probability * 100).toFixed(1)}%
                  </span>
                </div>

                <div className="progress">
                  <div
                    className="progress-fill"
                    style={{
                      width: `${result.positive_probability * 100}%`,
                    }}
                  />
                </div>

                <div className="probability-row">
                  <span>Negative</span>
                  <span>
                    {(result.negative_probability * 100).toFixed(1)}%
                  </span>
                </div>

                <div className="progress">
                  <div
                    className="progress-fill negative"
                    style={{
                      width: `${result.negative_probability * 100}%`,
                    }}
                  />
                </div>

              </div>

            </div>
          )}

        </section>
      </main>

      <footer>
        Built with React · FastAPI · PyTorch · LSTM
      </footer>
    </div>
  );
}

export default App;