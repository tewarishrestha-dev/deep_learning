import { useState } from "react";
import "./App.css";

import AIProcessVisualizer from "./components/AIProcessVisualizer";
import BatchAnalyzer from "./components/BatchAnalyzer";
import ModelInfo from "./components/ModelInfo";
import PredictionHistory from "./components/PredictionHistory";
const API_URL = "https://netflix-sentiment-api.onrender.com";


const examples = [
  "This movie was absolutely fantastic. I loved every minute of it.",
  "One of the worst movies I have ever watched. Completely disappointing.",
  "The acting was brilliant and the story was incredibly engaging.",
];

function App() {
  const [review, setReview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const analyzeReview = async () => {
    if (!review.trim()) return;

    setLoading(true);
    setResult(null);
    setError("");

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          review: review.trim(),
        }),
      });

      if (!response.ok) {
        throw new Error("API request failed");
      }

      const data = await response.json();

      setResult(data);
    } catch (err) {
      console.error(err);

      setError(
        "Unable to connect to the AI engine. Make sure the FastAPI server is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (event) => {
    if (
      (event.ctrlKey || event.metaKey) &&
      event.key === "Enter"
    ) {
      analyzeReview();
    }
  };

  const selectExample = (text) => {
    setReview(text);
    setResult(null);
    setError("");
  };

  const clearReview = () => {
    setReview("");
    setResult(null);
    setError("");
  };

  const positive =
    result?.positive_probability ?? 0;

  const negative =
    result?.negative_probability ?? 0;

  return (
    <div className="app">

      {/* BACKGROUND EFFECTS */}

      <div className="background-glow glow-one" />
      <div className="background-glow glow-two" />


      {/* =========================
          NAVBAR
      ========================== */}

      <nav className="navbar">

        <div className="brand">

          <div className="brand-mark">
            N
          </div>

          <div>
            <div className="brand-name">
              REVIEW AI
            </div>

            <div className="brand-subtitle">
              SENTIMENT ENGINE
            </div>
          </div>

        </div>


        <div className="nav-status">

          <span className="status-dot" />

          LSTM ENGINE ONLINE

        </div>

      </nav>


      {/* =========================
          MAIN
      ========================== */}

      <main className="main">


        {/* =========================
            HERO
        ========================== */}

        <section className="hero">

          <div className="eyebrow">

            <span>✦</span>

            AI-POWERED REVIEW ANALYSIS

          </div>


          <h1>

            Understand what your

            <span> review </span>

            really says.

          </h1>


          <p className="hero-description">

            Our LSTM neural network analyzes the language
            in your movie review and estimates whether the
            overall sentiment is positive or negative.

          </p>

        </section>


        {/* =========================
            SINGLE REVIEW ANALYZER
        ========================== */}

        <section className="analyzer-grid">


          {/* =========================
              INPUT PANEL
          ========================== */}

          <div className="input-panel">

            <div className="panel-header">

              <div>

                <span className="panel-label">
                  01
                </span>

                <h2>
                  Write your review
                </h2>

              </div>


              {review && (
                <button
                  className="clear-button"
                  onClick={clearReview}
                >
                  CLEAR
                </button>
              )}

            </div>


            <textarea
              value={review}
              onChange={(event) =>
                setReview(event.target.value)
              }
              onKeyDown={handleKeyDown}
              placeholder="Tell us what you thought about the movie..."
              maxLength={5000}
            />


            <div className="input-meta">

              <span>
                {review.length.toLocaleString()} / 5,000
              </span>

              <span>
                Ctrl + Enter to analyze
              </span>

            </div>


            <button
              className="analyze-button"
              onClick={analyzeReview}
              disabled={
                loading ||
                !review.trim()
              }
            >

              {loading ? (
                <>
                  <span className="spinner" />

                  ANALYZING REVIEW...
                </>
              ) : (
                <>
                  ANALYZE SENTIMENT

                  <span className="arrow">
                    →
                  </span>
                </>
              )}

            </button>


            {/* ERROR */}
{error && (
  <div className="error-message">
    <span className="error-icon">⚠</span>

    <div>
      <strong>ANALYSIS FAILED</strong>
      <div>{error}</div>
    </div>
  </div>
)}


            {/* =========================
                EXAMPLES
            ========================== */}

            <div className="examples">

              <div className="examples-title">
                TRY AN EXAMPLE
              </div>


              <div className="example-list">

                {examples.map(
                  (example, index) => (

                    <button
                      key={index}
                      className="example-button"
                      onClick={() =>
                        selectExample(example)
                      }
                    >

                      <span>
                        0{index + 1}
                      </span>

                      {example}

                    </button>

                  )
                )}

              </div>

            </div>

          </div>


          {/* =========================
              RESULT PANEL
          ========================== */}

          <div className="result-panel">


            {/* EMPTY STATE */}

            {!result && !loading && (

              <div className="empty-result">

                <div className="empty-icon">
                  ◉
                </div>

                <span className="panel-label">
                  02
                </span>

                <h2>
                  AI analysis
                </h2>

                <p>
                  Your sentiment analysis will appear
                  here after you submit a review.
                </p>

                <div className="model-chip">

                  <span />

                  PYTORCH · LSTM

                </div>

                {!review.trim() && (
  <button
    className="empty-action"
    onClick={() => selectExample(examples[0])}
  >
    TRY A SAMPLE REVIEW
    <span>→</span>
  </button>
)}

              </div>

            )}


            {/* LOADING STATE */}

           {loading && (

  <div className="empty-result analyzing">

    <div className="analysis-orbit">

      <div className="orbit-core">
        N
      </div>

    </div>

    <span className="panel-label">
      PROCESSING
    </span>

    <h2>
      Reading between the lines...
    </h2>

    <p>
      Your review is being processed
      by the sentiment engine.
    </p>

    <div className="analyzing-text">
      TOKENIZING · EMBEDDING · LSTM INFERENCE
    </div>

    <div className="analyzing-subtext">
      Please wait while the model processes your review
    </div>

  </div>

)}

            {/* =========================
                FINAL RESULT
            ========================== */}

            {result && !loading && (

              <div className="result-content">


                <div className="result-top">

                  <span className="panel-label">
                    02 · AI ANALYSIS
                  </span>


                  <div className="confidence-pill">

                    {(
                      result.confidence * 100
                    ).toFixed(1)}

                    % CONFIDENCE

                  </div>

                </div>


                <div className="sentiment-display">
  <span className="sentiment-caption">OVERALL SENTIMENT</span>

  <div className="sentiment-ring">
    <svg viewBox="0 0 160 160" className="confidence-ring">
      <circle
        className="ring-background"
        cx="80"
        cy="80"
        r="68"
      />

      <circle
        className={`ring-progress ${
          result.sentiment === "positive"
            ? "positive-ring"
            : "negative-ring"
        }`}
        cx="80"
        cy="80"
        r="68"
        style={{
          strokeDasharray: `${result.confidence * 427.26} 427.26`,
        }}
      />
    </svg>

    <div className="ring-content">
      <strong>
        {(result.confidence * 100).toFixed(1)}%
      </strong>
      <span>CONFIDENCE</span>
    </div>
  </div>

  <h2
    className={
      result.sentiment === "positive"
        ? "positive-text"
        : "negative-text"
    }
  >
    {result.sentiment}
  </h2>
</div>

                {/* =========================
                    PROBABILITIES
                ========================== */}

                <div className="probabilities">

                  <div className="probability-header">

                    <span>
                      SENTIMENT PROBABILITY
                    </span>

                    <span>
                      MODEL OUTPUT
                    </span>

                  </div>


                  {/* POSITIVE */}

                  <div className="probability">

                    <div className="probability-info">

                      <span>
                        Positive
                      </span>

                      <strong>
                        {(
                          positive * 100
                        ).toFixed(1)}
                        %
                      </strong>

                    </div>


                    <div className="bar">

                      <div
                        className="bar-fill positive-bar"
                        style={{
                          width:
                            `${positive * 100}%`,
                        }}
                      />

                    </div>

                  </div>


                  {/* NEGATIVE */}

                  <div className="probability">

                    <div className="probability-info">

                      <span>
                        Negative
                      </span>

                      <strong>
                        {(
                          negative * 100
                        ).toFixed(1)}
                        %
                      </strong>

                    </div>


                    <div className="bar">

                      <div
                        className="bar-fill negative-bar"
                        style={{
                          width:
                            `${negative * 100}%`,
                        }}
                      />

                    </div>

                  </div>

                </div>


                {/* =========================
                    ANALYZED REVIEW
                ========================== */}

                <div className="analyzed-review">

                  <div className="review-heading">

                    <span>
                      ANALYZED REVIEW
                    </span>

                    <span>
                      {review.length} CHARACTERS
                    </span>

                  </div>


                  <p>
                    "{review}"
                  </p>

                </div>

              </div>

            )}

          </div>

        </section>


        {/* =========================
            AI PROCESS VISUALIZER
        ========================== */}

        <AIProcessVisualizer
          active={loading}
          result={result}
          review={review}
        />


        {/* =========================
            BATCH ANALYSIS
        ========================== */}

        <BatchAnalyzer />

        <ModelInfo />

        <PredictionHistory
          result={result}
          review={review}
        />


        {/* =========================
            TECHNOLOGY PIPELINE
        ========================== */}

        <section className="tech-strip">

          <div>

            <span className="tech-number">
              01
            </span>

            <span>
              Tokenization
            </span>

          </div>


          <div>

            <span className="tech-number">
              02
            </span>

            <span>
              Embedding
            </span>

          </div>


          <div>

            <span className="tech-number">
              03
            </span>

            <span>
              LSTM inference
            </span>

          </div>


          <div>

            <span className="tech-number">
              04
            </span>

            <span>
              Probability output
            </span>

          </div>

        </section>

      </main>


      {/* =========================
          FOOTER
      ========================== */}

      <footer>

        <span>
          REVIEW AI
        </span>

        <span>
          Built with React · FastAPI · PyTorch · LSTM
        </span>

        <span>
          Independent AI project
        </span>

      </footer>

    </div>
  );
}

export default App;
