import { useEffect, useState } from "react";

const stages = [
  {
    id: "tokenization",
    number: "01",
    title: "TOKENIZATION",
    description: "Breaking the review into individual tokens",
  },
  {
    id: "embedding",
    number: "02",
    title: "EMBEDDING",
    description: "Converting tokens into learned vector representations",
  },
  {
    id: "lstm",
    number: "03",
    title: "LSTM ENGINE",
    description: "Processing the sequence and learning contextual relationships",
  },
  {
    id: "classification",
    number: "04",
    title: "CLASSIFICATION",
    description: "Converting the final hidden state into sentiment probabilities",
  },
];

function AIProcessVisualizer({ active, result }) {
  const [stageIndex, setStageIndex] = useState(0);
  const [running, setRunning] = useState(false);
  const [lstmTokenIndex, setLstmTokenIndex] = useState(0);

  const tokens = result?.tokens || [];

  /*
   * Start the visualization whenever a new analysis begins.
   */
  useEffect(() => {
    if (!active) return;

    setStageIndex(0);
    setRunning(true);
    setLstmTokenIndex(0);
  }, [active]);

  /*
   * Control the main pipeline stages.
   *
   * Tokenization       -> 3 seconds
   * Embedding          -> 3 seconds
   * LSTM               -> 8 seconds
   * Classification     -> 3 seconds
   */
  useEffect(() => {
    if (!running) return;

    const stageDuration =
      stageIndex === 2
        ? 8000
        : 3000;

    const timer = setTimeout(() => {
      setStageIndex((current) => {
        if (current < stages.length - 1) {
          return current + 1;
        }

        setRunning(false);
        return current;
      });
    }, stageDuration);

    return () => clearTimeout(timer);
  }, [running, stageIndex]);

  /*
   * Animate tokens through the LSTM one at a time.
   *
   * This is controlled by React instead of relying on CSS
   * animation timing, so tokens don't randomly skip.
   */
  useEffect(() => {
    if (!running || stageIndex !== 2) {
      return;
    }

    if (tokens.length === 0) {
      return;
    }

    setLstmTokenIndex(0);

    const interval = setInterval(() => {
      setLstmTokenIndex((current) => {
        if (current < tokens.length - 1) {
          return current + 1;
        }

        return 0;
      });
    }, 750);

    return () => clearInterval(interval);
  }, [running, stageIndex, tokens.length]);

  /*
   * Don't show the visualizer until the user has submitted
   * a review at least once.
   */
  if (!active && !result) {
    return null;
  }

  const currentStage = stages[stageIndex];

  return (
    <section className="ai-process-section">

      {/* HEADER */}
      <div className="ai-process-header">

        <div>
          <span className="panel-label">
            AI PIPELINE
          </span>

          <h2>
            Inside the sentiment engine
          </h2>

          <p>
            Watch your review move through the same
            processing pipeline used by the model.
          </p>
        </div>

        <div className="pipeline-status">
          <span className="status-dot" />

          {running
            ? "PROCESSING"
            : "ANALYSIS COMPLETE"}
        </div>

      </div>


      {/* STAGE INDICATOR */}
      <div className="pipeline-stages">

        {stages.map((stage, index) => (
          <div
            key={stage.id}
            className={`pipeline-stage ${
              index === stageIndex
                ? "active"
                : index < stageIndex
                ? "completed"
                : ""
            }`}
          >

            <div className="stage-number">
              {index < stageIndex
                ? "✓"
                : stage.number}
            </div>

            <span>
              {stage.title}
            </span>

          </div>
        ))}

      </div>


      {/* MAIN VISUALIZATION */}
      <div className="visualizer-card">

        {/* STAGE TITLE */}
        <div className="visualizer-title">

          <div>
            <span className="visualizer-stage-number">
              {currentStage.number}
            </span>

            <h3>
              {currentStage.title}
            </h3>
          </div>

          <p>
            {currentStage.description}
          </p>

        </div>


        {/* TOKENIZATION */}
        {currentStage.id === "tokenization" && (
          <div className="tokenization-view">

            <div className="raw-review">
              <span className="visual-label">
                RAW REVIEW
              </span>

              <p>
                "{result?.tokens?.join(" ") || "Waiting for review..."}"
              </p>
            </div>

            <div className="process-arrow">
              ↓
            </div>

            <div className="token-container">

              <span className="visual-label">
                TOKENS
              </span>

              <div className="token-list">

                {tokens.map((token, index) => (
                  <div
                    key={index}
                    className="token"
                  >
                    <span>
                      t{index + 1}
                    </span>

                    {token}
                  </div>
                ))}

              </div>

            </div>

          </div>
        )}


        {/* EMBEDDING */}
        {currentStage.id === "embedding" && (
          <div className="embedding-view">

            <div className="embedding-tokens">

              {tokens.slice(0, 8).map((token, index) => (
                <div
                  key={index}
                  className="embedding-token"
                  style={{
                    animationDelay: `${index * 0.15}s`,
                  }}
                >
                  <span>
                    t{index + 1}
                  </span>

                  {token}
                </div>
              ))}

            </div>

            <div className="embedding-arrow">
              →
            </div>

            <div className="vector-display">

              <span className="visual-label">
                128-DIMENSIONAL VECTOR
              </span>

              <div className="vector-grid">

                {Array.from(
                  { length: 48 },
                  (_, index) => (
                    <div
                      key={index}
                      className="vector-cell"
                      style={{
                        animationDelay:
                          `${index * 0.03}s`,
                      }}
                    />
                  )
                )}

              </div>

              <div className="vector-values">
                [ 0.184, -0.723, 0.391, 0.052,
                ... 124 more dimensions ]
              </div>

            </div>

          </div>
        )}


        {/* LSTM */}
        {currentStage.id === "lstm" && (
          <div className="lstm-view">

            <div className="lstm-sequence">

              <span className="visual-label">
                SEQUENTIAL INPUT
              </span>

              <div className="lstm-token-list">

                {tokens.slice(0, 10).map((token, index) => (
                  <div
                    key={index}
                    className={`lstm-token ${
                      index === lstmTokenIndex
                        ? "processing"
                        : index < lstmTokenIndex
                        ? "processed"
                        : ""
                    }`}
                  >

                    <span className="lstm-token-number">
                      t{index + 1}
                    </span>

                    <span className="lstm-token-word">
                      {token}
                    </span>

                  </div>
                ))}

              </div>

            </div>


            <div className="lstm-flow">

              <div className="flow-line" />

              <div className="lstm-core">

                <div className="lstm-core-ring" />

                <div className="lstm-core-inner">
                  LSTM
                </div>

                <span>
                  MEMORY
                </span>

              </div>

              <div className="flow-particle" />

            </div>


            <div className="hidden-state">

              <span className="visual-label">
                HIDDEN STATE
              </span>

              <div className="hidden-vector">

                {Array.from(
                  { length: 32 },
                  (_, index) => (
                    <div
                      key={index}
                      className="hidden-cell"
                      style={{
                        animationDelay:
                          `${index * 0.05}s`,
                      }}
                    />
                  )
                )}

              </div>

              <span className="hidden-dimension">
                128 DIMENSIONS
              </span>

            </div>

          </div>
        )}


        {/* CLASSIFICATION */}
        {currentStage.id === "classification" && (
          <div className="classification-view">

            <div className="classification-input">

              <span className="visual-label">
                FINAL HIDDEN STATE
              </span>

              <div className="classification-vector">

                {Array.from(
                  { length: 24 },
                  (_, index) => (
                    <div
                      key={index}
                      className="classification-cell"
                    />
                  )
                )}

              </div>

            </div>

            <div className="classification-arrow">
              →
            </div>

            <div className="classification-output">

              <span className="visual-label">
                MODEL OUTPUT
              </span>

              <div className="output-probabilities">

                <div className="output-row">

                  <div className="output-label">
                    <span>
                      Positive
                    </span>

                    <strong>
                      {result
                        ? `${(
                            result.positive_probability *
                            100
                          ).toFixed(1)}%`
                        : "—"}
                    </strong>
                  </div>

                  <div className="output-bar">
                    <div
                      className="output-fill positive-output"
                      style={{
                        width: result
                          ? `${result.positive_probability * 100}%`
                          : "0%",
                      }}
                    />
                  </div>

                </div>


                <div className="output-row">

                  <div className="output-label">
                    <span>
                      Negative
                    </span>

                    <strong>
                      {result
                        ? `${(
                            result.negative_probability *
                            100
                          ).toFixed(1)}%`
                        : "—"}
                    </strong>
                  </div>

                  <div className="output-bar">
                    <div
                      className="output-fill negative-output"
                      style={{
                        width: result
                          ? `${result.negative_probability * 100}%`
                          : "0%",
                      }}
                    />
                  </div>

                </div>

              </div>

            </div>

          </div>
        )}

      </div>


      {/* PIPELINE FOOTER */}
      <div className="pipeline-footer">

        <div>
          <span>MODEL</span>
          <strong>
            PyTorch LSTM
          </strong>
        </div>

        <div>
          <span>EMBEDDING</span>
          <strong>
            128 DIM
          </strong>
        </div>

        <div>
          <span>HIDDEN STATE</span>
          <strong>
            128 DIM
          </strong>
        </div>

        <div>
          <span>SEQUENCE</span>
          <strong>
            200 TOKENS
          </strong>
        </div>

        <div>
          <span>ACCURACY</span>
          <strong>
            86.52%
          </strong>
        </div>

      </div>

    </section>
  );
}

export default AIProcessVisualizer;