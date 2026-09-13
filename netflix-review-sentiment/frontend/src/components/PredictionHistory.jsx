import { useEffect, useState } from "react";

const STORAGE_KEY = "review-ai-history";

function PredictionHistory({ result, review }) {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const savedHistory = localStorage.getItem(STORAGE_KEY);

    if (savedHistory) {
      try {
        setHistory(JSON.parse(savedHistory));
      } catch {
        localStorage.removeItem(STORAGE_KEY);
      }
    }
  }, []);

  useEffect(() => {
    if (!result || !review.trim()) {
      return;
    }

    const newEntry = {
      id: Date.now(),
      review: review.trim(),
      sentiment: result.sentiment,
      confidence: result.confidence,
      positive_probability: result.positive_probability,
      negative_probability: result.negative_probability,
      timestamp: new Date().toISOString(),
    };

    setHistory((currentHistory) => {
      const updatedHistory = [
        newEntry,
        ...currentHistory,
      ].slice(0, 20);

      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify(updatedHistory)
      );

      return updatedHistory;
    });
  }, [result]);

  const clearHistory = () => {
    localStorage.removeItem(STORAGE_KEY);
    setHistory([]);
  };

  const formatTime = (timestamp) => {
    const date = new Date(timestamp);

    return date.toLocaleString([], {
      day: "2-digit",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <section className="history-section">

      <div className="history-header">

        <div>
          <span className="panel-label">
            05 · HISTORY
          </span>

          <h2>
            Recent analyses
          </h2>

          <p>
            Your latest sentiment predictions are stored
            locally in this browser.
          </p>
        </div>

        {history.length > 0 && (
          <button
            className="clear-history-button"
            onClick={clearHistory}
          >
            CLEAR HISTORY
          </button>
        )}

      </div>


      {history.length === 0 ? (

        <div className="history-empty">

          <div className="history-empty-icon">
            ◷
          </div>

          <h3>
            No analyses yet
          </h3>

          <p>
            Analyze a review and your prediction will
            appear here.
          </p>

        </div>

      ) : (

        <div className="history-list">

          {history.map((item, index) => (

            <div
              className="history-item"
              key={item.id}
            >

              <div className="history-index">
                {String(index + 1).padStart(2, "0")}
              </div>


              <div className="history-review">

                <p>
                  "{item.review}"
                </p>

                <span>
                  {formatTime(item.timestamp)}
                </span>

              </div>


              <div className="history-sentiment">

                <span
                  className={`history-badge ${item.sentiment}`}
                >
                  {item.sentiment}
                </span>

              </div>


              <div className="history-confidence">

                <div className="history-confidence-label">

                  <span>
                    CONFIDENCE
                  </span>

                  <strong>
                    {(item.confidence * 100).toFixed(1)}%
                  </strong>

                </div>

                <div className="history-confidence-bar">

                  <div
                    className={`history-confidence-fill ${item.sentiment}`}
                    style={{
                      width: `${item.confidence * 100}%`,
                    }}
                  />

                </div>

              </div>

            </div>

          ))}

        </div>

      )}

    </section>
  );
}

export default PredictionHistory;