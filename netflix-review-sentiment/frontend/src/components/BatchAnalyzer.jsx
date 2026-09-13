import { useState } from "react";

const API_URL = "http://127.0.0.1:8000";

function BatchAnalyzer() {
  const [file, setFile] = useState(null);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("all");

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];

    if (!selectedFile) return;

    if (!selectedFile.name.toLowerCase().endsWith(".csv")) {
      setError("Please select a CSV file.");
      return;
    }

    setFile(selectedFile);
    setData(null);
    setError("");
  };

  const analyzeBatch = async () => {
    if (!file) return;

    setLoading(true);
    setError("");
    setData(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_URL}/batch-predict`, {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail || "Batch analysis failed.");
      }

      setData(result);
    } catch (err) {
      console.error(err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const downloadResults = () => {
    if (!data) return;

    const headers = [
      "review",
      "sentiment",
      "confidence",
      "positive_probability",
      "negative_probability",
    ];

    const rows = data.results.map((item) => [
      `"${item.review.replace(/"/g, '""')}"`,
      item.sentiment,
      item.confidence.toFixed(4),
      item.positive_probability.toFixed(4),
      item.negative_probability.toFixed(4),
    ]);

    const csv = [
      headers.join(","),
      ...rows.map((row) => row.join(",")),
    ].join("\n");

    const blob = new Blob([csv], {
      type: "text/csv;charset=utf-8;",
    });

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = "sentiment_analysis_results.csv";

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    URL.revokeObjectURL(url);
  };

  const filteredResults =
    data?.results.filter((item) => {
      if (filter === "all") return true;
      return item.sentiment === filter;
    }) || [];

  return (
    <section className="batch-section">
      <div className="batch-heading">
        <div>
          <span className="panel-label">03</span>
          <h2>Batch analysis</h2>
          <p>
            Analyze up to 1,000 movie reviews at once using the LSTM
            engine.
          </p>
        </div>
      </div>

      <div className="batch-upload-card">
        <div className="upload-icon">↑</div>

        <div className="upload-content">
          <h3>Upload review dataset</h3>

          <p>
            CSV files must contain a <strong>review</strong> column.
          </p>

          <label className="file-button">
            CHOOSE CSV
            <input
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              hidden
            />
          </label>

          {file && (
            <div className="selected-file">
              <span>✓</span>
              {file.name}
            </div>
          )}
        </div>

        <button
          className="batch-analyze-button"
          onClick={analyzeBatch}
          disabled={!file || loading}
        >
          {loading ? (
            <>
              <span className="spinner" />
              ANALYZING...
            </>
          ) : (
            <>
              RUN BATCH ANALYSIS
              <span>→</span>
            </>
          )}
        </button>
      </div>

      {error && (
        <div className="batch-error">
          {error}
        </div>
      )}

      {data && (
        <>
          <div className="batch-stats">
            <div className="stat-card">
              <span className="stat-label">TOTAL REVIEWS</span>
              <strong>{data.total_reviews}</strong>
            </div>

            <div className="analytics-grid">
                <div className="sentiment-chart-card">
                   <div className="analytics-header">
                       <div>
                            <span className="panel-label">ANALYTICS</span>
                            <h3>Sentiment distribution</h3>
                        </div>
                     </div>

    <div className="donut-wrapper">
      <div
        className="donut"
        style={{
          background: `conic-gradient(
            #e50914 0% ${data.negative_percentage}%,
            #5cdb89 ${data.negative_percentage}% 100%
          )`,
        }}
      >
        <div className="donut-center">
          <strong>{data.total_reviews}</strong>
          <span>REVIEWS</span>
        </div>
      </div>

      <div className="chart-legend">
        <div>
          <span className="legend-dot negative-dot" />
          <div>
            <strong>Negative</strong>
            <span>
              {data.negative_percentage.toFixed(1)}%
            </span>
          </div>
        </div>

        <div>
          <span className="legend-dot positive-dot" />
          <div>
            <strong>Positive</strong>
            <span>
              {data.positive_percentage.toFixed(1)}%
            </span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div className="confidence-card">
    <div className="analytics-header">
      <div>
        <span className="panel-label">MODEL</span>
        <h3>Prediction confidence</h3>
      </div>
    </div>

    <div className="confidence-value">
      {(data.average_confidence * 100).toFixed(1)}
      <span>%</span>
    </div>

    <div className="confidence-track">
      <div
        className="confidence-fill"
        style={{
          width: `${data.average_confidence * 100}%`,
        }}
      />
    </div>

    <p>
      Average confidence across all valid predictions.
    </p>

    <div className="confidence-model">
      <span className="status-dot" />
      LSTM SENTIMENT ENGINE
    </div>
  </div>
</div>

            <div className="stat-card positive-stat">
              <span className="stat-label">POSITIVE</span>
              <strong>{data.positive_reviews}</strong>
              <small>
                {data.positive_percentage.toFixed(1)}%
              </small>
            </div>

            <div className="stat-card negative-stat">
              <span className="stat-label">NEGATIVE</span>
              <strong>{data.negative_reviews}</strong>
              <small>
                {data.negative_percentage.toFixed(1)}%
              </small>
            </div>

            <div className="stat-card">
              <span className="stat-label">AVG. CONFIDENCE</span>
              <strong>
                {(data.average_confidence * 100).toFixed(1)}%
              </strong>
            </div>
          </div>

          <div className="batch-results-card">
            <div className="batch-results-header">
              <div>
                <span className="panel-label">RESULTS</span>
                <h3>Review explorer</h3>
              </div>

              <div className="results-actions">
                <div className="filter-buttons">
                  {["all", "positive", "negative"].map((option) => (
                    <button
                      key={option}
                      className={filter === option ? "active" : ""}
                      onClick={() => setFilter(option)}
                    >
                      {option.toUpperCase()}
                    </button>
                  ))}
                </div>

                <button
                  className="download-button"
                  onClick={downloadResults}
                >
                  ↓ DOWNLOAD CSV
                </button>
              </div>
            </div>

            <div className="results-table-wrapper">
              <table className="results-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>REVIEW</th>
                    <th>SENTIMENT</th>
                    <th>CONFIDENCE</th>
                  </tr>
                </thead>

                <tbody>
                  {filteredResults.map((item, index) => (
                    <tr key={index}>
                      <td>{index + 1}</td>

                      <td className="review-cell">
                        {item.review}
                      </td>

                      <td>
                        <span
                          className={`sentiment-badge ${item.sentiment}`}
                        >
                          {item.sentiment}
                        </span>
                      </td>

                      <td>
                        <div className="table-confidence">
                          <div className="mini-bar">
                            <div
                              style={{
                                width: `${item.confidence * 100}%`,
                              }}
                            />
                          </div>

                          {(item.confidence * 100).toFixed(1)}%
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {filteredResults.length === 0 && (
              <div className="no-results">
                No reviews match the selected filter.
              </div>
            )}
          </div>
        </>
      )}
    </section>
  );
}

export default BatchAnalyzer;
