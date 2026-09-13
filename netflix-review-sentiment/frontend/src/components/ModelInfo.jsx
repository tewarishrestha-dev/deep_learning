function ModelInfo() {
  const specs = [
    {
      value: "86.52%",
      label: "TEST ACCURACY",
      description: "IMDb test set",
    },
    {
      value: "30K",
      label: "VOCABULARY",
      description: "Maximum vocabulary size",
    },
    {
      value: "128",
      label: "EMBEDDING DIM",
      description: "Learned word vectors",
    },
    {
      value: "128",
      label: "HIDDEN DIM",
      description: "LSTM hidden state",
    },
    {
      value: "200",
      label: "MAX SEQUENCE",
      description: "Tokens per review",
    },
    {
      value: "LSTM",
      label: "ARCHITECTURE",
      description: "Recurrent neural network",
    },
  ];

  return (
    <section className="model-info-section">

      <div className="model-info-header">

        <div>
          <span className="panel-label">
            04 · MODEL
          </span>

          <h2>
            What's powering the AI?
          </h2>

          <p>
            The sentiment engine uses a PyTorch LSTM
            trained on 50,000 IMDb movie reviews.
          </p>
        </div>

        <div className="model-badge">
          <span className="status-dot" />
          PYTORCH · LSTM
        </div>

      </div>


      {/* MODEL ARCHITECTURE */}

      <div className="architecture-card">

        <div className="architecture-header">
          <span className="visual-label">
            MODEL ARCHITECTURE
          </span>

          <span className="architecture-detail">
            SEQUENCE CLASSIFICATION
          </span>
        </div>


        <div className="architecture-flow">

          <div className="architecture-node">
            <span className="node-number">
              01
            </span>

            <strong>
              INPUT
            </strong>

            <small>
              Token IDs
            </small>
          </div>


          <div className="architecture-arrow">
            →
          </div>


          <div className="architecture-node highlighted">
            <span className="node-number">
              02
            </span>

            <strong>
              EMBEDDING
            </strong>

            <small>
              128 dimensions
            </small>
          </div>


          <div className="architecture-arrow">
            →
          </div>


          <div className="architecture-node highlighted">
            <span className="node-number">
              03
            </span>

            <strong>
              LSTM
            </strong>

            <small>
              128 hidden units
            </small>
          </div>


          <div className="architecture-arrow">
            →
          </div>


          <div className="architecture-node">
            <span className="node-number">
              04
            </span>

            <strong>
              LINEAR
            </strong>

            <small>
              1 output
            </small>
          </div>


          <div className="architecture-arrow">
            →
          </div>


          <div className="architecture-node result-node">
            <span className="node-number">
              05
            </span>

            <strong>
              SIGMOID
            </strong>

            <small>
              Probability
            </small>
          </div>

        </div>

      </div>


      {/* MODEL SPECS */}

      <div className="model-spec-grid">

        {specs.map((spec, index) => (
          <div
            className="model-spec-card"
            key={index}
          >

            <span className="spec-number">
              0{index + 1}
            </span>

            <strong className="spec-value">
              {spec.value}
            </strong>

            <span className="spec-label">
              {spec.label}
            </span>

            <span className="spec-description">
              {spec.description}
            </span>

          </div>
        ))}

      </div>


      {/* EXPLANATION */}

      <div className="model-explanation">

        <div className="explanation-icon">
          ∿
        </div>

        <div>
          <span className="visual-label">
            WHY LSTM?
          </span>

          <p>
            Unlike a simple feed-forward network, the LSTM
            processes the review sequentially and maintains
            a hidden state that carries contextual information
            across the sequence.
          </p>
        </div>

      </div>

    </section>
  );
}

export default ModelInfo;