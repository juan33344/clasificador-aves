/**
 * Componente PredictionCard
 * Muestra los resultados de la predicción del modelo
 */

export default function PredictionCard({ prediction, imageURL }) {
  if (!prediction) {
    return null;
  }

  const { className, confidence, allPredictions } = prediction;

  // Calcular color basado en confianza
  const getConfidenceColor = (conf) => {
    const percentage = parseFloat(conf);
    if (percentage >= 80) return '#10b981'; // verde
    if (percentage >= 60) return '#f59e0b'; // amarillo
    return '#ef4444'; // rojo
  };

  return (
    <div className="prediction-card">
      <div className="prediction-grid">
        {/* Sección de imagen */}
        <div className="prediction-image-section">
          <div className="prediction-image-container">
            <img
              src={imageURL}
              alt="Imagen predicha"
              className="prediction-image"
            />
          </div>
        </div>

        {/* Sección de resultados */}
        <div className="prediction-results-section">
          {/* Predicción principal */}
          <div className="top-prediction">
            <h2 className="prediction-title">Ave Detectada</h2>
            <div className="prediction-class">{className}</div>

            <div className="confidence-section">
              <p className="confidence-label">Confianza</p>
              <div className="confidence-meter">
                <div className="confidence-bar">
                  <div
                    className="confidence-fill"
                    style={{
                      width: `${confidence}%`,
                      backgroundColor: getConfidenceColor(confidence),
                    }}
                  ></div>
                </div>
                <p
                  className="confidence-value"
                  style={{ color: getConfidenceColor(confidence) }}
                >
                  {confidence}%
                </p>
              </div>
            </div>
          </div>

          {/* Todas las predicciones */}
          <div className="all-predictions">
            <h3 className="all-predictions-title">Todas las Predicciones</h3>
            <div className="predictions-list">
              {allPredictions.map((pred, index) => (
                <div key={index} className="prediction-item">
                  <span className="prediction-item-class">
                    {pred.class}
                  </span>
                  <span className="prediction-item-percentage">
                    {pred.percentage}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
