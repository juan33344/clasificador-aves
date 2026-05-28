/**
 * Página Home
 * Componente principal que gestiona el flujo de clasificación de aves
 */
import { useState, useEffect } from 'react';
import UploadImage from '../components/UploadImage';
import PredictionCard from '../components/PredictionCard';
import Loader from '../components/Loader';
import { loadModel, predictImage } from '../services/modelService';

export default function Home() {
  // Estados principales
  const [modelReady, setModelReady] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [modelError, setModelError] = useState(null);
  const [imageURL, setImageURL] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionError, setPredictionError] = useState(null);

  /**
   * useEffect: Cargar el modelo al montar el componente
   */
  useEffect(() => {
    const initializeModel = async () => {
      try {
        setIsLoading(true);
        setModelError(null);
        await loadModel();
        setModelReady(true);
      } catch (error) {
        console.error('Error al inicializar:', error);
        setModelError(error.message);
      } finally {
        setIsLoading(false);
      }
    };

    initializeModel();
  }, []);

  /**
   * Maneja la selección de imagen
   * @param {HTMLImageElement} imageElement - Elemento de imagen cargado
   * @param {string} url - URL de la imagen para mostrar
   */
  const handleImageSelected = async (file, url) => {
    try {
      setImageURL(url);
      setPrediction(null);
      setPredictionError(null);
      setIsPredicting(true);

      // Realizar predicción
      const result = await predictImage(file);
      setPrediction(result);
    } catch (error) {
      console.error('Error en predicción:', error);
      setPredictionError(error.message);
    } finally {
      setIsPredicting(false);
    }
  };

  /**
   * Reinicia la aplicación para una nueva predicción
   */
  const handleReset = () => {
    setImageURL(null);
    setPrediction(null);
    setPredictionError(null);
  };

  // Mostrar error de carga del modelo
  if (modelError) {
    return (
      <div className="error-container">
        <div className="error-box">
          <h2>❌ Error al Cargar el Modelo</h2>
          <p>{modelError}</p>
          <p className="error-hint">
            Verifica que los archivos del modelo estén en: public/model/
          </p>
          <button onClick={() => window.location.reload()} className="retry-button">
            Reintentar
          </button>
        </div>
      </div>
    );
  }

  // Mostrar loader mientras se carga el modelo
  if (!modelReady) {
    return <Loader message="Inicializando modelo de IA..." />;
  }

  return (
    <main className="home-container">
      {/* Sección de carga de imagen */}
      <section className="upload-section-wrapper">
        <UploadImage
          onImageSelected={handleImageSelected}
          isLoading={isPredicting}
        />
      </section>

      {/* Mostrar loader mientras se predice */}
      {isPredicting && <Loader message="Analizando imagen..." />}

      {/* Mostrar error de predicción */}
      {predictionError && !isPredicting && (
        <div className="error-message">
          <p>⚠️ {predictionError}</p>
        </div>
      )}

      {/* Mostrar resultados si hay predicción */}
      {prediction && !isPredicting && (
        <>
          <PredictionCard prediction={prediction} imageURL={imageURL} />

          <div className="action-buttons">
            <button onClick={handleReset} className="reset-button">
              🔄 Nueva Predicción
            </button>
          </div>
        </>
      )}

      {/* Mostrar hint cuando no hay predicción */}
      {!prediction && !isPredicting && (
        <div className="info-section">
          <p className="info-text">
            Sube una imagen de un ave para obtener una predicción automática
          </p>
        </div>
      )}
    </main>
  );
}
