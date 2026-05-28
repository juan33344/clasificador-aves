const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

/**
 * Verifica que la API Python esté disponible y el modelo cargado.
 * @returns {Promise<boolean>}
 */
export const loadModel = async () => {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error('No se pudo conectar con la API de predicción');
  }

  const data = await response.json();
  if (!data.model_loaded) {
    throw new Error('El modelo no está cargado en el servidor');
  }

  return true;
};

/**
 * Realiza una predicción enviando la imagen al servidor Python.
 * @param {File} file - Archivo de imagen seleccionado
 * @returns {Promise<Object>} Resultado de la predicción
 */
export const predictImage = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => null);
    throw new Error(errorBody?.detail || 'Error al predecir la imagen');
  }

  return await response.json();
};

/**
 * Obtiene las etiquetas disponibles desde el servidor.
 * @returns {Promise<Array<string>>}
 */
export const getLabels = async () => {
  const response = await fetch(`${API_BASE_URL}/labels`);
  if (!response.ok) {
    throw new Error('No se pudieron obtener las etiquetas');
  }

  const data = await response.json();
  return data.labels || [];
};
