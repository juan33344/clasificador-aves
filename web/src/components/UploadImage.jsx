/**
 * Componente UploadImage
 * Permite al usuario seleccionar una imagen desde su equipo
 */
import { useRef } from 'react';

export default function UploadImage({ onImageSelected, isLoading }) {
  const fileInputRef = useRef(null);

  /**
   * Maneja la selección de archivo
   * Valida que sea una imagen y la carga
   */
  const handleFileChange = (event) => {
    const file = event.target.files[0];

    if (!file) return;

    // Validar que sea una imagen
    if (!file.type.startsWith('image/')) {
      alert('Por favor selecciona un archivo de imagen válido');
      return;
    }

    // Crear un objeto URL para la imagen
    const imageURL = URL.createObjectURL(file);

    // Crear un elemento de imagen
    const img = new Image();
    img.src = imageURL;

    // Cuando la imagen cargue, pasarla al componente padre
    img.onload = () => {
      onImageSelected(file, imageURL);
    };

    img.onerror = () => {
      alert('Error al cargar la imagen');
    };
  };

  /**
   * Abre el diálogo de selección de archivo
   */
  const handleClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  return (
    <div className="upload-section">
      <div className="upload-container">
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          style={{ display: 'none' }}
          disabled={isLoading}
        />

        <button
          onClick={handleClick}
          className="upload-button"
          disabled={isLoading}
        >
          <span className="upload-icon">📸</span>
          <span className="upload-text">
            {isLoading ? 'Procesando...' : 'Seleccionar Imagen'}
          </span>
        </button>

        <p className="upload-hint">
          Haz clic para seleccionar una imagen de tu equipo
        </p>
      </div>
    </div>
  );
}
