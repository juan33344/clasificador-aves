/**
 * Componente Loader
 * Muestra una animación de carga mientras se procesan datos
 */
import '../styles/loader.css';

export default function Loader({ message = 'Cargando...' }) {
  return (
    <div className="loader-container">
      <div className="spinner"></div>
      <p className="loader-text">{message}</p>
    </div>
  );
}
