/**
 * Componente Navbar
 * Barra de navegación superior de la aplicación
 */

export default function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar-container">
        <div className="navbar-brand">
          <span className="brand-icon">🦅</span>
          <h1 className="brand-title">Clasificador de Aves</h1>
        </div>
        <p className="navbar-subtitle">
          Powered by TensorFlow.js & Teachable Machine
        </p>
      </div>
    </nav>
  );
}
