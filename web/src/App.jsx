/**
 * Componente App
 * Componente raíz de la aplicación
 * Renderiza la barra de navegación y la página principal
 */
import Navbar from './components/Navbar';
import Home from './pages/Home';
import './index.css';

function App() {
  return (
    <div className="app">
      <Navbar />
      <Home />
      <footer className="app-footer">
        <p>© 2024 Clasificador de Aves | Desarrollado con React + TensorFlow.js</p>
      </footer>
    </div>
  );
}

export default App;
