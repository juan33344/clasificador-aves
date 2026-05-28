"""
Script mejorado para hacer predicciones con el modelo de clasificación de aves
"""

import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import os
import sys

def cargar_modelo_y_etiquetas():
    """Carga el modelo y las etiquetas"""
    
    # Verificar que existe el modelo
    if not os.path.exists("keras_model.h5"):
        print("❌ Error: No se encuentra 'keras_model.h5'")
        print("   Por favor, entrena el modelo primero ejecutando: python train.py")
        sys.exit(1)
    
    if not os.path.exists("labels.txt"):
        print("❌ Error: No se encuentra 'labels.txt'")
        sys.exit(1)
    
    print("📦 Cargando modelo...")
    model = tensorflow.keras.models.load_model("keras_model.h5", compile=False)
    
    print("📋 Cargando etiquetas...")
    class_names = [line.strip() for line in open("labels.txt", "r", encoding="utf-8").readlines() if line.strip()]
    
    return model, class_names

def predecir_imagen(model, class_names, ruta_imagen):
    """Realiza predicción sobre una imagen"""
    
    # Verificar que existe la imagen
    if not os.path.exists(ruta_imagen):
        print(f"❌ Error: No se encuentra la imagen '{ruta_imagen}'")
        return None
    
    try:
        print(f"\n🖼️  Procesando imagen: {ruta_imagen}")
        
        # Crear array para la imagen
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        
        # Abrir imagen
        image = Image.open(ruta_imagen).convert("RGB")
        
        # Redimensionar imagen
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        
        # Convertir a array
        image_array = np.asarray(image)
        
        # Normalizar (método MobileNetV2: -1 a 1)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        
        # Cargar datos
        data[0] = normalized_image_array
        
        # Predecir
        print("🤖 Realizando predicción...")
        prediction = model.predict(data, verbose=0)
        
        # Obtener índice
        index = np.argmax(prediction)
        
        # Obtener clase
        class_name = class_names[index]
        if ' ' in class_name:
            class_name = class_name.split(' ', 1)[1].strip()
        
        # Precisión
        confidence_score = prediction[0][index]
        
        return {
            "clase": class_name,
            "confianza": float(confidence_score),
            "porcentaje": float(confidence_score * 100),
            "predicciones": prediction[0]
        }
    
    except Exception as e:
        print(f"❌ Error procesando imagen: {str(e)}")
        return None

def mostrar_resultados(resultado, class_names):
    """Muestra los resultados de la predicción"""
    
    if resultado is None:
        return
    
    print("\n" + "=" * 50)
    print("✨ RESULTADOS DE LA PREDICCIÓN")
    print("=" * 50)
    
    print(f"\n🎯 Clase predicha: {resultado['clase']}")
    print(f"📊 Confianza: {resultado['porcentaje']:.2f}%")
    
    # Mostrar top 3 predicciones
    print(f"\n📈 Top 3 predicciones:")
    
    predicciones_ordenadas = list(enumerate(resultado['predicciones']))
    predicciones_ordenadas.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (idx, score) in enumerate(predicciones_ordenadas[:3], 1):
        clase = class_names[idx]
        if ' ' in clase:
            clase = clase.split(' ', 1)[1].strip()
        print(f"   {rank}. {clase}: {score * 100:.2f}%")
    
    print("\n" + "=" * 50)

def main():
    """Función principal"""
    
    print("=" * 50)
    print("🦅 PREDICTOR DE AVES")
    print("=" * 50)
    
    # Cargar modelo y etiquetas
    model, class_names = cargar_modelo_y_etiquetas()
    print("✓ Modelo cargado exitosamente\n")
    
    # Procesamiento de argumentos o entrada interactiva
    if len(sys.argv) > 1:
        # Si se pasa una ruta como argumento
        ruta_imagen = sys.argv[1]
        resultado = predecir_imagen(model, class_names, ruta_imagen)
        mostrar_resultados(resultado, class_names)
    else:
        # Modo interactivo
        print("💡 Uso: python predict.py <ruta_imagen>")
        print("   Ejemplo: python predict.py test.jpg\n")
        
        while True:
            ruta = input("📁 Ingresa la ruta de la imagen (o 'salir' para terminar): ").strip()
            
            if ruta.lower() in ['salir', 'exit', 'quit']:
                print("👋 ¡Hasta luego!")
                break
            
            if ruta:
                resultado = predecir_imagen(model, class_names, ruta)
                mostrar_resultados(resultado, class_names)

if __name__ == "__main__":
    main()