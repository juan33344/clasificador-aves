"""
Script mejorado para entrenar un modelo de clasificación de aves
Usa Transfer Learning con MobileNetV2 + Data Augmentation para mejor precisión

Estructura esperada:
    data/
    ├── Aguila/
    │   ├── imagen1.jpg
    │   └── imagen2.jpg
    ├── Buho/
    │   ├── imagen1.jpg
    │   └── imagen2.jpg
    └── ... (una carpeta por cada ave)
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt

# Configuración
DATA_DIR = "data"
LABELS_FILE = "labels.txt"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

DEFAULT_CLASS_NAMES = [
    "Águila",
    "Búho",
    "Carpintero",
    "Colibrí",
    "Cuervo",
    "Flamenco",
    "Garza",
    "Gaviota",
    "Loro",
    "Paloma"
]

class_names = []
NUM_CLASSES = 0

METRICS = [
    'accuracy',
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]

def crear_directorio_datos():
    """Crea la estructura de carpetas si no existe"""
    if not os.path.exists(DATA_DIR):
        print(f"📁 Creando carpeta '{DATA_DIR}' para los datos...")
        os.makedirs(DATA_DIR)
        
        # Crear carpetas por clase
        for clase in DEFAULT_CLASS_NAMES:
            clase_dir = os.path.join(DATA_DIR, clase)
            os.makedirs(clase_dir, exist_ok=True)
            print(f"   ✓ Carpeta creada: {clase_dir}")
        
        print("\n⚠️  Por favor, coloca las imágenes de cada ave en su carpeta correspondiente:")
        print("    Ejemplo: data/Aguila/imagen1.jpg, data/Aguila/imagen2.jpg, etc.")
        print("\n💡 Recomendación: Usa al menos 20-50 imágenes por clase para mejor entrenamiento")
        return False
    return True

def crear_generadores_datos():
    """Crea generadores de datos con augmentación"""
    
    # Data augmentation para entrenamiento
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest',
        validation_split=VALIDATION_SPLIT
    )
    
    # Generador de datos de entrenamiento
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Generador de datos de validación (sin augmentación excesiva)
    val_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        validation_split=VALIDATION_SPLIT
    )
    
    val_generator = val_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator

def crear_modelo():
    """Crea un modelo CNN mejorado con Transfer Learning"""
    
    print("\n🔨 Construyendo modelo con Transfer Learning (MobileNetV2)...")
    
    # Cargar modelo preentrenado MobileNetV2
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Congelar capas base
    base_model.trainable = False
    
    # Crear modelo personalizado
    model = models.Sequential([
        base_model,
        
        # Capas de clasificación
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.4),
        
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Compilar modelo
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=METRICS
    )
    
    print("✓ Modelo construido exitosamente")
    model.summary()
    
    return model

def entrenar_modelo(model, train_gen, val_gen):
    """Entrena el modelo"""
    
    print("\n🚀 Iniciando entrenamiento...")
    
    # Callbacks
    callbacks = [
        # Early stopping para evitar overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Guardar mejor modelo
        ModelCheckpoint(
            'keras_model_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # Reducir learning rate si no hay mejoría
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Entrenar modelo
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        steps_per_epoch=len(train_gen),
        validation_steps=len(val_gen),
        workers=1,
        use_multiprocessing=False,
        max_queue_size=10
    )
    
    return history


def guardar_labels(class_names):
    """Guarda las etiquetas en un archivo para usar en la predicción"""
    with open(LABELS_FILE, 'w', encoding='utf-8') as f:
        for idx, clase in enumerate(class_names):
            f.write(f"{idx} {clase}\n")
    print(f"✓ Etiquetas guardadas en '{LABELS_FILE}'")


def descongelar_y_reentrenar(model, train_gen, val_gen):
    """Descongelar capas base y reentrenar con learning rate más bajo"""
    
    print("\n🔄 Descongelando capas base para fine-tuning...")
    
    # Descongelar últimas capas del modelo base
    for layer in model.layers[0].layers[-20:]:
        layer.trainable = True
    
    # Compilar con learning rate más bajo
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=METRICS
    )
    
    # Callbacks para fine-tuning
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('keras_model.h5', monitor='val_accuracy', save_best_only=True),
    ]
    
    # Reentrenar
    print("Reentrenant modelo con capas base desbloqueadas...")
    history_finetune = model.fit(
        train_gen,
        epochs=50,
        validation_data=val_gen,
        callbacks=callbacks,
        steps_per_epoch=len(train_gen),
        validation_steps=len(val_gen),
        workers=1,
        use_multiprocessing=False,
        max_queue_size=10
    )
    
    return history_finetune

def graficar_resultados(history, history_finetune=None):
    """Grafica los resultados del entrenamiento"""
    
    print("\n📊 Generando gráficos...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Combinar historiales
    if history_finetune:
        all_acc = history.history['accuracy'] + history_finetune.history['accuracy']
        all_val_acc = history.history['val_accuracy'] + history_finetune.history['val_accuracy']
        all_loss = history.history['loss'] + history_finetune.history['loss']
        all_val_loss = history.history['val_loss'] + history_finetune.history['val_loss']
    else:
        all_acc = history.history['accuracy']
        all_val_acc = history.history['val_accuracy']
        all_loss = history.history['loss']
        all_val_loss = history.history['val_loss']
    
    # Gráfico de precisión
    axes[0].plot(all_acc, label='Precisión Entrenamiento', linewidth=2)
    axes[0].plot(all_val_acc, label='Precisión Validación', linewidth=2)
    axes[0].set_title('Precisión del Modelo', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Precisión')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Gráfico de pérdida
    axes[1].plot(all_loss, label='Pérdida Entrenamiento', linewidth=2)
    axes[1].plot(all_val_loss, label='Pérdida Validación', linewidth=2)
    axes[1].set_title('Pérdida del Modelo', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Pérdida')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("✓ Gráfico guardado como 'training_history.png'")
    plt.show()

def main():
    """Función principal"""
    
    print("=" * 60)
    print("🦅 ENTRENAMIENTO DE CLASIFICADOR DE AVES - VERSIÓN MEJORADA")
    print("=" * 60)
    
    # Crear estructura de carpetas
    datos_existen = crear_directorio_datos()
    
    if not datos_existen:
        print("\n⏸️  Por favor, agrega imágenes de aves en las carpetas correspondientes.")
        print("    Luego ejecuta este script nuevamente.")
        return
    
    # Detectar carpetas de clase en DATA_DIR (esto admite acentos y nombres variados)
    detected_classes = [d for d in sorted(os.listdir(DATA_DIR)) if os.path.isdir(os.path.join(DATA_DIR, d))]
    if not detected_classes:
        print("\n❌ No se encontraron subcarpetas de clases en 'data/'. Por favor, agrega imágenes organizadas por carpeta.")
        return

    # Actualizar class_names y NUM_CLASSES dinámicamente según las carpetas encontradas
    global class_names, NUM_CLASSES
    class_names = detected_classes
    NUM_CLASSES = len(class_names)

    # Contar imágenes por clase
    total_images = 0
    for clase in class_names:
        clase_dir = os.path.join(DATA_DIR, clase)
        try:
            num_images = len([f for f in os.listdir(clase_dir) if os.path.isfile(os.path.join(clase_dir, f))])
        except FileNotFoundError:
            num_images = 0
        total_images += num_images
        print(f"  {clase}: {num_images} imágenes")

    if total_images == 0:
        print("\n❌ No se encontraron imágenes en las carpetas de clases. Por favor, agrega imágenes a las carpetas.")
        return

    print(f"\n✓ Total de imágenes: {total_images}")
    
    # Crear generadores
    train_gen, val_gen = crear_generadores_datos()
    
    # Crear modelo
    model = crear_modelo()
    
    # Entrenar
    history = entrenar_modelo(model, train_gen, val_gen)
    
    # Fine-tuning
    history_finetune = descongelar_y_reentrenar(model, train_gen, val_gen)
    
    # Guardar modelo final
    model.save('keras_model.h5')
    guardar_labels(class_names)
    print("\n💾 Modelo guardado como 'keras_model.h5'")
    
    # Graficar resultados
    graficar_resultados(history, history_finetune)
    
    # Evaluar en validación
    print("\n📈 EVALUACIÓN FINAL")
    results = model.evaluate(val_gen)
    metrics = dict(zip(model.metrics_names, results))
    print(f"  Precisión: {metrics.get('accuracy', 0) * 100:.2f}%")
    print(f"  Pérdida: {metrics.get('loss', 0):.4f}")
    if 'precision' in metrics:
        print(f"  Precision: {metrics['precision']:.4f}")
    if 'recall' in metrics:
        print(f"  Recall: {metrics['recall']:.4f}")
    
    print("\n✅ ¡Entrenamiento completado!")
    print("   Usa el modelo con: predict.py")

if __name__ == "__main__":
    main()
