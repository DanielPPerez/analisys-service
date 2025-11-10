# train_siamese.py
"""
Script principal para entrenar la red siamesa.
Refactorizado para seguir arquitectura limpia.
"""
import os
import tensorflow as tf

from src.ml_core.data import EMNISTDataLoader, SiamesePairGenerator
from src.ml_core.training import SiameseTrainer

# --- CONFIGURACIÓN ---
print("TensorFlow Version:", tf.__version__)

IMG_SIZE = (128, 128)
IMG_SHAPE = (128, 128, 1)
BATCH_SIZE = 64
EPOCHS = 15
MODEL_SAVE_DIR = "ml_models"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "base_handwriting_model.h5")


def main():
    """
    Función principal que orquesta el entrenamiento del modelo siamés.
    """
    # 1. Cargar dataset
    print("\n=== PASO 1: CARGANDO DATASET ===")
    data_loader = EMNISTDataLoader(img_size=IMG_SIZE, batch_size=BATCH_SIZE)
    datasets, ds_info = data_loader.load_dataset('emnist/balanced')
    
    train_dataset = datasets['train']
    test_dataset = datasets['test']
    num_classes = data_loader.get_num_classes()
    
    print(f"Dataset cargado exitosamente. Número de clases: {num_classes}")
    
    # 2. Crear generador de pares
    print("\n=== PASO 2: CREANDO PARES DE IMÁGENES ===")
    pair_generator = SiamesePairGenerator(num_classes=num_classes, buffer_size=10000)
    
    # Crear datasets de pares para entrenamiento y validación
    train_pairs_dataset = pair_generator.create_pairs_dataset(train_dataset)
    test_pairs_dataset = pair_generator.create_pairs_dataset(test_dataset)
    
    print("Pares creados exitosamente.")
    print("Nota: Cada batch de imágenes genera 2x pares (positivos y negativos).")
    
    # 3. Entrenar modelo
    print("\n=== PASO 3: ENTRENANDO MODELO ===")
    trainer = SiameseTrainer(
        img_shape=IMG_SHAPE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        model_save_dir=MODEL_SAVE_DIR,
        model_save_path=MODEL_SAVE_PATH
    )
    
    # Entrenar
    history = trainer.train(
        train_dataset=train_pairs_dataset,
        val_dataset=test_pairs_dataset,
        verbose=1
    )
    
    # 4. Visualizar resultados
    print("\n=== PASO 4: VISUALIZANDO RESULTADOS ===")
    trainer.plot_training_history(
        save_path='training_loss.png',
        show=True
    )
    
    print("\n=== ENTRENAMIENTO COMPLETADO ===")
    print(f"Modelo guardado en: {MODEL_SAVE_PATH}")
    print("El modelo está listo para ser usado por analysis_service.py")


if __name__ == "__main__":
    main()
