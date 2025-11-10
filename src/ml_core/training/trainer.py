# src/ml_core/training/trainer.py
"""
Módulo de entrenamiento para la red siamesa.
"""
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import Dict, Any, Optional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from ..models.siamese_model import build_siamese_model
from ..models.losses import contrastive_loss


class SiameseTrainer:
    """
    Clase para entrenar el modelo siamés.
    """
    
    def __init__(
        self,
        img_shape: tuple = (128, 128, 1),
        batch_size: int = 64,
        epochs: int = 15,
        model_save_dir: str = "ml_models",
        model_save_path: Optional[str] = None
    ):
        """
        Args:
            img_shape: Forma de las imágenes de entrada
            batch_size: Tamaño del batch para entrenamiento
            epochs: Número de épocas
            model_save_dir: Directorio para guardar el modelo
            model_save_path: Ruta completa para guardar el modelo (si None, usa model_save_dir/base_handwriting_model.h5)
        """
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_save_dir = model_save_dir
        self.model_save_path = model_save_path or os.path.join(model_save_dir, "base_handwriting_model.h5")
        
        # Crear directorio si no existe
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Modelos (se crean en train)
        self.siamese_model = None
        self.base_network = None
        self.history = None
    
    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Entrena el modelo siamés.
        
        Args:
            train_dataset: Dataset de entrenamiento con pares ((image1, image2), labels)
            val_dataset: Dataset de validación con pares ((image1, image2), labels)
            verbose: Verbosidad del entrenamiento (0, 1, o 2)
            
        Returns:
            Diccionario con el historial de entrenamiento
        """
        print("\n--- CONSTRUYENDO MODELO ---")
        self.siamese_model, self.base_network = build_siamese_model(self.img_shape)
        
        # Crear wrapper para la función de pérdida con margen
        def contrastive_loss_wrapper(y_true, y_pred):
            return contrastive_loss(y_true, y_pred, margin=1.0)
        
        # Compilar el modelo
        self.siamese_model.compile(
            loss=contrastive_loss_wrapper,
            optimizer="adam",
            metrics=["accuracy"]
        )
        
        self.siamese_model.summary()
        
        # Callbacks para mejorar el entrenamiento
        callbacks = [
            ModelCheckpoint(
                filepath=self.model_save_path.replace('.h5', '_checkpoint.h5'),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        print("\n--- INICIANDO ENTRENAMIENTO ---")
        self.history = self.siamese_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Guardar el modelo base final (el que se usa en producción)
        print(f"\nEntrenamiento completado. Guardando la red base en: {self.model_save_path}")
        self.base_network.save(self.model_save_path)
        print("¡Modelo guardado exitosamente!")
        
        return self.history.history
    
    def plot_training_history(self, save_path: Optional[str] = None, show: bool = True):
        """
        Visualiza el historial de entrenamiento.
        
        Args:
            save_path: Ruta para guardar el gráfico (opcional)
            show: Si mostrar el gráfico
        """
        if self.history is None:
            raise ValueError("El modelo no ha sido entrenado aún. Llama a train() primero.")
        
        plt.figure(figsize=(12, 4))
        
        # Gráfico de pérdida
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Loss del Entrenamiento')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Gráfico de accuracy
        plt.subplot(1, 2, 2)
        if 'accuracy' in self.history.history:
            plt.plot(self.history.history['accuracy'], label='Training Accuracy')
            plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Accuracy del Entrenamiento')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Gráfico guardado en: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()

