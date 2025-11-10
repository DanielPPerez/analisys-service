# src/ml_core/data/dataset_loader.py
"""
Módulo para cargar y preprocesar datasets.
"""
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Tuple, Dict, Any


class EMNISTDataLoader:
    """
    Carga y preprocesa el dataset EMNIST para entrenamiento.
    Usa tf.data.Dataset para manejar grandes volúmenes de datos sin cargar todo a memoria.
    """
    
    def __init__(self, img_size: Tuple[int, int] = (128, 128), batch_size: int = 64):
        """
        Args:
            img_size: Tamaño objetivo de las imágenes (altura, ancho)
            batch_size: Tamaño del batch para el dataset
        """
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_classes = None
        self.ds_info = None
    
    def preprocess_image(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Normaliza y redimensiona las imágenes del dataset EMNIST.
        EMNIST es 28x28, nuestro modelo necesita 128x128.
        
        Args:
            image: Imagen del dataset EMNIST
            label: Etiqueta de la imagen
            
        Returns:
            Tupla de (imagen procesada, etiqueta)
        """
        # EMNIST está rotado y transpuesto, lo corregimos
        # Asegurar que la imagen tenga la forma correcta
        image = tf.cast(image, tf.float32)
        
        # Asegurar que tenga dimensión de canal (EMNIST viene como [28, 28])
        # Verificar el rank de la imagen
        rank = tf.rank(image)
        image = tf.cond(
            tf.equal(rank, 2),
            lambda: tf.expand_dims(image, axis=-1),  # Agregar canal si es 2D
            lambda: image  # Ya tiene canal
        )
        
        # Transponer: EMNIST viene transpuesto, necesitamos corregirlo
        # Si la imagen es [H, W, C], transponer [1, 0, 2] intercambia H y W
        image = tf.transpose(image, [1, 0, 2])
        
        # Voltear horizontalmente (EMNIST está espejado)
        image = tf.image.flip_left_right(image)
        
        # Redimensionar a tamaño objetivo
        # tf.image.resize funciona con [H, W, C]
        image = tf.image.resize(image, self.img_size, method='bilinear')
        
        # Asegurar que tenga solo 1 canal (por si acaso)
        # Si tiene más canales, tomar solo el primero
        image_shape = tf.shape(image)
        num_channels = image_shape[2]
        image = tf.cond(
            tf.greater(num_channels, 1),
            lambda: image[:, :, 0:1],  # Tomar solo el primer canal
            lambda: image  # Ya tiene 1 canal
        )
        
        # Normalizar a [0, 1]
        image = image / 255.0
        
        return image, label
    
    def load_dataset(self, dataset_name: str = 'emnist/balanced') -> Tuple[Dict[str, tf.data.Dataset], Any]:
        """
        Carga el dataset EMNIST y lo preprocesa.
        
        Args:
            dataset_name: Nombre del dataset a cargar
            
        Returns:
            Tupla de (diccionario con splits, información del dataset)
        """
        print(f"Cargando y preprocesando el dataset {dataset_name}...")
        
        # Cargar dataset
        (ds_train, ds_test), ds_info = tfds.load(
            dataset_name,
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )
        
        self.ds_info = ds_info
        self.num_classes = ds_info.features['label'].num_classes
        print(f"Dataset cargado. Número de clases: {self.num_classes}")
        
        # Aplicar preprocesamiento
        ds_train = ds_train.map(
            self.preprocess_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        ds_test = ds_test.map(
            self.preprocess_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Mezclar antes de hacer batching (importante para crear pares diversos)
        # Nota: No cacheamos aquí porque los pares se generan dinámicamente después
        ds_train = ds_train.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
        ds_test = ds_test.shuffle(buffer_size=10000, reshuffle_each_iteration=False)
        
        # Hacer batching
        ds_train = ds_train.batch(self.batch_size)
        ds_test = ds_test.batch(self.batch_size)
        
        # Prefetch para mejorar rendimiento
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
        ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
        
        print(f"Dataset preparado con batch_size={self.batch_size}")
        
        return {
            'train': ds_train,
            'test': ds_test
        }, ds_info
    
    def get_num_classes(self) -> int:
        """Retorna el número de clases del dataset."""
        if self.num_classes is None:
            raise ValueError("Dataset no ha sido cargado aún. Llama a load_dataset() primero.")
        return self.num_classes

