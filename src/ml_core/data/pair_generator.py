# src/ml_core/data/pair_generator.py
"""
Generador de pares para entrenamiento de redes siamesas.
Crea pares positivos (misma clase) y negativos (diferentes clases) sobre la marcha.
"""
import tensorflow as tf
from typing import Tuple


class SiamesePairGenerator:
    """
    Genera pares de imágenes para entrenamiento de redes siamesas.
    Usa tf.data.Dataset para generar pares de forma eficiente sin cargar todo a memoria.
    """
    
    def __init__(self, num_classes: int, buffer_size: int = 10000):
        """
        Args:
            num_classes: Número de clases en el dataset
            buffer_size: Tamaño del buffer para mezclar
        """
        self.num_classes = num_classes
        self.buffer_size = buffer_size
    
    def create_pairs_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Crea un dataset de pares a partir de un dataset de imágenes.
        
        Por cada imagen en el batch, crea:
        - 1 par positivo (misma clase)
        - 1 par negativo (diferente clase)
        
        Args:
            dataset: Dataset de TensorFlow con batches de (images, labels)
            
        Returns:
            Dataset con estructura ((image1_batch, image2_batch), pair_labels)
            donde pair_labels: 1 = par positivo, 0 = par negativo
        """
        def create_pairs_batch(images, labels):
            """
            Crea pares para un batch de imágenes usando una estrategia simplificada.
            """
            batch_size = tf.shape(images)[0]
            batch_size_int32 = tf.cast(batch_size, tf.int32)
            
            # Convertir labels a int32
            labels = tf.cast(labels, tf.int32)
            indices = tf.range(batch_size_int32, dtype=tf.int32)
            
            # Crear matriz de comparación de etiquetas
            labels_i = tf.expand_dims(labels, 1)  # [batch, 1]
            labels_j = tf.expand_dims(labels, 0)  # [1, batch]
            same_class_matrix = tf.equal(labels_i, labels_j)  # [batch, batch]
            
            # Para pares positivos: crear máscara sin diagonal y usar argmax
            eye = tf.eye(batch_size_int32, dtype=tf.bool)
            same_no_diag = tf.logical_and(same_class_matrix, tf.logical_not(eye))
            
            # Convertir a float para usar con argmax
            # Asignar valores: True -> 1.0, False -> -1.0
            same_float = tf.cast(same_no_diag, tf.float32) * 2.0 - 1.0
            # Ahora True es 1.0 y False es -1.0
            
            # Para cada fila, encontrar el primer índice con valor positivo
            # Si todos son negativos, argmax devuelve el primer índice (0)
            # Necesitamos detectar si realmente hay un match
            max_vals = tf.reduce_max(same_float, axis=1)  # Valor máximo por fila
            argmax_indices = tf.argmax(same_float, axis=1, output_type=tf.int32)
            
            # Si max_val > 0, hay un match válido, usar argmax_indices
            # Si max_val <= 0, no hay match, usar el índice actual
            has_match = tf.greater(max_vals, 0.0)
            positive_indices = tf.where(has_match, argmax_indices, indices)
            positive_indices = tf.cast(positive_indices, tf.int32)
            
            # Para pares negativos: buscar imágenes con clase diferente
            different_matrix = tf.logical_not(same_class_matrix)
            different_float = tf.cast(different_matrix, tf.float32)
            
            # Encontrar primera imagen de clase diferente
            max_vals_neg = tf.reduce_max(different_float, axis=1)
            argmax_indices_neg = tf.argmax(different_float, axis=1, output_type=tf.int32)
            
            # Si hay clase diferente (max_val > 0), usar argmax
            # Si no hay (max_val == 0, todos de la misma clase), usar índice aleatorio
            has_diff = tf.greater(max_vals_neg, 0.0)
            random_fallback = tf.random.uniform(
                [batch_size_int32], 
                0, 
                batch_size_int32, 
                dtype=tf.int32
            )
            negative_indices = tf.where(has_diff, argmax_indices_neg, random_fallback)
            negative_indices = tf.cast(negative_indices, tf.int32)
            
            # Obtener imágenes
            positive_img2 = tf.gather(images, positive_indices)
            negative_img2 = tf.gather(images, negative_indices)
            
            # Concatenar
            image1_batch = tf.concat([images, images], axis=0)
            image2_batch = tf.concat([positive_img2, negative_img2], axis=0)
            pair_labels = tf.concat([
                tf.ones(batch_size, dtype=tf.float32),
                tf.zeros(batch_size, dtype=tf.float32)
            ], axis=0)
            
            return (image1_batch, image2_batch), pair_labels
        
        # Aplicar función
        paired_dataset = dataset.map(
            create_pairs_batch,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Mezclar
        paired_dataset = paired_dataset.shuffle(buffer_size=self.buffer_size)
        
        return paired_dataset
