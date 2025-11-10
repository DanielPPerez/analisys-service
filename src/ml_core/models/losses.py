# src/ml_core/models/losses.py
"""
Funciones de pérdida para entrenamiento de redes siamesas.
"""
import tensorflow as tf
from tensorflow.keras import backend as K


def contrastive_loss(y_true, y_pred, margin: float = 1.0):
    """
    Función de pérdida Contrastive para redes siamesas.
    
    Empuja los pares similares (y_true=1) a tener distancia 0
    y los pares diferentes (y_true=0) a tener una distancia mayor que `margin`.
    
    Args:
        y_true: Etiquetas verdaderas (1 para pares similares, 0 para diferentes)
        y_pred: Distancias predichas entre los embeddings
        margin: Margen para pares negativos (distancia mínima deseada)
        
    Returns:
        Valor de pérdida
    """
    y_true = tf.cast(y_true, tf.float32)
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

