# src/ml_core/models/siamese_model.py
"""
Definición de la arquitectura de la red siamesa.
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Lambda
from tensorflow.keras import backend as K
from typing import Tuple


def build_base_network(input_shape: Tuple[int, int, int] = (128, 128, 1)):
    """
    Construye la red CNN base que procesa cada imagen para generar un embedding.
    
    Esta es la red que se guarda y se usa en producción para generar embeddings.
    
    Args:
        input_shape: Forma de las imágenes de entrada (altura, ancho, canales)
        
    Returns:
        Modelo Keras de la red base
    """
    input_layer = Input(shape=input_shape, name="base_input")
    
    # Primera capa convolucional
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D()(x)
    
    # Segunda capa convolucional
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)
    
    # Tercera capa convolucional
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)
    
    # Aplanar y capas densas
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Capa de embedding final (sin activación para permitir distancias euclidianas)
    embedding = Dense(256, activation=None, name="embedding")(x)
    
    return Model(input_layer, embedding, name="base_network")


def euclidean_distance(vectors):
    """
    Calcula la distancia euclidiana entre dos vectores de embedding.
    
    Args:
        vectors: Lista de dos tensores [vector1, vector2]
        
    Returns:
        Tensor con las distancias euclidianas
    """
    x, y = vectors
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def build_siamese_model(input_shape: Tuple[int, int, int] = (128, 128, 1)):
    """
    Construye el modelo siamés completo que toma dos imágenes y calcula su distancia.
    
    Este modelo se usa solo durante el entrenamiento.
    
    Args:
        input_shape: Forma de las imágenes de entrada (altura, ancho, canales)
        
    Returns:
        Modelo Keras del modelo siamés completo
    """
    base_network = build_base_network(input_shape)
    
    # Dos entradas para el par de imágenes
    input_a = Input(shape=input_shape, name="input_a")
    input_b = Input(shape=input_shape, name="input_b")
    
    # Procesar ambas imágenes con la misma red base (pesos compartidos)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    # Calcular distancia euclidiana
    distance = Lambda(euclidean_distance, name="distance")([processed_a, processed_b])
    
    # Crear modelo siamés
    siamese_model = Model([input_a, input_b], distance, name="siamese_model")
    
    return siamese_model, base_network

