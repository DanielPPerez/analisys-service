# src/ml_core/siamese_network.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from .image_preprocessor import IMG_SIZE

def build_base_network(input_shape):
    """Construye la red convolucional que procesa cada imagen individualmente."""
    input = Input(shape=input_shape)
    x = Conv2D(64, (3,3), activation='relu')(input)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (3,3), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)

def euclidean_distance(vectors):
    """Calcula la distancia euclidiana entre los dos vectores de salida."""
    x, y = vectors
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))

def build_siamese_model(input_shape=(*IMG_SIZE, 1)):
    """Construye el modelo siamés completo."""
    # Define las dos entradas de la red
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # La red base se reutiliza para ambas entradas (pesos compartidos)
    base_network = build_base_network(input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    # Calcula la distancia entre las dos salidas
    distance = Lambda(euclidean_distance)([processed_a, processed_b])

    # Define el modelo con dos entradas y una salida
    model = Model([input_a, input_b], distance)
    return model

# En un escenario real, cargarías un modelo ya entrenado
def load_trained_model(path: str):
    print(f"Cargando modelo desde {path}...")
    # model = tf.keras.models.load_model(path, custom_objects={'euclidean_distance': euclidean_distance})
    # Por ahora, solo construiremos la arquitectura para simulación
    model = build_siamese_model()
    print("Modelo cargado (simulado).")
    return model