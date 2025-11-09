import numpy as np
import tensorflow as tf
import os
import random
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Lambda
from tensorflow.keras import backend as K
from src.ml_core.image_preprocessor import preprocess_image

def create_pairs(image_paths, num_classes):
    # Lógica para crear pares positivos (misma letra) y negativos (diferentes letras)
    # ...
    pass # Esta es una implementación compleja, para empezar podemos usar un generador más simple