import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import random
import matplotlib.pyplot as plt

# Importamos la función de preprocesamiento del script de entrenamiento
# (Asegúrate de que este archivo esté en la misma carpeta que train_siamese.py o ajusta la ruta)
from train_siamese import preprocess_emnist_image

# --- CONFIGURACIÓN ---
MODEL_PATH = "ml_models/base_handwriting_model.h5"
NUM_TEST_PAIRS = 5

# --- CARGAR MODELO Y DATOS ---
print("Cargando el modelo base entrenado...")
base_model = tf.keras.models.load_model(MODEL_PATH)
print("Modelo cargado exitosamente.")

print("Cargando dataset de prueba de EMNIST...")
(ds_test), ds_info = tfds.load(
    'emnist/balanced',
    split=['test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Preprocesar y convertir a NumPy
ds_test = ds_test[0].map(preprocess_emnist_image)
x_test, y_test = [], []
for img, lbl in ds_test:
    x_test.append(img.numpy())
    y_test.append(lbl.numpy())
x_test = np.array(x_test)
y_test = np.array(y_test)

# Mapeo de etiquetas a caracteres para una mejor visualización
# Según la documentación de EMNIST/balanced
label_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"

# --- FUNCIÓN DE PRUEBA ---
def test_similarity():
    for _ in range(NUM_TEST_PAIRS):
        print("\n" + "="*40)
        
        # 1. Seleccionar una imagen de anclaje (anchor)
        anchor_idx = random.randint(0, len(x_test) - 1)
        anchor_image = x_test[anchor_idx]
        anchor_label_idx = y_test[anchor_idx]
        anchor_label_char = label_map[anchor_label_idx]

        # 2. Encontrar una imagen positiva (misma clase)
        positive_indices = np.where(y_test == anchor_label_idx)[0]
        positive_idx = random.choice(positive_indices)
        positive_image = x_test[positive_idx]

        # 3. Encontrar una imagen negativa (diferente clase)
        negative_indices = np.where(y_test != anchor_label_idx)[0]
        negative_idx = random.choice(negative_indices)
        negative_image = x_test[negative_idx]
        negative_label_char = label_map[y_test[negative_idx]]

        # 4. Predecir los embeddings
        anchor_embedding = base_model.predict(np.expand_dims(anchor_image, axis=0))
        positive_embedding = base_model.predict(np.expand_dims(positive_image, axis=0))
        negative_embedding = base_model.predict(np.expand_dims(negative_image, axis=0))

        # 5. Calcular distancias
        positive_distance = np.linalg.norm(anchor_embedding - positive_embedding)
        negative_distance = np.linalg.norm(anchor_embedding - negative_embedding)

        print(f"Anclaje: Letra '{anchor_label_char}'")
        print(f"Par Positivo (otra '{anchor_label_char}'): Distancia = {positive_distance:.4f}")
        print(f"Par Negativo (letra '{negative_label_char}'): Distancia = {negative_distance:.4f}")

        if positive_distance < negative_distance:
            print("✅ PRUEBA SUPERADA: El modelo agrupa correctamente las imágenes similares.")
        else:
            print("❌ PRUEBA FALLIDA: El modelo está confundido.")

        # Visualizar
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
        ax1.imshow(anchor_image.squeeze(), cmap='gray')
        ax1.set_title(f"Anclaje: '{anchor_label_char}'")
        ax2.imshow(positive_image.squeeze(), cmap='gray')
        ax2.set_title(f"Positivo: '{anchor_label_char}'\nDist: {positive_distance:.2f}")
        ax3.imshow(negative_image.squeeze(), cmap='gray')
        ax3.set_title(f"Negativo: '{negative_label_char}'\nDist: {negative_distance:.2f}")
        for ax in [ax1, ax2, ax3]: ax.axis('off')
        plt.show()

if __name__ == "__main__":
    test_similarity()