# src/ml_core/analysis_service.py
import numpy as np
import tensorflow as tf
import cv2
import os

from .image_preprocessor import preprocess_image # Usamos nuestra función mejorada

class HandwritingAnalysisService:
    def __init__(self, model_path: str = "ml_models/base_handwriting_model.h5"):
        # Carga SOLO la red base entrenada
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"El modelo no se encontró en {model_path}. Asegúrate de entrenarlo y guardarlo.")
        self.base_model = tf.keras.models.load_model(model_path)
        print(f"Modelo base cargado desde {model_path}")
        
        # Carga las plantillas perfectas
        self.templates = self._load_templates("dataset/plantillas")

    def _load_templates(self, templates_dir: str):
        templates = {}
        if not os.path.isdir(templates_dir):
            print(f"ADVERTENCIA: El directorio de plantillas '{templates_dir}' no existe.")
            return {}
            
        for filename in os.listdir(templates_dir):
            char = filename.split('_')[0]
            path = os.path.join(templates_dir, filename)
            
            # Leer y preprocesar la plantilla
            with open(path, 'rb') as f:
                image_bytes = f.read()
            
            processed_template = preprocess_image(image_bytes)
            # Extraer su embedding y guardarlo para no recalcularlo cada vez
            templates[char] = self.base_model.predict(np.expand_dims(processed_template, axis=0))[0]
            
        print(f"Se cargaron y procesaron {len(templates)} plantillas.")
        return templates

    def _distance_to_score(self, distance: float, max_distance=15.0) -> int:
        # El valor de max_distance depende de tu espacio de embedding, se ajusta empíricamente
        similarity = max(0, 1 - (distance / max_distance))
        return int(similarity * 100)
    
    def _analizar_errores_cv(self, user_image_bytes):
        # Implementa aquí las funciones de análisis detallado (inclinación, etc.)
        # usando OpenCV como se describió en el plan.
        # Por ahora, devolvemos valores simulados.
        return {
            "inclinacion": 5, # en grados
            "proporcion_wh": 0.8 # width/height
        }

    def analyze_handwriting(self, image_bytes: bytes, template_char: str) -> dict:
        # 1. Obtener el embedding pre-calculado de la plantilla
        template_embedding = self.templates.get(template_char)
        if template_embedding is None:
            raise ValueError(f"No se encontró una plantilla para el caracter '{template_char}'.")

        # 2. Preprocesar la imagen del usuario
        user_img_processed = preprocess_image(image_bytes)

        # 3. Extraer el embedding de la imagen del usuario
        user_embedding = self.base_model.predict(np.expand_dims(user_img_processed, axis=0))[0]

        # 4. Calcular la distancia euclidiana entre los embeddings
        distance = np.linalg.norm(user_embedding - template_embedding)

        # 5. Convertir distancia a una puntuación global
        score_global = self._distance_to_score(float(distance))

        # 6. Realizar análisis detallado con Computer Vision
        detalles_cv = self._analizar_errores_cv(image_bytes)

        # 7. Generar feedback basado en reglas
        # Esta es una implementación simple, se puede hacer mucho más compleja
        fortalezas = "Buen intento, sigue practicando."
        areas_mejora = "Concéntrate en la forma general de la letra."
        if score_global > 85:
            fortalezas = "¡Excelente! La forma es muy similar a la plantilla."
        if detalles_cv['inclinacion'] > 10:
            areas_mejora = "Intenta mantener la letra un poco más vertical."

        return {
            "puntuacion_general": score_global,
            "puntuacion_proporcion": 80, # Simulado
            "puntuacion_inclinacion": 90, # Simulado
            "puntuacion_espaciado": 75, # Simulado
            "puntuacion_consistencia": 85, # Simulado
            "fortalezas": fortalezas,
            "areas_mejora": areas_mejora
        }