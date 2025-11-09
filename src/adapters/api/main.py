# src/adapters/api/main.py
from fastapi import FastAPI, status

# Importamos el router que contiene nuestros endpoints de análisis
from src.adapters.api import analysis_routes

# --- Creación de la Aplicación Principal FastAPI ---
app = FastAPI(
    title="Servicio de Análisis de IA - Scriptoria AI",
    description="Microservicio para analizar la caligrafía usando un modelo de Red Siamesa.",
    version="1.0.0"
)

# --- Inclusión de Rutas ---
app.include_router(analysis_routes.router)

# --- Endpoints de Nivel de Aplicación ---
@app.get("/health", status_code=status.HTTP_200_OK, tags=["Monitoring"])
def health_check():
    """
    Verifica que el servicio esté funcionando correctamente.
    """
    return {"status": "ok", "service": "AnalysisService"}