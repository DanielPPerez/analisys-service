# src/domain/entities/analisis_caligrafico.py
import uuid
import datetime
from pydantic import BaseModel, Field
from ..value_objects.puntuacion import Puntuacion
from ..value_objects.consejo import Consejo

class AnalisisCaligrafico(BaseModel):
    """
    Entidad Raíz que representa el resultado completo de un análisis.
    """
    analysis_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    practice_id: uuid.UUID
    puntuacion_general: Puntuacion
    puntuacion_proporcion: Puntuacion
    puntuacion_inclinacion: Puntuacion
    puntuacion_espaciado: Puntuacion
    puntuacion_consistencia: Puntuacion
    consejos: Consejo
    fecha_analisis: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)