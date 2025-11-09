# src/domain/value_objects/puntuacion.py
from pydantic import BaseModel, Field, conint

class Puntuacion(BaseModel):
    """
    Objeto de Valor que representa una puntuación validada entre 0 y 100.
    """
    valor: conint(ge=0, le=100) = Field(..., description="El valor de la puntuación, de 0 a 100.")