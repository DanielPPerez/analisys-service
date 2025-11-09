# src/domain/value_objects/consejo.py
from pydantic import BaseModel

class Consejo(BaseModel):
    """
    Objeto de Valor que encapsula los consejos generados.
    """
    fortalezas: str
    areas_mejora: str