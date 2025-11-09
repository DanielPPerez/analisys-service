# src/use_cases/dtos.py
from pydantic import BaseModel
import uuid

# DTO para la petición que recibe este servicio (desde el bus de eventos o una API)
class AnalysisRequestDTO(BaseModel):
    practice_id: uuid.UUID
    image_url: str
    template_char: str

# DTO para la respuesta de este servicio
class AnalysisResponseDTO(BaseModel):
    practice_id: str
    status: str
    message: str