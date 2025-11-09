# src/ports/trace_service_port.py
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any

class ITraceServicePort(ABC):
    """
    Interfaz (Puerto) que define cómo notificar los resultados de un análisis
    al servicio de trazos.
    """
    @abstractmethod
    def notify_analysis_complete(self, practice_id: uuid.UUID, analysis_data: Dict[str, Any]) -> bool:
        """
        Envía los datos del análisis al TraceService.

        Args:
            practice_id: El ID de la práctica que fue analizada.
            analysis_data: Un diccionario con los resultados del análisis.

        Returns:
            True si la notificación fue exitosa, False en caso contrario.
        """
        pass