# src/adapters/trace_service_adapter.py
import uuid
import requests
from typing import Dict, Any
from src.ports.trace_service_port import ITraceServicePort
from src.config import settings

class TraceServiceAdapter(ITraceServicePort):
    """
    Implementación (Adaptador) del puerto para comunicarse con el TraceService.
    """
    def notify_analysis_complete(self, practice_id: uuid.UUID, analysis_data: Dict[str, Any]) -> bool:
        """
        Realiza una llamada HTTP PUT al endpoint del TraceService.
        """
        # Construye la URL del endpoint específico
        url = f"{settings.trace_service_base_url}/practices/{practice_id}/analysis"
        
        try:
            print(f"Enviando resultados a TraceService en la URL: {url}")
            print(f"Datos: {analysis_data}")
            
            response = requests.put(url, json=analysis_data, timeout=10) # Timeout de 10 segundos
            
            # Lanza una excepción si la respuesta es un error HTTP (4xx o 5xx)
            response.raise_for_status()
            
            print(f"Notificación exitosa para practice_id {practice_id}. Estado: {response.status_code}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Error al notificar a TraceService para practice_id {practice_id}: {e}")
            return False