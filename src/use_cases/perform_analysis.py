# src/use_cases/perform_analysis.py
import requests
from src.ml_core.analysis_service import HandwritingAnalysisService
from src.ports.trace_service_port import ITraceServicePort
from .dtos import AnalysisRequestDTO, AnalysisResponseDTO

class PerformAnalysisUseCase:
    """
    Caso de uso principal que orquesta el proceso completo de análisis.
    """
    def __init__(
        self,
        analysis_service: HandwritingAnalysisService,
        trace_service_adapter: ITraceServicePort,
    ):
        self.analysis_service = analysis_service
        self.trace_service_adapter = trace_service_adapter

    def execute(self, request: AnalysisRequestDTO) -> AnalysisResponseDTO:
        try:
            # 1. Descargar la imagen desde la URL proporcionada
            print(f"Descargando imagen desde: {request.image_url}")
            response = requests.get(request.image_url, timeout=10)
            response.raise_for_status()
            image_bytes = response.content
            
            # 2. Llamar al servicio de IA para obtener los resultados
            print("Iniciando análisis con el modelo de IA...")
            analysis_results = self.analysis_service.analyze_handwriting(
                image_bytes=image_bytes,
                template_char=request.template_char
            )
            print("Análisis de IA completado.")

            # 3. Notificar al TraceService con los resultados
            success = self.trace_service_adapter.notify_analysis_complete(
                practice_id=request.practice_id,
                analysis_data=analysis_results
            )

            if not success:
                raise RuntimeError("Falló la notificación al TraceService.")
                
            return AnalysisResponseDTO(
                practice_id=str(request.practice_id),
                status="COMPLETED",
                message="Análisis completado y notificado exitosamente."
            )

        except requests.exceptions.RequestException as e:
            print(f"Error al descargar la imagen: {e}")
            # Aquí podrías notificar al TraceService que hubo un error
            return AnalysisResponseDTO(practice_id=str(request.practice_id), status="ERROR", message="No se pudo descargar la imagen.")
        except Exception as e:
            print(f"Error durante el caso de uso de análisis: {e}")
            # Aquí también podrías notificar un error
            return AnalysisResponseDTO(practice_id=str(request.practice_id), status="ERROR", message=f"Ocurrió un error inesperado: {e}")