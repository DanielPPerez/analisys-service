# src/adapters/api/analysis_routes.py
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from src.use_cases.dtos import AnalysisRequestDTO, AnalysisResponseDTO
from src.use_cases.perform_analysis import PerformAnalysisUseCase

# Importación de nuestras dependencias
from src.ml_core.analysis_service import HandwritingAnalysisService
from src.adapters.trace_service_adapter import TraceServiceAdapter

router = APIRouter(prefix="/analysis", tags=["Análisis de Caligrafía"])

# --- Inyección de Dependencias (Singleton para el modelo de IA) ---
# Creamos una única instancia del servicio de análisis para que el modelo de ML
# se cargue en memoria solo una vez al iniciar la aplicación.
handwriting_service_singleton = HandwritingAnalysisService()
trace_service_adapter_singleton = TraceServiceAdapter()

def get_perform_analysis_use_case() -> PerformAnalysisUseCase:
    return PerformAnalysisUseCase(
        analysis_service=handwriting_service_singleton,
        trace_service_adapter=trace_service_adapter_singleton
    )

# --- Endpoints ---

@router.post("/perform", response_model=AnalysisResponseDTO, status_code=status.HTTP_202_ACCEPTED)
async def perform_analysis(
    request: AnalysisRequestDTO,
    background_tasks: BackgroundTasks,
    use_case: PerformAnalysisUseCase = Depends(get_perform_analysis_use_case)
):
    """
    Recibe una solicitud de análisis, la procesa en segundo plano y responde inmediatamente.
    """
    print(f"Recibida solicitud de análisis para practice_id: {request.practice_id}")
    
    # El análisis de IA puede tardar. Lo ejecutamos como una tarea en segundo plano
    # para no bloquear la respuesta HTTP. El cliente recibe un 202 Aceptado
    # y nuestro servicio trabaja por detrás.
    background_tasks.add_task(use_case.execute, request)
    
    return AnalysisResponseDTO(
        practice_id=str(request.practice_id),
        status="QUEUED",
        message="La solicitud de análisis ha sido aceptada y está en proceso."
    )