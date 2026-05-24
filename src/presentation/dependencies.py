import os
import torch
from fastapi import Depends
from sqlalchemy.orm import Session
from src.infrastructure.database import get_db
from src.infrastructure.repositories import LocalModelRepository, PostgreSqlSettingsRepository
from src.infrastructure.services import MatplotlibPlotterService, PdfReportGenerator, OpenRouterLLMService
from src.application.use_cases import AnalyzeTelemetryUseCase, GetAIRecommendationUseCase, ManageSettingsUseCase

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_repository = LocalModelRepository(MODEL_DIR, DEVICE)
_plotter = MatplotlibPlotterService()
_report_generator = PdfReportGenerator()
_llm_service = OpenRouterLLMService()

_analyze_telemetry_use_case = AnalyzeTelemetryUseCase(_repository, _plotter)

def get_analyze_telemetry_use_case() -> AnalyzeTelemetryUseCase:
    return _analyze_telemetry_use_case

def get_report_generator() -> PdfReportGenerator:
    return _report_generator

def get_llm_service() -> OpenRouterLLMService:
    return _llm_service

def get_ai_recommendation_use_case(
    db: Session = Depends(get_db)
) -> GetAIRecommendationUseCase:
    settings_repo = PostgreSqlSettingsRepository(db)
    return GetAIRecommendationUseCase(settings_repo, _llm_service)

def get_manage_settings_use_case(
    db: Session = Depends(get_db)
) -> ManageSettingsUseCase:
    settings_repo = PostgreSqlSettingsRepository(db)
    return ManageSettingsUseCase(settings_repo)