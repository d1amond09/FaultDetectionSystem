from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np

class IAnomalyModel(ABC):
    @abstractmethod
    def predict(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

class IModelRepository(ABC):
    @abstractmethod
    def load_neural_model(self, name: str, input_dim: int) -> IAnomalyModel:
        pass

    @abstractmethod
    def load_classical_model(self, name: str) -> Any:
        pass

    @abstractmethod
    def load_scaler(self) -> Any:
        pass

    @abstractmethod
    def load_valid_columns(self) -> List[str]:
        pass

    @abstractmethod
    def load_thresholds(self) -> Dict[str, float]:
        pass

    @abstractmethod
    def save_thresholds(self, thresholds: Dict[str, float]) -> None:
        pass

class ISettingsRepository(ABC):
    @abstractmethod
    def get_settings(self) -> Dict[str, str]:
        pass

    @abstractmethod
    def save_settings(self, settings: Dict[str, str]) -> None:
        pass

class IHistoryRepository(ABC):
    @abstractmethod
    def log_activity(self, user_id: str, action: str, details: str) -> None:
        pass

    @abstractmethod
    def log_analysis(self, user_id: str, summary: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def get_user_history(self, user_id: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_all_history(self) -> List[Dict[str, Any]]:
        pass

class ILLMService(ABC):
    @abstractmethod
    def analyze(self, telemetry_summary: Dict[str, Any], api_key: str, model_name: str) -> str:
        pass

class IPlotterService(ABC):
    @abstractmethod
    def create_trend_graph(self, x: np.ndarray, y: np.ndarray, trend: np.ndarray, threshold: float, steps: Any) -> str:
        pass

    @abstractmethod
    def create_simple_line_graph(self, data: np.ndarray, threshold: float) -> str:
        pass

    @abstractmethod
    def create_heatmap(self, data: np.ndarray, labels: List[str]) -> str:
        pass

    @abstractmethod
    def create_distribution_graph(self, data: np.ndarray) -> str:
        pass

    @abstractmethod
    def create_importance_graph(self, importance: np.ndarray, labels: List[str]) -> str:
        pass

    @abstractmethod
    def create_cumulative_graph(self, data: np.ndarray) -> str:
        pass

    @abstractmethod
    def create_correlation_graph(self, matrix: np.ndarray) -> str:
        pass

    @abstractmethod
    def create_adaptive_graph(self, data: np.ndarray, adaptive: np.ndarray, threshold: float) -> str:
        pass

    @abstractmethod
    def create_acceleration_graph(self, data: np.ndarray) -> str:
        pass

    @abstractmethod
    def create_holt_graph(self, data: np.ndarray, forecast: np.ndarray, adaptive_threshold: float) -> str:
        pass

class IReportGenerator(ABC):
    @abstractmethod
    def generate_pdf(self, summary: Dict[str, Any], graphs: List[str]) -> bytes:
        pass