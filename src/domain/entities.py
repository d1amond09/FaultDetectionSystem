import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field

@dataclass(frozen=True)
@dataclass
class TelemetryData:
    columns: List[str]
    values: np.ndarray

    def validate(self) -> None:
        if self.values.size == 0:
            raise ValueError("Empty telemetry data")

@dataclass
class FailurePrediction:
    remaining_useful_life: str
    soh: int
    num_anomalies: int
    total_windows: int
    threshold: float
    culprit: str
    recommendation: str
    confidence: str
    stream_data: List[float]

@dataclass
class TrainingProgress:
    status: str
    logs: List[str] = field(default_factory=list)

    def add_log(self, message: str) -> None:
        self.logs.append(message)

    def set_status(self, status: str) -> None:
        self.status = status