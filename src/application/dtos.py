from pydantic import BaseModel
from typing import List, Dict, Optional, Any

class SensorDiagnosticInfo(BaseModel):
    sensor_name: str
    soh: int
    final_rul: str
    current_error: float
    threshold: float
    survival_probabilities: List[float]

class TelemetryAnalysisResponse(BaseModel):
    filename: str
    model_type: str
    soh: int
    final_rul: str
    trend_graph: str
    t_graph: str
    h_graph: str
    dist_graph: str
    imp_graph: str
    cum_graph: str
    corr_graph: str
    adaptive_graph: str
    accel_graph: str
    holt_graph: str
    num_anomalies: int
    total_windows: int
    threshold: float
    culprit: str
    recommendation: str
    conf: str
    stream_data: str
    survival_timeline: List[int]
    global_survival_probabilities: List[float]
    survival_statement: str
    sensors_diagnostics: List[SensorDiagnosticInfo]