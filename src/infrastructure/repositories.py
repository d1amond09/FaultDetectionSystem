import os
import json
import joblib
import torch
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from src.domain.interfaces import IModelRepository, IAnomalyModel, ISettingsRepository, IHistoryRepository
from src.infrastructure.database import LlmSetting, UserActivityLog, TelemetryAnalysisRecord
from src.infrastructure.pytorch_models import (
    LSTMAutoencoder, GRUAutoencoder, Conv1DAutoencoder,
    VAEAutoencoder, TransformerAutoencoder, TCNAutoencoder, NeuralModelAdapter
)

class LocalModelRepository(IModelRepository):
    def __init__(self, model_dir: str, device: str):
        self._model_dir = model_dir
        self._device = torch.device(device)
        self._neural_classes = {
            'lstm': LSTMAutoencoder,
            'gru': GRUAutoencoder,
            'cnn': Conv1DAutoencoder,
            'vae': VAEAutoencoder,
            'transformer': TransformerAutoencoder,
            'tcn': TCNAutoencoder
        }

    def load_neural_model(self, name: str, input_dim: int) -> IAnomalyModel:
        cls = self._neural_classes[name]
        module = cls(input_dim)
        path = os.path.join(self._model_dir, f"{name}.pth")
        module.load_state_dict(torch.load(path, map_location=self._device, weights_only=True))
        return NeuralModelAdapter(module, self._device)

    def load_classical_model(self, name: str) -> Any:
        return joblib.load(os.path.join(self._model_dir, f"{name}.pkl"))

    def load_scaler(self) -> Any:
        return joblib.load(os.path.join(self._model_dir, "scaler.pkl"))

    def load_valid_columns(self) -> List[str]:
        return joblib.load(os.path.join(self._model_dir, "valid_cols.pkl"))

    def load_thresholds(self) -> Dict[str, float]:
        with open(os.path.join(self._model_dir, "thresholds.json"), "r") as f:
            return json.load(f)

    def save_thresholds(self, thresholds: Dict[str, float]) -> None:
        with open(os.path.join(self._model_dir, "thresholds.json"), "w") as f:
            json.dump(thresholds, f, indent=2)

class PostgreSqlSettingsRepository(ISettingsRepository):
    def __init__(self, db_session: Session):
        self._db = db_session

    def get_settings(self) -> Dict[str, str]:
        record = self._db.query(LlmSetting).first()
        if not record:
            return {"api_key": "", "model_name": "openai/gpt-latest"}
        return {"api_key": record.api_key, "model_name": record.model_name}

    def save_settings(self, settings: Dict[str, str]) -> None:
        record = self._db.query(LlmSetting).first()
        if not record:
            record = LlmSetting(api_key=settings.get("api_key", ""), model_name=settings.get("model_name", "openai/gpt-latest"))
            self._db.add(record)
        else:
            record.api_key = settings.get("api_key", "")
            record.model_name = settings.get("model_name", "")
        self._db.commit()

class PostgreSqlHistoryRepository(IHistoryRepository):
    def __init__(self, db_session: Session):
        self._db = db_session

    def log_activity(self, user_id: str, action: str, details: str) -> None:
        log = UserActivityLog(user_id=user_id, action=action, details=details)
        self._db.add(log)
        self._db.commit()

    def log_analysis(self, user_id: str, summary: Dict[str, Any]) -> None:
        record = TelemetryAnalysisRecord(
            user_id=user_id,
            filename=summary.get("filename", ""),
            model_type=summary.get("model_type", ""),
            soh=int(summary.get("soh", 0)),
            final_rul=summary.get("final_rul", ""),
            culprit=summary.get("culprit", ""),
            recommendation=summary.get("recommendation", ""),
            confidence=summary.get("conf", "")
        )
        self._db.add(record)
        self._db.commit()

    def get_user_history(self, user_id: str) -> List[Dict[str, Any]]:
        logs = self._db.query(UserActivityLog).filter(UserActivityLog.user_id == user_id).order_by(UserActivityLog.created_at.desc()).all()
        return [
            {
                "date": log.created_at.strftime("%Y-%m-%d %H:%M"),
                "event": log.action,
                "details": log.details
            }
            for log in logs
        ]

    def get_all_history(self) -> List[Dict[str, Any]]:
        logs = self._db.query(UserActivityLog).order_by(UserActivityLog.created_at.desc()).all()
        return [
            {
                "user_id": log.user_id,
                "date": log.created_at.strftime("%Y-%m-%d %H:%M"),
                "event": log.action,
                "details": log.details
            }
            for log in logs
        ]