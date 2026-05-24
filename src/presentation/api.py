import os
import io
import json
import jwt
import threading
from fastapi import APIRouter, Depends, File, UploadFile, Form, Response, Header, HTTPException
from fastapi.responses import Response as FastApiResponse
from pydantic import BaseModel
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from src.infrastructure.database import get_db
from src.application.dtos import TelemetryAnalysisResponse
from src.application.use_cases import AnalyzeTelemetryUseCase, GetAIRecommendationUseCase, ManageSettingsUseCase
from src.infrastructure.services import PdfReportGenerator
from src.infrastructure.repositories import PostgreSqlSettingsRepository, PostgreSqlHistoryRepository
from src.presentation.dependencies import (
    get_analyze_telemetry_use_case, get_report_generator,
    get_ai_recommendation_use_case, get_manage_settings_use_case
)

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORAGE_DIR = os.path.join(BASE_DIR, 'storage')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DEVICE = "cpu"

training_log = []
training_status = "idle"


class SettingsSchema(BaseModel):
    api_key: str
    model_name: str


def get_current_user(authorization: str = Header(None)) -> Dict[str, Any]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Отсутствует или невалиден заголовок авторизации")
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, options={"verify_signature": False})
        user_id = payload.get("sub") or payload.get(
            "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/nameidentifier")
        roles_claim = payload.get("role") or payload.get("http://schemas.microsoft.com/ws/2008/06/identity/claims/role")
        roles = [roles_claim] if isinstance(roles_claim, str) else (roles_claim or [])
        return {"user_id": user_id, "roles": roles}
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Ошибка разбора сессионного токена: {str(e)}")


def check_admin(user: Dict[str, Any] = Depends(get_current_user)):
    if "Admin" not in user["roles"]:
        raise HTTPException(status_code=403, detail="Доступ запрещен. Требуются права администратора.")
    return user


@router.post("/predict", response_model=TelemetryAnalysisResponse)
def predict(
        file: UploadFile = File(...),
        model_type: str = Form(...),
        use_case: AnalyzeTelemetryUseCase = Depends(get_analyze_telemetry_use_case),
        user: Dict[str, Any] = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    try:
        content = file.file.read()
        result = use_case.execute(content, file.filename, model_type)

        history_repo = PostgreSqlHistoryRepository(db)
        history_repo.log_activity(user["user_id"], "Предиктивный анализ",
                                  f"Запуск анализа файла '{file.filename}' по модели '{model_type}' (Индекс здоровья SOH: {result.soh}%)")
        history_repo.log_analysis(user["user_id"], result.model_dump())

        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка обработки на сервере: {str(e)}")


@router.post("/export_pdf")
def export_pdf(
        filename: str = Form(...),
        soh: int = Form(...),
        final_rul: str = Form(...),
        total_windows: int = Form(...),
        num_anomalies: int = Form(...),
        threshold: float = Form(...),
        culprit: str = Form(...),
        recommendation: str = Form(...),
        conf: str = Form(...),
        b64_trend: str = Form(...),
        b64_t: str = Form(...),
        b64_h: str = Form(...),
        b64_d: str = Form(...),
        b64_i: str = Form(...),
        b64_c: str = Form(...),
        b64_corr: str = Form(...),
        b64_adaptive: str = Form(...),
        b64_accel: str = Form(...),
        report_generator: PdfReportGenerator = Depends(get_report_generator),
        user: Dict[str, Any] = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    summary = {
        "filename": filename,
        "soh": soh,
        "final_rul": final_rul,
        "total_windows": total_windows,
        "num_anomalies": num_anomalies,
        "threshold": threshold,
        "culprit": culprit,
        "recommendation": recommendation,
        "conf": conf
    }
    graphs = [b64_trend, b64_t, b64_h, b64_d, b64_i, b64_c, b64_corr, b64_adaptive, b64_accel]
    pdf_bytes = report_generator.generate_pdf(summary, graphs)

    history_repo = PostgreSqlHistoryRepository(db)
    history_repo.log_activity(user["user_id"], "Экспорт отчета",
                              f"Экспорт детального диагностического PDF-отчета для файла '{filename}'")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=Detailed_Analysis_{filename}.pdf"}
    )


@router.get("/settings", response_model=SettingsSchema)
def get_settings(
        db: Session = Depends(get_db),
        user: Dict[str, Any] = Depends(check_admin)
):
    repo = PostgreSqlSettingsRepository(db)
    data = repo.get_settings()
    return SettingsSchema(api_key=data.get("api_key", ""), model_name=data.get("model_name", ""))


@router.post("/settings")
def save_settings(
        payload: SettingsSchema,
        db: Session = Depends(get_db),
        user: Dict[str, Any] = Depends(check_admin)
):
    repo = PostgreSqlSettingsRepository(db)
    repo.save_settings({"api_key": payload.api_key, "model_name": payload.model_name})

    history_repo = PostgreSqlHistoryRepository(db)
    history_repo.log_activity(user["user_id"], "Изменение настроек ИИ",
                              f"Обновление параметров OpenRouter. Модель: '{payload.model_name}'")

    return {"status": "success"}


@router.post("/ai-analysis")
def ai_analysis(
        payload: Dict[str, Any],
        use_case: GetAIRecommendationUseCase = Depends(get_ai_recommendation_use_case),
        user: Dict[str, Any] = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    result = use_case.execute(payload)

    history_repo = PostgreSqlHistoryRepository(db)
    history_repo.log_activity(user["user_id"], "Инференс ИИ-Анализа",
                              f"Получена инженерно-экспертная сводка от LLM для датчика {payload.get('culprit')}")

    return {"recommendation": result}


@router.post("/start_training")
def start_training(
        files: List[UploadFile] = File(...),
        model_choice: str = Form('all'),
        user: Dict[str, Any] = Depends(check_admin),
        db: Session = Depends(get_db)
):
    global training_log, training_status
    saved_paths = []
    uploads_dir = os.path.join(STORAGE_DIR, 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)

    for f in files:
        path = os.path.join(uploads_dir, f.filename)
        with open(path, "wb") as f_out:
            f_out.write(f.file.read())
        saved_paths.append(path)

    history_repo = PostgreSqlHistoryRepository(db)
    history_repo.log_activity(user["user_id"], "Старт полного обучения",
                              f"Инициализация обучения группы моделей: '{model_choice}'")

    def threaded_training():
        global training_log, training_status
        training_log = ["Начало обучения..."]
        training_status = "running"
        try:
            from src.train_utils import run_full_training
            run_full_training(saved_paths, model_choice, log_callback=lambda msg: training_log.append(msg))
            training_status = "done"
        except Exception as e:
            training_status = "error"
            training_log.append(f"ОШИБКА: {str(e)}")

    thread = threading.Thread(target=threaded_training)
    thread.start()
    return {"status": "started"}


@router.get("/train_status")
def get_train_status(user: Dict[str, Any] = Depends(check_admin)):
    global training_log, training_status
    return {
        "training_log": training_log,
        "training_status": training_status
    }


@router.post("/retrain")
def retrain(
        model_type: str = Form(...),
        user: Dict[str, Any] = Depends(check_admin),
        db: Session = Depends(get_db)
):
    f_path = os.path.join(STORAGE_DIR, 'uploads', 'last.csv')
    if not os.path.exists(f_path):
        return {"error": "Нет данных для адаптации"}
    try:
        import joblib
        import pandas as pd
        import numpy as np
        from src.infrastructure.pytorch_models import LSTMAutoencoder

        valid_cols = joblib.load(os.path.join(MODEL_DIR, 'valid_cols.pkl'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        thresholds = json.load(open(os.path.join(MODEL_DIR, 'thresholds.json')))

        df = pd.read_csv(f_path)
        df.drop(columns=['Столбец_0', 'Столбец_1'], inplace=True, errors='ignore')
        df_c = df[valid_cols].ffill().fillna(0).astype(np.float32)
        scaled = np.clip(scaler.transform(df_c), 0, 1)
        X = np.array([scaled[i:i + 10] for i in range(len(scaled) - 10)])
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)

        model = LSTMAutoencoder(len(valid_cols)).to(DEVICE)
        model.load_state_dict(
            torch.load(os.path.join(MODEL_DIR, f'{model_type}.pth'), map_location=DEVICE, weights_only=True))

        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-5)
        crit = nn.HuberLoss()
        for _ in range(3):
            opt.zero_grad()
            loss = crit(model(X_tensor), X_tensor)
            loss.backward()
            opt.step()
        model.eval()
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'{model_type}.pth'))

        with torch.no_grad():
            errs = torch.mean(torch.abs(model(X_tensor) - X_tensor), dim=(1, 2)).cpu().numpy()
        smooth = pd.Series(errs).rolling(10, min_periods=1).mean().values
        new_th = float(np.percentile(smooth, 95))
        thresholds[model_type] = new_th
        with open(os.path.join(MODEL_DIR, 'thresholds.json'), 'w') as f:
            json.dump(thresholds, f, indent=2)

        history_repo = PostgreSqlHistoryRepository(db)
        history_repo.log_activity(user["user_id"], "Адаптация модели",
                                  f"Локальное дообучение архитектуры '{model_type}'. Смещение порога детекции к значению: {new_th:.4f}")

        return {"info": f"Модель {model_type} адаптирована. Новый порог: {new_th:.4f}", "model_type": model_type}
    except Exception as e:
        return {"error": str(e)}


@router.get("/api/users/{id_str}/history")
def get_user_history(
        id_str: str,
        db: Session = Depends(get_db),
        user: Dict[str, Any] = Depends(check_admin)
):
    history_repo = PostgreSqlHistoryRepository(db)
    return history_repo.get_user_history(id_str)


@router.get("/api/history")
def get_all_history(
        db: Session = Depends(get_db),
        user: Dict[str, Any] = Depends(check_admin)
):
    history_repo = PostgreSqlHistoryRepository(db)
    return history_repo.get_all_history()