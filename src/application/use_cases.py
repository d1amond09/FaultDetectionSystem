import io
import numpy as np
import pandas as pd
from typing import List, Any, Dict, Tuple
from sklearn.linear_model import LinearRegression
from src.domain.interfaces import IModelRepository, IPlotterService, IReportGenerator, ISettingsRepository, ILLMService
from src.domain.services import RulEstimator, RuleEngine
from src.domain.entities import TelemetryData
from src.application.dtos import TelemetryAnalysisResponse, SensorDiagnosticInfo


class AnalyzeTelemetryUseCase:
    def __init__(
            self,
            repository: IModelRepository,
            plotter: IPlotterService,
    ):
        self._repository = repository
        self._plotter = plotter

    def _calculate_weibull_survival(
            self,
            errors_series: np.ndarray,
            threshold: float,
            avg_interval_sec: float
    ) -> Tuple[List[int], List[float], str]:
        window_len = min(len(errors_series), 100)
        x_trend = np.arange(window_len).reshape(-1, 1)
        y_trend = errors_series[-window_len:]

        lr = LinearRegression().fit(x_trend, y_trend)
        slope = float(lr.coef_[0])
        current_val = float(y_trend[-1])

        rul_hours = 720.0
        if current_val >= threshold:
            rul_hours = 0.0
        elif slope > 1e-6:
            steps = (threshold - current_val) / slope
            rul_hours = (steps * avg_interval_sec) / 3600.0

        accel_arr = np.gradient(np.gradient(errors_series))
        mean_accel = float(np.mean(accel_arr[-10:])) if len(accel_arr) >= 10 else 0.0

        beta = 1.0
        if mean_accel > 1e-6:
            beta = 1.5 + 2.0 * min(1.0, mean_accel * 1000.0)

        eta = rul_hours / (np.log(2) ** (1.0 / beta)) if rul_hours > 0 else 720.0

        timeline = [0, 12, 24, 36, 48, 72, 96, 120, 144, 168]
        probabilities = []

        for t in timeline:
            if rul_hours == 0.0:
                prob = 0.0 if t > 0 else 1.0
            else:
                prob = float(np.exp(-((t / eta) ** beta)))
            probabilities.append(prob)

        failure_80_hour = None
        for t_val, prob_val in zip(timeline, probabilities):
            if prob_val <= 0.8:
                failure_80_hour = t_val
                break

        if failure_80_hour is not None:
            statement = f"С вероятностью 80% отказ произойдет в течение следующих {failure_80_hour} часов."
        else:
            statement = "Вероятность критического отказа в течение следующих 7 дней оценивается ниже 20%."

        return timeline, probabilities, statement

    def execute(self, file_content: bytes, filename: str, model_type: str) -> TelemetryAnalysisResponse:
        df = pd.read_csv(io.BytesIO(file_content))
        timestamps = pd.to_datetime(df['Столбец_1']) if 'Столбец_1' in df.columns else None
        df.drop(columns=['Столбец_0', 'Столбец_1'], inplace=True, errors='ignore')

        valid_cols = self._repository.load_valid_columns()
        scaler = self._repository.load_scaler()
        thresholds = self._repository.load_thresholds()

        df_c = df[valid_cols].ffill().fillna(0).astype(np.float32)
        scaled = np.clip(scaler.transform(df_c), 0, 1)

        X = np.array([scaled[i:i + 10] for i in range(len(scaled) - 10)])
        X_flat = X.reshape(len(X), -1)

        f_errors = np.zeros((len(X), len(valid_cols)))
        errors = np.zeros(len(X))

        if model_type in ['lstm', 'gru', 'cnn', 'vae', 'transformer', 'tcn']:
            model = self._repository.load_neural_model(model_type, len(valid_cols))
            f_errors, errors = model.predict(X)
        elif model_type == 'pca':
            pca = self._repository.load_classical_model('pca')
            recon = pca.inverse_transform(pca.transform(X_flat))
            f_errors = np.abs(X_flat - recon).reshape(len(X), 10, -1).mean(axis=1)
            errors = f_errors.mean(axis=1)
        else:
            ml_model = self._repository.load_classical_model(model_type)
            if model_type == 'elliptic':
                pca_transformer = self._repository.load_classical_model('pca_ee_transformer')
                errors = ml_model.mahalanobis(pca_transformer.transform(X_flat))
            elif model_type in ['iforest', 'lof']:
                errors = -ml_model.score_samples(X_flat)
            elif model_type == 'ocsvm':
                errors = -ml_model.decision_function(X_flat)

        errors_s = pd.Series(errors).rolling(10, min_periods=1).mean().values
        threshold = thresholds.get(model_type, float(np.percentile(errors_s, 95)))
        anom = errors_s > threshold
        soh = max(0, min(100, int(100 * (1 - (errors_s[-1] / (threshold * 2 + 1e-6))))))

        avg_interval = 3600.0
        if timestamps is not None and len(timestamps) > 1:
            intervals = np.diff(timestamps).astype('timedelta64[s]').astype(np.float64)
            pos_intervals = intervals[intervals > 0]
            if len(pos_intervals) > 0:
                avg_interval = float(np.median(pos_intervals))

        rul_label, steps_linear, trend_line = RulEstimator.estimate_linear_rul(errors_s, threshold, avg_interval)

        roll_mean = pd.Series(errors_s).rolling(window=50, min_periods=10).mean().values
        roll_std = pd.Series(errors_s).rolling(window=50, min_periods=10).std().values
        adaptive_thr = np.nan_to_num(roll_mean + 3.0 * roll_std, nan=threshold)

        accel = np.gradient(np.gradient(errors_s))

        culprit, recommendation = RuleEngine.determine_culprit(f_errors, anom, valid_cols)

        window_len = min(len(errors_s), 100)
        x_trend = np.arange(window_len).reshape(-1, 1)
        y_trend = errors_s[-window_len:]

        trend_graph = self._plotter.create_trend_graph(x_trend, y_trend, trend_line, threshold, steps_linear)
        t_graph = self._plotter.create_simple_line_graph(errors_s, threshold)
        h_graph = self._plotter.create_heatmap(f_errors, valid_cols)
        dist_graph = self._plotter.create_distribution_graph(errors_s)

        importance = np.mean(f_errors[anom], axis=0) if f_errors.any() and anom.any() else np.zeros(len(valid_cols))
        imp_graph = self._plotter.create_importance_graph(importance, valid_cols)
        cum_graph = self._plotter.create_cumulative_graph(errors_s)
        corr_graph = self._plotter.create_correlation_graph(df_c.corr().values)
        adaptive_graph = self._plotter.create_adaptive_graph(errors_s, adaptive_thr, threshold)
        accel_graph = self._plotter.create_acceleration_graph(accel)

        last_error = float(errors_s[-1])
        distance = abs(last_error - threshold) / (threshold + 1e-6)
        conf_val = 50.0 + 50.0 * (1.0 - np.exp(-3.0 * distance))

        g_timeline, g_probabilities, g_statement = self._calculate_weibull_survival(errors_s, threshold, avg_interval)

        sensors_diagnostics = []
        for col_idx, col_name in enumerate(valid_cols):
            sensor_errs = f_errors[:, col_idx]
            sensor_errs_s = pd.Series(sensor_errs).rolling(10, min_periods=1).mean().values
            sensor_thr = float(np.percentile(sensor_errs_s, 95))

            s_soh = max(0, min(100, int(100 * (1 - (sensor_errs_s[-1] / (sensor_thr * 2 + 1e-6))))))
            s_rul_label, _, _ = RulEstimator.estimate_linear_rul(sensor_errs_s, sensor_thr, avg_interval)

            _, s_probabilities, _ = self._calculate_weibull_survival(sensor_errs_s, sensor_thr, avg_interval)

            sensors_diagnostics.append(
                SensorDiagnosticInfo(
                    sensor_name=col_name,
                    soh=s_soh,
                    final_rul=s_rul_label,
                    current_error=float(sensor_errs_s[-1]),
                    threshold=sensor_thr,
                    survival_probabilities=s_probabilities
                )
            )

        return TelemetryAnalysisResponse(
            filename=filename,
            model_type=model_type,
            soh=soh,
            final_rul=rul_label,
            trend_graph=trend_graph,
            t_graph=t_graph,
            h_graph=h_graph,
            dist_graph=dist_graph,
            imp_graph=imp_graph,
            cum_graph=cum_graph,
            corr_graph=corr_graph,
            adaptive_graph=adaptive_graph,
            accel_graph=accel_graph,
            holt_graph="",
            num_anomalies=int(np.sum(anom)),
            total_windows=len(errors_s),
            threshold=threshold,
            culprit=culprit,
            recommendation=recommendation,
            conf=f"{conf_val:.1f}%",
            stream_data=pd.Series(errors_s).to_json(orient='records'),
            survival_timeline=g_timeline,
            global_survival_probabilities=g_probabilities,
            survival_statement=g_statement,
            sensors_diagnostics=sensors_diagnostics
        )


class GetAIRecommendationUseCase:
    def __init__(self, settings_repo: ISettingsRepository, llm_service: ILLMService):
        self._settings_repo = settings_repo
        self._llm_service = llm_service

    def execute(self, telemetry_summary: Dict[str, Any]) -> str:
        settings = self._settings_repo.get_settings()
        return self._llm_service.analyze(
            telemetry_summary,
            settings.get("api_key", ""),
            settings.get("model_name", "openai/gpt-latest")
        )


class ManageSettingsUseCase:
    def __init__(self, settings_repo: ISettingsRepository):
        self._settings_repo = settings_repo

    def get_settings(self) -> Dict[str, str]:
        return self._settings_repo.get_settings()

    def save_settings(self, api_key: str, model_name: str) -> None:
        self._settings_repo.save_settings({
            "api_key": api_key,
            "model_name": model_name
        })