import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.linear_model import LinearRegression


class RuleEngine:
    @staticmethod
    def determine_culprit(features_errors: np.ndarray, is_anomaly: np.ndarray, columns: List[str]) -> Tuple[str, str]:
        if features_errors.any() and is_anomaly.any():
            mean_errors = np.mean(features_errors[is_anomaly], axis=0)
            idx = int(np.argmax(mean_errors))
            culprit = columns[idx]
            recommendation = f"Внимание: Проверьте {culprit}. Рекомендуется диагностика узла."
            return culprit, recommendation
        return "Н/Д", "Система в норме"


class RulEstimator:
    @staticmethod
    def estimate_linear_rul(errors: np.ndarray, threshold: float, avg_interval: float) -> Tuple[
        str, Optional[float], np.ndarray]:
        window_len = min(len(errors), 100)
        x_trend = np.arange(window_len).reshape(-1, 1)
        y_trend = errors[-window_len:]

        lr = LinearRegression().fit(x_trend, y_trend)
        slope = float(lr.coef_[0])
        current_val = float(y_trend[-1])
        trend_line = lr.predict(x_trend)

        if current_val >= threshold:
            return "Критично", None, trend_line

        if slope > 1e-6:
            steps = (threshold - current_val) / slope
            sec = steps * avg_interval
            days = int(sec // 86400)
            hours = int((sec % 86400) // 3600)
            if days == 0 and hours == 0:
                return "Критично", steps, trend_line
            return (f"{days} дн. {hours} ч." if days > 0 else f"{hours} ч."), steps, trend_line

        return "Стабильно", None, trend_line