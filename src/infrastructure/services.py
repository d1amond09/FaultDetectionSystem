import io
import base64
import requests
from typing import List, Any, Dict
from fpdf import FPDF
import tempfile
import os
from src.domain.interfaces import IPlotterService, IReportGenerator, ILLMService


class OpenRouterLLMService(ILLMService):
    def __init__(self):
        import threading
        self._is_running = False
        self._lock = threading.Lock()

    def analyze(self, telemetry_summary: Dict[str, Any], api_key: str, model_name: str) -> str:
        if not api_key:
            return "API ключ OpenRouter не задан. Пожалуйста, укажите его на вкладке Настройки."

        with self._lock:
            if self._is_running:
                return "Сервер занят обработкой другого запроса к ИИ. Пожалуйста, повторите попытку через минуту."
            self._is_running = True

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        prompt = (
            "Вы являетесь ведущим инженером-диагностом по обслуживанию промышленного электрооборудования "
            "и тяжелых дробильно-сортировочных комплексов (щековые, конусные, роторные дробилки гранита).\n\n"
            "Проанализируйте телеметрические показатели неисправности оборудования:\n"
            f"- Файл данных телеметрии: `{telemetry_summary.get('filename')}`\n"
            f"- Примененная модель детекции: `{telemetry_summary.get('model_type')}`\n"
            f"- Текущий индекс здоровья (SOH): **{telemetry_summary.get('soh')}%**\n"
            f"- Аппаратный ресурс до отказа (RUL): **{telemetry_summary.get('final_rul')}**\n"
            f"- Выявленный аварийный порог: {telemetry_summary.get('threshold')}\n"
            f"- Потенциально неисправный узел: {telemetry_summary.get('culprit')}\n"
            f"- Базовая инженерная рекомендация: {telemetry_summary.get('recommendation')}\n"
            f"- Количество проанализированных временных окон: {telemetry_summary.get('total_windows')}\n"
            f"- Число зарегистрированных аномальных интервалов: {telemetry_summary.get('num_anomalies')}\n"
            f"- Достоверность математической оценки: {telemetry_summary.get('conf')}\n\n"
            "Подготовьте структурированный экспертный отчет исключительно на русском языке в формате Markdown. "
            "Используйте emoji, жирный шрифт для ключевых параметров, списки и четкую разметку. "
            "Отчет должен строго состоять из следующих разделов:\n\n"
            "### 📊 1. Инженерный анализ состояния\n"
            "Оцените индекс SOH и показатель RUL в условиях промышленного дробления высокопрочных пород.\n\n"
            "### 🔍 2. Физика процессов и локализация дефекта\n"
            f"Объясните вероятные причины сбоя узла '{telemetry_summary.get('culprit')}'. Свяжите это со спецификой эксплуатации дробилок "
            "(критические вибрации, попадание недробимых тел, абразивный износ пылью, перекос валов или перегрузка электродвигателя).\n\n"
            "### ⚠️ 3. Оценка рисков и сценарий развития отказа\n"
            "Опишите потенциальные последствия при продолжении работы без ремонта (аварийный останов, разрушение подшипников, КЗ).\n\n"
            "### 🛠️ 4. Технологический регламент восстановительных работ\n"
            "Дайте конкретный пошаговый план действий для сервисной службы (дефектовка, очистка, виброналадка, регламент смазки)."
        )

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=20)
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                return f"Ошибка при запросе к LLM API ({response.status_code}): {response.text}"
        except Exception as e:
            return f"Исключение при обращении к LLM: {str(e)}"
        finally:
            with self._lock:
                self._is_running = False


class MatplotlibPlotterService(IPlotterService):
    def _to_b64(self, fig: Any) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', facecolor='#1e293b')
        import matplotlib.pyplot as plt
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()

    def create_trend_graph(self, x: Any, y: Any, trend: Any, threshold: float, steps: Any) -> str:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.linear_model import LinearRegression
        import numpy as np

        fig, ax = plt.subplots(figsize=(10, 2.5))
        fig.patch.set_facecolor('#1e293b')
        ax.set_facecolor('#1e293b')

        ax.plot(x, y, color='#3b82f6', label='MAE (сглаж.)')

        max_x = len(x)
        if steps is not None and steps > 0:
            max_x = max(max_x, int(steps * 1.15))

        lr = LinearRegression().fit(x.reshape(-1, 1), y)
        x_extended = np.arange(max_x)
        trend_extended = lr.predict(x_extended.reshape(-1, 1))

        ax.plot(x_extended, trend_extended, '--', color='#fbbf24', label='Линейный тренд')
        ax.axhline(threshold, color='#ef4444', linestyle='--', label=f'Порог ({threshold:.3f})')

        if steps is not None and steps > 0:
            ax.plot(steps, threshold, 'rx', markersize=10, label='Прогноз сбоя (пересечение)')

        ax.legend(loc='upper left', facecolor='#1e293b', edgecolor='#334155', labelcolor='white', fontsize=8)
        ax.tick_params(colors='white')
        return self._to_b64(fig)

    def create_simple_line_graph(self, data: Any, threshold: float) -> str:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 3))
        fig.patch.set_facecolor('#1e293b')
        ax.set_facecolor('#1e293b')
        ax.plot(data, color='#3b82f6', label='Текущая ошибка MAE')
        ax.axhline(threshold, color='red', ls='--', label=f'Порог ({threshold:.3f})')
        ax.legend(loc='upper left', facecolor='#1e293b', edgecolor='#334155', labelcolor='white', fontsize=8)
        ax.tick_params(colors='white')
        return self._to_b64(fig)

    def create_heatmap(self, data: Any, labels: List[str]) -> str:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#1e293b')
        source = data[-60:].T if data.any() else np.zeros((len(labels), 60))
        sns.heatmap(source, xticklabels=False, yticklabels=labels, cmap='rocket', ax=ax, cbar=False)
        ax.tick_params(axis='y', colors='white', labelsize=6)
        return self._to_b64(fig)

    def create_distribution_graph(self, data: Any) -> str:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_facecolor('#1e293b')
        ax.set_facecolor('#1e293b')
        sns.histplot(data, kde=True, color='#a78bfa', ax=ax, label='Распределение ошибок')
        ax.legend(loc='upper right', facecolor='#1e293b', edgecolor='#334155', labelcolor='white', fontsize=8)
        ax.tick_params(colors='white')
        return self._to_b64(fig)

    def create_importance_graph(self, importance: Any, labels: List[str]) -> str:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_facecolor('#1e293b')
        ax.set_facecolor('#1e293b')
        idxs = np.argsort(importance)[-10:]
        sns.barplot(x=importance[idxs], y=np.array(labels)[idxs], hue=np.array(labels)[idxs], palette='mako', ax=ax,
                    legend=False)
        ax.tick_params(colors='white', labelsize=7)
        return self._to_b64(fig)

    def create_cumulative_graph(self, data: Any) -> str:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(10, 3))
        fig.patch.set_facecolor('#1e293b')
        ax.set_facecolor('#1e293b')
        ax.plot(np.cumsum(data), color='#fbbf24', lw=2, label='Кумулятивная сумма ошибок')
        ax.legend(loc='upper left', facecolor='#1e293b', edgecolor='#334155', labelcolor='white', fontsize=8)
        ax.tick_params(colors='white')
        return self._to_b64(fig)

    def create_correlation_graph(self, matrix: Any) -> str:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor('#1e293b')
        sns.heatmap(matrix, cmap='coolwarm', center=0, cbar=False, xticklabels=False, yticklabels=False)
        return self._to_b64(fig)

    def create_adaptive_graph(self, data: Any, adaptive: Any, threshold: float) -> str:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 2.5))
        fig.patch.set_facecolor('#1e293b')
        ax.set_facecolor('#1e293b')
        ax.plot(data, color='#3b82f6', label='Ошибка MAE')
        ax.plot(adaptive, color='#f97316', linestyle='--', label='Адаптивный порог')
        ax.axhline(threshold, color='#ef4444', linestyle=':', label=f'Базовый порог ({threshold:.3f})')
        ax.legend(loc='upper left', facecolor='#1e293b', edgecolor='#334155', labelcolor='white', fontsize=8)
        ax.tick_params(colors='white')
        return self._to_b64(fig)

    def create_acceleration_graph(self, data: Any) -> str:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 2.5))
        fig.patch.set_facecolor('#1e293b')
        ax.set_facecolor('#1e293b')
        ax.plot(data, color='#a78bfa', label='Ускорение MAE')
        ax.axhline(0, color='#94a3b8', linestyle='--', label='Нулевой уровень')
        ax.legend(loc='upper left', facecolor='#1e293b', edgecolor='#334155', labelcolor='white', fontsize=8)
        ax.tick_params(colors='white')
        return self._to_b64(fig)

    def create_holt_graph(self, data: Any, forecast: Any, adaptive_threshold: float) -> str:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 2.5))
        fig.patch.set_facecolor('#1e293b')
        ax.set_facecolor('#1e293b')
        ax.plot(range(len(data)), data, color='#3b82f6', label='Ошибка MAE')
        idx = range(len(data) - 1, len(data) + len(forecast) - 1)
        ax.plot(idx, forecast, color='#fbbf24', linestyle='--', label='Прогноз Хольта')
        ax.axhline(adaptive_threshold, color='#f97316', linestyle='--', label='Адаптивный порог')
        ax.legend(loc='upper left', facecolor='#1e293b', edgecolor='#334155', labelcolor='white', fontsize=8)
        ax.tick_params(colors='white')
        return self._to_b64(fig)


class PdfReportGenerator(IReportGenerator):
    def generate_pdf(self, filename: str, soh: int, graphs: List[str]) -> bytes:
        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("Arial", "", "C:\\Windows\\Fonts\\arial.ttf", uni=True)
        pdf.set_font("Arial", "", 16)
        pdf.cell(0, 10, f"ОТЧЕТ ДИАГНОСТИКИ ИИ: {filename}", ln=1, align="C")
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"ИНДЕКС ЗДОРОВЬЯ: {soh}%", ln=1, align="C")

        for graph_b64 in graphs:
            if graph_b64 and len(graph_b64) > 100:
                tmp_path = tempfile.mktemp(suffix=".png")
                with open(tmp_path, "wb") as f:
                    f.write(base64.b64decode(graph_b64))
                pdf.image(tmp_path, x=10, w=190)
                pdf.ln(5)
                os.remove(tmp_path)
                if pdf.get_y() > 220:
                    pdf.add_page()

        return pdf.output(dest='S').encode('latin-1')