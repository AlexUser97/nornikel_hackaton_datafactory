"""«Кто твой шлиф» — импортонезависимый QC-ассистент металлографа.

Публичный контракт пакета — единственная функция :func:`analyze`, которая
превращает снимок оптической микроскопии в количественный «паспорт образца».

Всё ядро работает офлайн, на CPU, без ML-весов: сегментация — классический
OpenCV (multi-Otsu + watershed), неуверенность — разброс ансамбля прогонов.
"""
from .analyze import analyze
from .config import DEFAULT_CLASS_NAMES
from .evaluation import evaluate_segmentation

__all__ = ["analyze", "evaluate_segmentation", "DEFAULT_CLASS_NAMES"]
__version__ = "0.1.0"
