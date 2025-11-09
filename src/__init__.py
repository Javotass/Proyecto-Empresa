"""
Sistema de detección de operaciones atípicas en transacciones financieras
"""

__version__ = "1.0.0"
__author__ = "Javier Revilla"

from .data_generator import TransactionDataGenerator
from .data_processor import DataProcessor
from .model_trainer import AnomalyDetector
from .evaluator import ModelEvaluator
from .visualizer import ResultVisualizer

__all__ = [
    'TransactionDataGenerator',
    'DataProcessor',
    'AnomalyDetector',
    'ModelEvaluator',
    'ResultVisualizer'
]