"""
Module initialization for trainers package
"""
from .base_trainer import BaseTrainer
from .yolo_trainer import YOLOTrainer
from .faster_rcnn_trainer import FasterRCNNTrainer

__all__ = [
    'BaseTrainer',
    'YOLOTrainer',
    'FasterRCNNTrainer'
]