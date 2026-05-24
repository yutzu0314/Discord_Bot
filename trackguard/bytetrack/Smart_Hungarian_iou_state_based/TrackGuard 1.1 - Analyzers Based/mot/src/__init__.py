"""
TrackGuard - Enhanced YOLO Object Tracking
==========================================

Main TrackGuard package initialization.
"""

# Core components
from .core.bbox_handler import BBoxHandler
from .core.confidence_handler import ConfidenceHandler

# Analyzers
from .analyzers.color_analyzer import ColorAnalyzer
from .analyzers.shape_analyzer import ShapeAnalyzer

# Models
from .models.yolo_detector import YOLODetector

# Speed detection module
from .speed.speed_estimator import SpeedEstimator

__version__ = "1.0.0"
__all__ = [
    "BBoxHandler", 
    "ConfidenceHandler",
    "ColorAnalyzer",
    "ShapeAnalyzer",
    "YOLODetector",
    "SpeedEstimator",
]