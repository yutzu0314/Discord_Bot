"""
TrackGuard Speed Detection Module
==================================

Confidence-Weighted Speed Fusion (CWSF) for traffic monitoring.
Integrates with TrackGuard tracking system to provide real-time speed estimation.

Main Components:
- SpeedEstimator: Main orchestrator for speed detection
- Calculators: Multiple speed calculation methods (pixel, trajectory, optical flow)
- Fusion: Confidence-weighted fusion and temporal smoothing
- Calibration: Camera calibration for real-world speed conversion
"""

from .speed_estimator import SpeedEstimator

__version__ = "1.0.0"
__author__ = "TrackGuard Speed Team"

__all__ = [
    "SpeedEstimator",
]