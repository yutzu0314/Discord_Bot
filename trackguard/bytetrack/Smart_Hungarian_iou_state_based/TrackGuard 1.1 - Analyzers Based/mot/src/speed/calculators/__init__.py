"""
Speed Calculation Methods for CWSF
==================================

This module contains various speed calculation methods:
- PixelDisplacementCalculator: Fast pixel-based calculation
- TrajectoryFittingCalculator: Trajectory-based calculation  
- OpticalFlowCalculator: Optical flow-based calculation

Each calculator provides different accuracy vs performance trade-offs.
"""

from .pixel_displacement import PixelDisplacementCalculator
from .trajectory_fitting import TrajectoryFittingCalculator

__all__ = [
    "PixelDisplacementCalculator",
    "TrajectoryFittingCalculator",
]