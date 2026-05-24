from abc import ABC, abstractmethod
import numpy as np
from typing import Union, List

class BaseAnalyzer(ABC):
    """Abstract base class for all analyzers"""
    
    @abstractmethod
    def analyze(self, frame: np.ndarray, bbox: Union[List, np.ndarray]) -> float:
        """
        Analyze the region of interest in the frame
        
        Args:
            frame: Current video frame
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            float: Analysis score between 0 and 1
        """
        pass
    
    def _extract_roi(self, frame: np.ndarray, bbox: Union[List, np.ndarray]) -> np.ndarray:
        """
        Extract region of interest from frame
        
        Args:
            frame: Current video frame
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            np.ndarray: ROI image
        """
        bbox = np.array(bbox, dtype=np.int32)
        x1, y1, x2, y2 = bbox
        
        # Ensure bbox is within frame boundaries
        height, width = frame.shape[:2]
        x1 = max(0, min(x1, width-1))
        y1 = max(0, min(y1, height-1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        
        return frame[y1:y2, x1:x2]
