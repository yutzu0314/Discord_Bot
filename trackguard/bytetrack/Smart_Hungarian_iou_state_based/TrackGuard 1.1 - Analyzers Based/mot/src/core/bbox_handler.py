import numpy as np
from typing import List, Tuple, Union

class BBoxHandler:
    """Handle all bounding box operations for TrackGuard"""
    
    def __init__(self, expansion_factor: float = 1.1):  # Nilai dasar yang seimbang antara 1.0 dan 1.2
        self.expansion_factor = expansion_factor
        self.min_expansion = 1.05  # Minimal expansion untuk objek besar
        self.max_expansion = 1.25  # Maksimal expansion untuk objek kecil
    
    def get_analysis_region(self, bbox: Union[List, np.ndarray]) -> np.ndarray:
        """
        Create adaptive analysis region around detection bbox
        
        Args:
            bbox: Original bounding box [x1, y1, x2, y2]
            
        Returns:
            np.ndarray: Expanded bounding box with adaptive factor
        """
        bbox = np.array(bbox)
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # Calculate bbox area relative to frame
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        frame_area = 1920 * 1080  # Standard reference area
        area_ratio = area / frame_area
        
        # Smaller objects get larger expansion
        adaptive_factor = max(
            self.min_expansion,
            min(self.max_expansion, self.max_expansion - (area_ratio * 15))
        )
        
        width = width * adaptive_factor
        height = height * adaptive_factor
        
        return np.array([
            max(0, center_x - width/2),
            max(0, center_y - height/2),
            center_x + width/2,
            center_y + height/2
        ])
    
    def get_analysis_region_mot(self, bbox: Union[List, np.ndarray]) -> np.ndarray:
        """
        Non-expanded version for MOT evaluation - returns original bbox
        
        Args:
            bbox: Original bounding box [x1, y1, x2, y2]
            
        Returns:
            np.ndarray: Same bounding box without expansion
        """
        return np.array(bbox)
    
    def calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """
        Calculate Intersection over Union between two bboxes
        
        Args:
            bbox1: First bbox [x1, y1, x2, y2]
            bbox2: Second bbox [x1, y1, x2, y2]
            
        Returns:
            float: IoU score
        """
        bbox1 = np.array(bbox1)
        bbox2 = np.array(bbox2)
        
        # Calculate intersection coordinates
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        # Calculate area of intersection
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate areas of both bboxes
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        # Calculate IoU
        iou = intersection / (bbox1_area + bbox2_area - intersection + 1e-6)
        return float(iou)
    
    def clip_bbox(self, bbox: np.ndarray, frame_size: Tuple[int, int]) -> np.ndarray:
        """
        Clip bbox coordinates to frame boundaries
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            frame_size: (height, width) of frame
            
        Returns:
            np.ndarray: Clipped bbox
        """
        height, width = frame_size
        return np.array([
            max(0, min(bbox[0], width)),
            max(0, min(bbox[1], height)),
            max(0, min(bbox[2], width)),
            max(0, min(bbox[3], height))
        ])
    
    def get_bbox_center(self, bbox: np.ndarray) -> Tuple[float, float]:
        """
        Calculate center point of bbox
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Tuple[float, float]: (center_x, center_y)
        """
        return (
            (bbox[0] + bbox[2]) / 2,
            (bbox[1] + bbox[3]) / 2
        )
    
    def get_bbox_dimensions(self, bbox: np.ndarray) -> Tuple[float, float]:
        """
        Calculate width and height of bbox
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Tuple[float, float]: (width, height)
        """
        return (
            bbox[2] - bbox[0],
            bbox[3] - bbox[1]
        )