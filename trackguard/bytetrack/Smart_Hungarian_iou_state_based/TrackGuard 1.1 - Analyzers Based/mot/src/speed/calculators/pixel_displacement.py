import numpy as np
from typing import List, Tuple, Optional
import math

class PixelDisplacementCalculator:
    """
    Fast pixel-based speed calculation method
    
    Calculates speed based on bounding box center displacement between frames.
    This is the fastest method and works well for high-confidence tracks.
    """
    
    def __init__(self, pixels_per_meter: float = 20.0, fps: float = 30.0):
        """
        Initialize pixel displacement calculator
        
        Args:
            pixels_per_meter: Camera calibration parameter (pixels per meter)
            fps: Video frames per second
        """
        self.pixels_per_meter = pixels_per_meter
        self.fps = fps
        self.frame_time = 1.0 / fps  # Time between frames in seconds
        
    def calculate_speed(self, bbox_current: List[float], bbox_previous: List[float], 
                       time_delta: Optional[float] = None) -> float:
        """
        Calculate speed based on bounding box center displacement
        
        Args:
            bbox_current: Current frame bbox [x1, y1, x2, y2]
            bbox_previous: Previous frame bbox [x1, y1, x2, y2]
            time_delta: Time difference between frames (if None, uses default fps)
            
        Returns:
            float: Speed in m/s
        """
        if time_delta is None:
            time_delta = self.frame_time
            
        # Calculate center points
        center_current = self._get_bbox_center(bbox_current)
        center_previous = self._get_bbox_center(bbox_previous)
        
        # Calculate pixel displacement
        pixel_displacement = self._calculate_euclidean_distance(center_current, center_previous)
        
        # Convert to real-world displacement (meters)
        meter_displacement = pixel_displacement / self.pixels_per_meter
        
        # Calculate speed (m/s)
        speed_ms = meter_displacement / time_delta if time_delta > 0 else 0.0
        
        return max(0.0, speed_ms)  # Ensure non-negative speed
    
    def calculate_speed_kmh(self, bbox_current: List[float], bbox_previous: List[float], 
                           time_delta: Optional[float] = None) -> float:
        """
        Calculate speed in km/h
        
        Args:
            bbox_current: Current frame bbox [x1, y1, x2, y2]
            bbox_previous: Previous frame bbox [x1, y1, x2, y2]
            time_delta: Time difference between frames
            
        Returns:
            float: Speed in km/h
        """
        speed_ms = self.calculate_speed(bbox_current, bbox_previous, time_delta)
        return speed_ms * 3.6  # Convert m/s to km/h
    
    def calculate_velocity_vector(self, bbox_current: List[float], bbox_previous: List[float], 
                                 time_delta: Optional[float] = None) -> Tuple[float, float]:
        """
        Calculate velocity vector (vx, vy) in m/s
        
        Args:
            bbox_current: Current frame bbox [x1, y1, x2, y2]
            bbox_previous: Previous frame bbox [x1, y1, x2, y2]
            time_delta: Time difference between frames
            
        Returns:
            Tuple[float, float]: Velocity vector (vx, vy) in m/s
        """
        if time_delta is None:
            time_delta = self.frame_time
            
        center_current = self._get_bbox_center(bbox_current)
        center_previous = self._get_bbox_center(bbox_previous)
        
        # Calculate displacement vector in pixels
        dx_pixels = center_current[0] - center_previous[0]
        dy_pixels = center_current[1] - center_previous[1]
        
        # Convert to meters
        dx_meters = dx_pixels / self.pixels_per_meter
        dy_meters = dy_pixels / self.pixels_per_meter
        
        # Calculate velocity components
        if time_delta > 0:
            vx = dx_meters / time_delta
            vy = dy_meters / time_delta
        else:
            vx = vy = 0.0
            
        return vx, vy
    
    def calculate_direction(self, bbox_current: List[float], bbox_previous: List[float]) -> float:
        """
        Calculate movement direction in degrees (0-360)
        
        Args:
            bbox_current: Current frame bbox [x1, y1, x2, y2]
            bbox_previous: Previous frame bbox [x1, y1, x2, y2]
            
        Returns:
            float: Direction in degrees (0° = right, 90° = down, 180° = left, 270° = up)
        """
        center_current = self._get_bbox_center(bbox_current)
        center_previous = self._get_bbox_center(bbox_previous)
        
        dx = center_current[0] - center_previous[0]
        dy = center_current[1] - center_previous[1]
        
        # Calculate angle in radians, then convert to degrees
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        # Normalize to 0-360 degrees
        if angle_deg < 0:
            angle_deg += 360
            
        return angle_deg
    
    def update_calibration(self, pixels_per_meter: float, fps: float = None):
        """
        Update calibration parameters
        
        Args:
            pixels_per_meter: New calibration parameter
            fps: New fps value (optional)
        """
        self.pixels_per_meter = pixels_per_meter
        if fps is not None:
            self.fps = fps
            self.frame_time = 1.0 / fps
    
    def _get_bbox_center(self, bbox: List[float]) -> Tuple[float, float]:
        """
        Calculate center point of bounding box
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Tuple[float, float]: Center point (cx, cy)
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        return center_x, center_y
    
    def _calculate_euclidean_distance(self, point1: Tuple[float, float], 
                                    point2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two points
        
        Args:
            point1: First point (x1, y1)
            point2: Second point (x2, y2)
            
        Returns:
            float: Euclidean distance
        """
        dx = point1[0] - point2[0]
        dy = point1[1] - point2[1]
        return math.sqrt(dx * dx + dy * dy)
    
    def get_speed_confidence(self, bbox_current: List[float], bbox_previous: List[float]) -> float:
        """
        Calculate confidence score for speed estimation based on bbox stability
        
        Args:
            bbox_current: Current frame bbox [x1, y1, x2, y2]
            bbox_previous: Previous frame bbox [x1, y1, x2, y2]
            
        Returns:
            float: Confidence score (0-1)
        """
        # Calculate area change ratio
        area_current = (bbox_current[2] - bbox_current[0]) * (bbox_current[3] - bbox_current[1])
        area_previous = (bbox_previous[2] - bbox_previous[0]) * (bbox_previous[3] - bbox_previous[1])
        
        if area_previous > 0:
            area_ratio = min(area_current, area_previous) / max(area_current, area_previous)
        else:
            area_ratio = 0.0
        
        # Calculate aspect ratio consistency
        def get_aspect_ratio(bbox):
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            return w / h if h > 0 else 1.0
        
        aspect_current = get_aspect_ratio(bbox_current)
        aspect_previous = get_aspect_ratio(bbox_previous)
        aspect_ratio = min(aspect_current, aspect_previous) / max(aspect_current, aspect_previous)
        
        # Combined confidence score
        confidence = (area_ratio * 0.6 + aspect_ratio * 0.4)
        return max(0.0, min(1.0, confidence))