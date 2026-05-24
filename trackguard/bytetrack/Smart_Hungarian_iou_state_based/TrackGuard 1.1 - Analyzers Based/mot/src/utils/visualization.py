import cv2
import numpy as np
from typing import List, Dict, Tuple

class Visualizer:
    """Visualization handler for TrackGuard"""
    
    def __init__(self):
        self.colors = {}  # track_id to color mapping
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
        
    def draw_frame(self, 
                   frame: np.ndarray, 
                   detections: List[Dict], 
                   metrics: Dict) -> np.ndarray:
        """
        Draw detections and metrics on frame
        
        Args:
            frame: Current video frame
            detections: List of enhanced detections from TrackGuard
            metrics: Current frame metrics
            
        Returns:
            np.ndarray: Frame with visualizations
        """
        # Create a copy of the frame
        vis_frame = frame.copy()
        
        # Draw each detection
        for det in detections:
            self._draw_detection(vis_frame, det)
            
        # Draw metrics
        self._draw_metrics(vis_frame, metrics)
        
        return vis_frame
        
    def _draw_detection(self, frame: np.ndarray, detection: Dict):
        """Draw single detection with its metadata"""
        bbox = detection['bbox']
        conf = detection['confidence']
        track_id = detection.get('track_id', None)
        
        # Get color for this track
        if track_id not in self.colors:
            self.colors[track_id] = self._generate_color()
        color = self.colors[track_id]
        
        # Draw YOLO bbox
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)
        
        # Draw analysis bbox if available (using dotted line effect)
        if 'analysis_bbox' in detection:
            ax1, ay1, ax2, ay2 = map(int, detection['analysis_bbox'])
            # Create dotted line effect manually
            dash_length = 10
            for i in range(ax1, ax2, dash_length):
                cv2.line(frame, 
                        (i, ay1), 
                        (min(i + dash_length//2, ax2), ay1), 
                        color, 
                        1)
                cv2.line(frame, 
                        (i, ay2), 
                        (min(i + dash_length//2, ax2), ay2), 
                        color, 
                        1)
            for i in range(ay1, ay2, dash_length):
                cv2.line(frame, 
                        (ax1, i), 
                        (ax1, min(i + dash_length//2, ay2)), 
                        color, 
                        1)
                cv2.line(frame, 
                        (ax2, i), 
                        (ax2, min(i + dash_length//2, ay2)), 
                        color, 
                        1)
        
        # Draw text information
        text = f"ID:{track_id} Conf:{conf:.2f}"
        text_size, _ = cv2.getTextSize(text, self.font, self.font_scale, self.thickness)
        text_w, text_h = text_size
        
        # Draw text background
        cv2.rectangle(frame, 
                     (x1, y1 - text_h - 8), 
                     (x1 + text_w, y1), 
                     color, 
                     -1)
        
        # Draw text
        cv2.putText(frame, 
                    text,
                    (x1, y1 - 5),
                    self.font,
                    self.font_scale,
                    (255, 255, 255),
                    self.thickness)
                    
    def _draw_metrics(self, frame: np.ndarray, metrics: Dict):
        """Draw performance metrics on frame"""
        height, width = frame.shape[:2]
        padding = 10
        y_offset = padding
        
        # Draw metrics background
        metrics_height = 80
        cv2.rectangle(frame,
                     (0, 0),
                     (250, metrics_height),
                     (0, 0, 0),
                     -1)
        cv2.rectangle(frame,
                     (0, 0),
                     (250, metrics_height),
                     (255, 255, 255),
                     1)
                     
        # Draw metrics text
        for key, value in metrics.items():
            if isinstance(value, float):
                text = f"{key}: {value:.2f}"
            else:
                text = f"{key}: {value}"
                
            cv2.putText(frame,
                       text,
                       (padding, y_offset + 15),
                       self.font,
                       0.5,
                       (255, 255, 255),
                       1)
            y_offset += 20
            
    def _generate_color(self) -> Tuple[int, int, int]:
        """Generate a random color for new tracks"""
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        return color