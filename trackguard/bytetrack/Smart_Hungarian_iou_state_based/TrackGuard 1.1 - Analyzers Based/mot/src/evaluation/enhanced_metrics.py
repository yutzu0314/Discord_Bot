import numpy as np
from typing import List, Dict
from collections import defaultdict

class EnhancedMetrics:
    def __init__(self):
        # Basic counters
        self.total_objects = []  # per frame
        self.ghost_effects = 0
        self.double_ids = 0
        self.bbox_areas = []
        
        # Tracking history
        self.track_history = defaultdict(list)  # track_id -> bbox history
        
    def update(self, detections: List[Dict], frame_shape: tuple = None):
        """Update metrics with new detections"""
        # Count objects
        self.total_objects.append(len(detections))
        
        # Track centers for double ID detection
        centers = {}
        
        for det in detections:
            # Calculate center
            bbox = det['bbox']
            center = (
                (bbox[0] + bbox[2])/2,
                (bbox[1] + bbox[3])/2
            )
            
            # Check for ghost effects (very low confidence)
            if det['confidence'] < 0.25:  # threshold
                self.ghost_effects += 1
            
            # Check for double IDs
            center_key = f"{center[0]:.1f},{center[1]:.1f}"
            if center_key in centers:
                self.double_ids += 1
            centers[center_key] = det['track_id']
            
            # Calculate and store bbox area
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            if frame_shape:
                # Normalize by frame size
                frame_area = frame_shape[0] * frame_shape[1]
                area = area / frame_area
            self.bbox_areas.append(area)
            
            # Update track history
            track_id = det['track_id']
            self.track_history[track_id].append(bbox)
    
    def get_summary(self) -> Dict:
        """Get summary of metrics"""
        return {
            'avg_objects_per_frame': np.mean(self.total_objects),
            'total_ghost_effects': self.ghost_effects,
            'total_double_ids': self.double_ids,
            'avg_bbox_area': np.mean(self.bbox_areas) if self.bbox_areas else 0,
            'max_bbox_area': np.max(self.bbox_areas) if self.bbox_areas else 0,
            'min_bbox_area': np.min(self.bbox_areas) if self.bbox_areas else 0
        }