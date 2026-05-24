import csv
import numpy as np
from typing import Dict, List
from pathlib import Path
import time
import pandas as pd

class YOLOMetricsExporter:
    def __init__(self, fps: int):
        self.fps = fps
        self.detections_data = {}  # detection_id -> duration data
        self.current_id = 0
        
    def update_detection(self, bbox: List[float], class_name: str, confidence: float, frame_number: int):
        """Track YOLO detections"""
        # Simple tracking based on bbox overlap
        matched = False
        for det_id, det_info in self.detections_data.items():
            if self._calculate_iou(bbox, det_info['last_bbox']) > 0.5:
                # Update existing detection
                det_info['last_frame'] = frame_number
                det_info['duration_frames'] = frame_number - det_info['start_frame']
                det_info['confidences'].append(confidence)
                det_info['last_bbox'] = bbox
                matched = True
                break
        
        if not matched:
            # New detection
            self.detections_data[self.current_id] = {
                'class': class_name,
                'start_frame': frame_number,
                'last_frame': frame_number,
                'duration_frames': 1,
                'confidences': [confidence],
                'last_bbox': bbox,
                'first_detection': time.time()
            }
            self.current_id += 1
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bboxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        return intersection / (area1 + area2 - intersection + 1e-6)
    
    def export_to_csv(self, output_path: str):
        """Export YOLO metrics to CSV"""
        data = []
        
        for det_id, det_info in self.detections_data.items():
            duration_seconds = det_info['duration_frames'] / self.fps
            avg_conf = np.mean(det_info['confidences'])
            
            data.append({
                'detection_id': det_id,
                'class': det_info['class'],
                'duration_seconds': round(duration_seconds, 2),
                'duration_frames': det_info['duration_frames'],
                'average_confidence': round(avg_conf, 3),
                'detection_method': 'YOLO',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # Convert to DataFrame and export
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"\n✨ YOLO metrics exported to: {output_path}")
        
        # Print summary statistics
        print("\n📊 YOLO Performance Summary:")
        print(f"Total detections: {len(data)}")
        print(f"Average duration: {df['duration_seconds'].mean():.2f} seconds")
        print(f"Max duration: {df['duration_seconds'].max():.2f} seconds")
        print(f"Average confidence: {df['average_confidence'].mean():.3f}")