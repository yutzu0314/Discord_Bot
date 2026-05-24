# src/evaluation/metrics.py

import numpy as np
from typing import Dict, List
from collections import defaultdict
import time
import csv
from pathlib import Path

class RetentionMetrics:
    """Track and calculate retention metrics for TrackGuard"""
    
    def __init__(self):
        self.track_history = defaultdict(lambda: {
            'start_time': None,
            'last_time': None,
            'total_frames': 0,
            'continuous_detection': 0,
            'max_continuous_detection': 0,
            'interruption_count': 0,
            'confidence_history': [],
            'bbox_history': [],
            'last_bbox': None,  # untuk tracking ID switch
            'switches': 0       # count ID switches
        })
        
        # Global metrics
        self.current_frame = 0
        self.start_time = time.time()
        self.total_bboxes = 0
        self.total_id_switches = 0
        self.last_frame_tracks = set()  # untuk tracking ID switches
        
    def update(self, detections: List[Dict]):
        """Update metrics with new detections"""
        self.current_frame += 1
        current_tracks = set()
        
        # Update total bounding boxes
        self.total_bboxes += len(detections)
        
        # Track current frame detections
        for det in detections:
            track_id = det['track_id']
            current_tracks.add(track_id)
            
            track = self.track_history[track_id]
            if track['start_time'] is None:
                track['start_time'] = self.current_frame
                
            # Update tracking statistics
            track['last_time'] = self.current_frame
            track['total_frames'] += 1
            track['continuous_detection'] += 1
            track['max_continuous_detection'] = max(
                track['max_continuous_detection'],
                track['continuous_detection']
            )
            
            # Store history
            track['confidence_history'].append(det['confidence'])
            track['bbox_history'].append(det['bbox'])
            
            # Check for ID switch
            if track['last_bbox'] is not None:
                iou = self._calculate_iou(track['last_bbox'], det['bbox'])
                if iou < 0.3:  # threshold for considering it a switch
                    track['switches'] += 1
                    self.total_id_switches += 1
                    
            track['last_bbox'] = det['bbox']
            
        # Handle interrupted tracks
        for track_id in self.track_history:
            if track_id not in current_tracks:
                track = self.track_history[track_id]
                if track['continuous_detection'] > 0:
                    track['interruption_count'] += 1
                    track['continuous_detection'] = 0
                    
        self.last_frame_tracks = current_tracks

    def get_current_metrics(self) -> Dict:
        """
        Calculate current frame metrics
        
        Returns:
            Dict: Current metrics including retention stats
        """
        if not self.track_history:
            return {
                'avg_retention': 0,
                'max_retention': 0,
                'active_tracks': 0,
                'total_tracks': 0,
                'avg_confidence': 0,
                'id_switches': self.total_id_switches
            }
            
        # Calculate retention statistics
        retentions = []
        confidences = []
        active_tracks = 0
        
        for track_id, track in self.track_history.items():
            if track['continuous_detection'] > 0:
                active_tracks += 1
                
            retention = track['total_frames']
            retentions.append(retention)
            
            if track['confidence_history']:
                confidences.extend(track['confidence_history'][-10:])  # Last 10 frames
                
        metrics = {
            'avg_retention': np.mean(retentions) if retentions else 0,
            'max_retention': max(retentions) if retentions else 0,
            'active_tracks': active_tracks,
            'total_tracks': len(self.track_history),
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'id_switches': self.total_id_switches
        }
        
        return metrics

    def get_summary_metrics(self) -> Dict:
        """Get final summary metrics"""
        duration = time.time() - self.start_time
        
        if not self.track_history:
            return {
                'total_frames': self.current_frame,
                'duration_seconds': duration,
                'total_tracks': 0,
                'avg_retention': 0,
                'max_retention': 0,
                'total_bboxes': self.total_bboxes,
                'id_switches': self.total_id_switches,
                'avg_duration_per_id': 0
            }
            
        retentions = [t['total_frames'] for t in self.track_history.values()]
        
        return {
            'total_frames': self.current_frame,
            'duration_seconds': duration,
            'total_tracks': len(self.track_history),
            'avg_retention': np.mean(retentions),
            'max_retention': max(retentions),
            'total_bboxes': self.total_bboxes,
            'id_switches': self.total_id_switches,
            'avg_duration_per_id': duration / len(self.track_history)
        }
        
    def export_to_csv(self, output_path: str):
        """Export metrics to CSV file"""
        metrics = self.get_summary_metrics()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow([
                'Total Frames',
                'Duration (s)',
                'Total Tracks',
                'Avg Retention (frames)',
                'Max Retention (frames)',
                'Total Bounding Boxes',
                'ID Switch',
                'Avg Duration per ID (s)'
            ])
            # Write values
            writer.writerow([
                metrics['total_frames'],
                f"{metrics['duration_seconds']:.3f}",
                metrics['total_tracks'],
                f"{metrics['avg_retention']:.3f}",
                metrics['max_retention'],
                metrics['total_bboxes'],
                metrics['id_switches'],
                f"{metrics['avg_duration_per_id']:.3f}"
            ])

    def get_track_duration(self, track_id: int) -> int:
        """
        Get duration of specific track in frames
        
        Args:
            track_id: Track identifier
            
        Returns:
            int: Duration in frames
        """
        if track_id in self.track_history:
            track = self.track_history[track_id]
            return track['total_frames']
        return 0
            
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union = bbox1_area + bbox2_area - intersection
        
        return intersection / (union + 1e-6)