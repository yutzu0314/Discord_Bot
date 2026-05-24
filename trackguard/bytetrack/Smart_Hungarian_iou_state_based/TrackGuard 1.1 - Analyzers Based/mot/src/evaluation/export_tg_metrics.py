import csv
import numpy as np
from typing import Dict, List
from pathlib import Path
import time
import pandas as pd

class MetricsExporter:
    def __init__(self, fps: int):
        self.fps = fps
        self.track_data = {}  # track_id -> duration data
        
    def update_track(self, track_id: int, class_name: str, confidence: float, frame_number: int):
        """Update track information"""
        if track_id not in self.track_data:
            self.track_data[track_id] = {
                'class': class_name,
                'start_frame': frame_number,
                'last_frame': frame_number,
                'duration_frames': 1,
                'avg_confidence': [confidence],
                'first_detection': time.time()
            }
        else:
            track = self.track_data[track_id]
            track['last_frame'] = frame_number
            track['duration_frames'] = frame_number - track['start_frame']
            track['avg_confidence'].append(confidence)
    
    def export_to_csv(self, output_path: str, method: str = "TrackGuard"):
        """
        Export metrics to CSV
        
        Args:
            output_path: Path to save CSV file
            method: "TrackGuard" or "YOLO" to indicate which method's metrics
        """
        data = []
        current_time = time.time()
        
        for track_id, track in self.track_data.items():
            duration_seconds = track['duration_frames'] / self.fps
            avg_conf = np.mean(track['avg_confidence'])
            
            data.append({
                'track_id': track_id,
                'class': track['class'],
                'duration_seconds': round(duration_seconds, 2),
                'duration_frames': track['duration_frames'],
                'average_confidence': round(avg_conf, 3),
                'detection_method': method,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # Convert to DataFrame and export
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"\n✨ Metrics exported to: {output_path}")
        
        # Print summary statistics
        print("\n📊 Summary Statistics:")
        print(f"Total tracks: {len(data)}")
        print(f"Average duration: {df['duration_seconds'].mean():.2f} seconds")
        print(f"Max duration: {df['duration_seconds'].max():.2f} seconds")
        print(f"Average confidence: {df['average_confidence'].mean():.3f}")