import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from pathlib import Path
import time

class PerformanceVisualizer:
    """Visualize TrackGuard performance metrics"""
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.retention_history = []
        self.confidence_history = []
        self.active_tracks_history = []
        self.timestamps = []
        
    def update(self, metrics: Dict):
        """
        Update performance history with new metrics
        
        Args:
            metrics: Dictionary of current metrics
        """
        # Add timestamp
        self.timestamps.append(time.time())
        
        # Update histories
        self.retention_history.append(metrics['avg_retention'])
        self.confidence_history.append(metrics['avg_confidence'])
        self.active_tracks_history.append(metrics['active_tracks'])
        
        # Keep only recent history
        if len(self.retention_history) > self.history_size:
            self.retention_history.pop(0)
            self.confidence_history.pop(0)
            self.active_tracks_history.pop(0)
            self.timestamps.pop(0)
            
    def create_performance_plots(self, save_path: str = None):
        """
        Create performance visualization plots
        
        Args:
            save_path: Optional path to save visualization
        """
        plt.style.use('dark_background')
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('TrackGuard Performance Metrics', fontsize=16)
        
        # Plot retention history
        ax1.plot(self.retention_history, 'g-', label='Average Retention')
        ax1.set_title('Object Retention Duration')
        ax1.set_ylabel('Frames')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot confidence history
        ax2.plot(self.confidence_history, 'b-', label='Average Confidence')
        ax2.set_title('Detection Confidence')
        ax2.set_ylabel('Confidence Score')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot active tracks
        ax3.plot(self.active_tracks_history, 'r-', label='Active Tracks')
        ax3.set_title('Active Tracks')
        ax3.set_ylabel('Count')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
    def create_realtime_overlay(self, frame: np.ndarray, metrics: Dict) -> np.ndarray:
        """
        Create realtime performance visualization overlay
        
        Args:
            frame: Current video frame
            metrics: Current performance metrics
            
        Returns:
            np.ndarray: Frame with performance overlay
        """
        height, width = frame.shape[:2]
        
        # Create overlay
        overlay = frame.copy()
        
        # Draw background for metrics
        cv2.rectangle(overlay, 
                     (10, 10), 
                     (300, 150), 
                     (0, 0, 0), 
                     -1)
        
        # Draw metrics
        y_offset = 35
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(overlay, 
                   f"Avg Retention: {metrics['avg_retention']:.1f}", 
                   (20, y_offset), 
                   font, 
                   0.6, 
                   (0, 255, 0), 
                   2)
                   
        cv2.putText(overlay, 
                   f"Confidence: {metrics['avg_confidence']:.2f}", 
                   (20, y_offset + 30), 
                   font, 
                   0.6, 
                   (255, 255, 0), 
                   2)
                   
        cv2.putText(overlay, 
                   f"Active Tracks: {metrics['active_tracks']}", 
                   (20, y_offset + 60), 
                   font, 
                   0.6, 
                   (0, 255, 255), 
                   2)
                   
        cv2.putText(overlay, 
                   f"Total Tracks: {metrics['total_tracks']}", 
                   (20, y_offset + 90), 
                   font, 
                   0.6, 
                   (255, 200, 0), 
                   2)
                   
        # Create mini performance graph
        if self.retention_history:
            graph_width = 200
            graph_height = 50
            graph_x = width - graph_width - 20
            graph_y = 20
            
            # Draw graph background
            cv2.rectangle(overlay,
                         (graph_x, graph_y),
                         (graph_x + graph_width, graph_y + graph_height),
                         (0, 0, 0),
                         -1)
            
            # Draw retention history graph
            recent_history = self.retention_history[-graph_width:]
            max_retention = max(recent_history) if recent_history else 1
            
            for i, retention in enumerate(recent_history):
                if i > 0:
                    pt1 = (graph_x + i - 1, 
                          graph_y + graph_height - int((recent_history[i-1]/max_retention) * graph_height))
                    pt2 = (graph_x + i,
                          graph_y + graph_height - int((retention/max_retention) * graph_height))
                    cv2.line(overlay, pt1, pt2, (0, 255, 0), 1)
        
        # Blend overlay with original frame
        alpha = 0.7
        output = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        return output
        
    def save_metrics(self, save_path: str):
        """
        Save performance metrics visualization
        
        Args:
            save_path: Path to save visualization image
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.create_performance_plots(str(save_path))