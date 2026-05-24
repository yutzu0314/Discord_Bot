import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class TrackGuardAnalyzer:
    def __init__(self):
        # YOLO metrics
        self.yolo_metrics = {
            'tracks': 696,
            'duration': 0.043,
            'processing_time': 13.845,
            'total_frames': 412
        }
        
        # TrackGuard metrics
        self.tg_metrics = {
            'tracks': 32,
            'duration': 0.497,
            'processing_time': 14.401,
            'total_frames': 412
        }

    def create_performance_plots(self):
        """Create comprehensive performance visualization"""
        plt.style.use('default')
        sns.set_theme()
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('TrackGuard Performance Analysis', fontsize=16, y=0.95)
        
        # Methods labels
        methods = ['YOLO', 'YOLO + TG']
        
        # 1. Track Count Reduction
        tracks = [self.yolo_metrics['tracks'], self.tg_metrics['tracks']]
        ax1.bar(methods, tracks, color=['#FF9999', '#99FF99'])
        ax1.set_title('Track Count Reduction')
        ax1.set_ylabel('Number of Tracks')
        ax1.grid(True, linestyle='--', alpha=0.7)
        for i, v in enumerate(tracks):
            ax1.text(i, v, str(v), ha='center', va='bottom')
        
        # 2. Duration Improvement
        durations = [self.yolo_metrics['duration'], self.tg_metrics['duration']]
        ax2.bar(methods, durations, color=['#FF9999', '#99FF99'])
        ax2.set_title('Average Duration per Track')
        ax2.set_ylabel('Duration (seconds)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        for i, v in enumerate(durations):
            ax2.text(i, v, f'{v:.3f}s', ha='center', va='bottom')
        
        # 3. Track Timeline
        # YOLO tracks (many short ones)
        ax3.set_title('YOLO: Multiple Short Tracks')
        for i in range(20):  # 20 sample tracks
            start = np.random.uniform(0, 13)
            duration = self.yolo_metrics['duration']
            # Gunakan rectangle untuk visualisasi yang lebih jelas
            rect = plt.Rectangle(
                (start, i-0.25),  # (x, y)
                duration,         # width
                0.5,             # height
                color='red',
                alpha=0.5        # Reduced transparency
            )
            ax3.add_patch(rect)
            # Tambah titik start/end yang lebih besar
            ax3.scatter([start, start+duration], [i, i], 
                       color='darkred', s=30, zorder=5)
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Track ID')
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.set_ylim(-0.5, 20.5)
        ax3.set_xlim(0, 13)
        
        # TG tracks (fewer, longer ones)
        ax4.set_title('TrackGuard: Stable Long Tracks')
        for i in range(5):  # 5 sample tracks
            start = np.random.uniform(0, 13)
            duration = self.tg_metrics['duration']
            # Gunakan rectangle
            rect = plt.Rectangle(
                (start, i-0.25),
                duration,
                0.5,
                color='green',
                alpha=0.5
            )
            ax4.add_patch(rect)
            # Tambah titik start/end yang lebih besar
            ax4.scatter([start, start+duration], [i, i], 
                       color='darkgreen', s=30, zorder=5)
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Track ID')
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.set_ylim(-0.5, 5.5)
        ax4.set_xlim(0, 13)
        
        plt.tight_layout()
        return fig

    def create_roi_visualization(self):
        """Create ROI metrics visualization"""
        # Calculate ROI metrics
        roi_data = {
            'track_reduction_percent': ((self.yolo_metrics['tracks'] - self.tg_metrics['tracks']) / 
                                      self.yolo_metrics['tracks']) * 100,
            'duration_increase_percent': ((self.tg_metrics['duration'] - self.yolo_metrics['duration']) / 
                                        self.yolo_metrics['duration']) * 100,
            'processing_overhead_percent': ((self.tg_metrics['processing_time'] - self.yolo_metrics['processing_time']) / 
                                          self.yolo_metrics['processing_time']) * 100,
            'efficiency_ratio': (self.tg_metrics['duration'] / self.yolo_metrics['duration'])
        }
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('TrackGuard ROI Analysis', fontsize=16)
        
        # Improvement Metrics
        metrics = ['Track\nReduction', 'Duration\nIncrease', 'Processing\nOverhead']
        values = [roi_data['track_reduction_percent'],
                 roi_data['duration_increase_percent'],
                 roi_data['processing_overhead_percent']]
        
        colors = ['green' if v > 0 else 'red' for v in values]
        ax1.bar(metrics, values, color=colors)
        ax1.set_title('Performance Metrics (%)')
        ax1.set_ylabel('Percentage')
        ax1.grid(True, linestyle='--', alpha=0.7)
        for i, v in enumerate(values):
            ax1.text(i, v, f'{v:.1f}%', ha='center', va='bottom' if v > 0 else 'top')
        
        # Efficiency Comparison
        labels = ['YOLO', 'YOLO + TG']
        sizes = [1, roi_data['efficiency_ratio']]
        ax2.bar(labels, sizes, color=['#FF9999', '#99FF99'])
        ax2.set_title('Track Duration Efficiency')
        ax2.set_ylabel('Relative Efficiency')
        ax2.grid(True, linestyle='--', alpha=0.7)
        for i, v in enumerate(sizes):
            ax2.text(i, v, f'{v:.1f}x', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig

    def generate_summary_table(self):
        """Generate summary table of metrics"""
        roi_data = {
            'track_reduction_percent': ((self.yolo_metrics['tracks'] - self.tg_metrics['tracks']) / 
                                      self.yolo_metrics['tracks']) * 100,
            'duration_increase_percent': ((self.tg_metrics['duration'] - self.yolo_metrics['duration']) / 
                                        self.yolo_metrics['duration']) * 100,
            'processing_overhead_percent': ((self.tg_metrics['processing_time'] - self.yolo_metrics['processing_time']) / 
                                          self.yolo_metrics['processing_time']) * 100,
            'efficiency_ratio': (self.tg_metrics['duration'] / self.yolo_metrics['duration'])
        }
        
        summary_data = {
            'Metric': [
                'Total Tracks',
                'Avg Duration per Track (s)',
                'Processing Time (s)',
                'Track Reduction (%)',
                'Duration Increase (%)',
                'Processing Overhead (%)',
                'Efficiency Ratio'
            ],
            'YOLO': [
                self.yolo_metrics['tracks'],
                f"{self.yolo_metrics['duration']:.3f}",
                f"{self.yolo_metrics['processing_time']:.3f}",
                '-',
                '-',
                '-',
                '1.0x'
            ],
            'YOLO + TG': [
                self.tg_metrics['tracks'],
                f"{self.tg_metrics['duration']:.3f}",
                f"{self.tg_metrics['processing_time']:.3f}",
                f"{roi_data['track_reduction_percent']:.1f}%",
                f"{roi_data['duration_increase_percent']:.1f}%",
                f"{roi_data['processing_overhead_percent']:.1f}%",
                f"{roi_data['efficiency_ratio']:.1f}x"
            ]
        }
        
        return pd.DataFrame(summary_data)