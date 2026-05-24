import cv2
import numpy as np
import argparse
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import sys
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import csv
import gc   # Import untuk manual garbage collection

# Core components
from src.core.track_manager import TrackManager
from src.core.bbox_handler import BBoxHandler
from src.core.confidence_handler import ConfidenceHandler

# Analyzers
from src.analyzers.color_analyzer import ColorAnalyzer
from src.analyzers.shape_analyzer import ShapeAnalyzer

# Utils & Visualization
from src.utils.visualization import Visualizer
from src.evaluation.metrics import RetentionMetrics
from src.evaluation.visualizer import PerformanceVisualizer
from src.evaluation.enhanced_metrics import EnhancedMetrics

# YOLO detector
from src.models.yolo_detector import YOLODetector

# Analytics
from src.analysis.analyzer import TrackGuardAnalyzer

class RetentionMetrics:
    def __init__(self):
        self.track_history = {}
        self.current_frame = 0
        self.confidence_scores = defaultdict(list)
        self.track_durations = defaultdict(int)
        
    def update(self, detections: List[Dict]):
        self.current_frame += 1
        current_tracks = set()
        
        for det in detections:
            track_id = det['track_id']
            current_tracks.add(track_id)
            
            # Track confidence scores
            self.confidence_scores[track_id].append(det['confidence'])
            
            if track_id not in self.track_history:
                self.track_history[track_id] = {
                    'total_frames': 0,
                    'continuous_detection': 0,
                    'max_continuous_detection': 0,
                    'last_detected_frame': -1,
                    'id_switch_count': 0
                }
            
            track = self.track_history[track_id]
            
            if track['last_detected_frame'] == self.current_frame - 1:
                track['continuous_detection'] += 1
            else:
                track['continuous_detection'] = 1
                
            track['total_frames'] += 1
            track['max_continuous_detection'] = max(
                track['max_continuous_detection'],
                track['continuous_detection']
            )
            track['last_detected_frame'] = self.current_frame
            self.track_durations[track_id] += 1

    def get_track_duration(self, track_id):
        return self.track_durations.get(track_id, 0)

    def get_summary_metrics(self) -> Dict:
        if not self.track_history:
            return {
                'total_tracks': 0,
                'avg_retention': 0,
                'max_retention': 0,
                'total_bboxes': 0,
                'id_switches': 0,
                'avg_duration_per_id': 0,
                'avg_confidence': 0,
                'min_confidence': 0,
                'max_confidence': 0
            }

        total_tracks = len(self.track_history)
        retentions = [track['total_frames'] for track in self.track_history.values()]
        avg_retention = np.mean(retentions) if retentions else 0
        max_retention = max(track['max_continuous_detection'] 
                          for track in self.track_history.values())
        
        total_bboxes = sum(track['total_frames'] for track in self.track_history.values())
        id_switches = sum(track['id_switch_count'] for track in self.track_history.values())
        
        # Calculate duration metrics
        durations = list(self.track_durations.values())
        avg_duration_per_id = np.mean(durations) if durations else 0
        
        # Calculate confidence metrics
        all_confidences = [conf for conf_list in self.confidence_scores.values() 
                         for conf in conf_list]
        if all_confidences:
            avg_confidence = np.mean(all_confidences)
            min_confidence = np.min(all_confidences)
            max_confidence = np.max(all_confidences)
        else:
            avg_confidence = min_confidence = max_confidence = 0

        return {
            'total_tracks': total_tracks,
            'avg_retention': avg_retention,
            'max_retention': max_retention,
            'total_bboxes': total_bboxes,
            'id_switches': id_switches,
            'avg_duration_per_id': avg_duration_per_id,
            'avg_confidence': avg_confidence,
            'min_confidence': min_confidence,
            'max_confidence': max_confidence
        }

    def export_to_csv(self, path: Path):
        metrics = self.get_summary_metrics()
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metric', 'Value'])
            for key, value in metrics.items():
                writer.writerow([key, value])

    def cleanup_stale_tracks(self, active_track_ids: List):
        """Cleanup unused track history to prevent memory growth"""
        stale_tracks = [tid for tid in self.track_history.keys() 
                       if tid not in active_track_ids]
        
        for tid in stale_tracks:
            if tid in self.track_history:
                del self.track_history[tid]
            if tid in self.confidence_scores:
                del self.confidence_scores[tid]
            if tid in self.track_durations:
                del self.track_durations[tid]
                
        return len(stale_tracks)

class TrackGuard:
    """Main TrackGuard class that orchestrates all components"""
    
    def __init__(self, config: Dict = None, 
                bbox_handler = None,
                color_analyzer = None,
                shape_analyzer = None,
                visualization: bool = True):
        # Initialize analyzers first if not provided
        self.color_analyzer = color_analyzer if color_analyzer is not None else ColorAnalyzer()
        self.shape_analyzer = shape_analyzer if shape_analyzer is not None else ShapeAnalyzer()
        
        # Initialize BBoxHandler if not provided
        self.bbox_handler = bbox_handler if bbox_handler is not None else BBoxHandler()
        
        # Initialize confidence handler
        self.confidence_handler = ConfidenceHandler()
        
        # Configuration
        self.config = config or self._default_config()
        
        # Extract hybrid detection parameters
        self.hybrid_mode = self.config.get('hybrid_mode', True)
        self.adaptive_bbox = self.config.get('adaptive_bbox', True)
        self.occlusion_aware = self.config.get('occlusion_aware', True)
        self.two_stage_threshold = self.config.get('two_stage_threshold', True)
        
        # Initialize core components with analyzers and config
        self.track_manager = TrackManager(
            color_analyzer=self.color_analyzer,
            shape_analyzer=self.shape_analyzer,
            config=self.config,
            use_acf=True,
            use_shm=True,
            use_hbm=True
        )
        
        # Initialize visualization and metrics
        self.visualizer = Visualizer() if visualization else None
        self.metrics = RetentionMetrics()
        self.enhanced_metrics = EnhancedMetrics()
        self.performance_viz = PerformanceVisualizer()
        
        # Frame counter for periodic cleanup
        self.frame_counter = 0
        self.cleanup_interval = 100  # Melakukan cleanup setiap 100 frame
        
        # Tracking results (untuk export)
        self.tracking_results = []
        
        # Log information about enabled features
        print(f"[INFO] TrackGuard initialized with hybrid_mode={self.hybrid_mode}, "
              f"adaptive_bbox={self.adaptive_bbox}, occlusion_aware={self.occlusion_aware}, "
              f"two_stage_threshold={self.two_stage_threshold}")

    def _default_config(self) -> Dict:
        return {
            'confidence_threshold': 0.5,
            'iou_threshold': 0.5,
            'retention_threshold': 30,
            'visualization_enabled': True,
            'hybrid_mode': True,
            'adaptive_bbox': True,
            'occlusion_aware': True,
            'two_stage_threshold': True,
            'high_det_thresh': 0.5,
            'low_det_thresh': 0.25,
            'match_thresh': 0.4
        }

    def process_frame(self, frame: np.ndarray, yolo_detections: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Process a single frame with YOLO detections"""
        # Increment frame counter
        self.frame_counter += 1
        
        # Update tracking
        tracked_objects = self.track_manager.update(frame, yolo_detections)
        
        # Process each tracked object
        enhanced_detections = []
        for track in tracked_objects:
            # Get analysis region - use adaptive or standard version based on config
            if self.adaptive_bbox:
                analysis_bbox = self.bbox_handler.get_analysis_region(track['bbox'])
            else:
                # Use fixed expansion factor for non-adaptive mode
                analysis_bbox = self.bbox_handler.get_analysis_region_mot(track['bbox'])
            
            # Analyze region
            color_score = self.color_analyzer.analyze(frame, analysis_bbox, track['track_id'])
            shape_score = self.shape_analyzer.analyze(frame, analysis_bbox, track['track_id'])
            
            # Check for potential occlusion if enabled
            is_occluded = False
            if self.occlusion_aware and hasattr(self.track_manager, '_estimate_occlusion'):
                is_occluded = self.track_manager._estimate_occlusion(track)
            
            # Update confidence with occlusion awareness
            enhanced_conf = self.confidence_handler.update_confidence(
                track['track_id'],
                track['confidence'], 
                color_score, 
                shape_score,
                track['history']
            )
            
            # Create enhanced detection
            enhanced_det = {
                'bbox': track['bbox'],
                'confidence': enhanced_conf,
                'track_id': track['track_id'],
                'class_id': track['class_id'],
                'analysis_bbox': analysis_bbox,
                'is_occluded': is_occluded if self.occlusion_aware else False
            }
            enhanced_detections.append(enhanced_det)
            
            # Add to tracking results
            x1, y1, x2, y2 = track['bbox']
            self.tracking_results.append([
                self.frame_counter,       # frame
                track['track_id'],        # id
                x1,                       # x1
                y1,                       # y1 
                x2 - x1,                  # width
                y2 - y1,                  # height
                enhanced_conf,            # confidence
                track['class_id'],        # class_id
                -1                        # visibility
            ])
        
        # Update metrics
        self.metrics.update(enhanced_detections)
        frame_shape = frame.shape[:2]  # height, width
        self.enhanced_metrics.update(enhanced_detections, frame_shape)
        
        # Periodic cleanup untuk mencegah memory growth
        if self.frame_counter % self.cleanup_interval == 0:
            self.cleanup_resources()
        
        current_metrics = self.metrics.get_summary_metrics()
        
        return enhanced_detections, current_metrics
        
    def cleanup_resources(self) -> Dict:
        """
        Periodic cleanup untuk mencegah memory growth dan penurunan performa.
        
        Returns:
            Dict: Statistik tentang resources yang dibersihkan
        """
        # Dapatkan active track IDs
        active_track_ids = list(self.track_manager.tracks.keys())
        
        # Jalankan cleanup pada semua komponen
        cleaned_stats = {
            'confidence_handler': 0,
            'color_analyzer': 0,
            'shape_analyzer': 0,
            'track_manager': 0,
            'metrics': 0,
            'total': 0
        }
        
        # 1. Confidence Handler
        if hasattr(self.confidence_handler, 'cleanup_stale_tracks'):
            cleaned_stats['confidence_handler'] = self.confidence_handler.cleanup_stale_tracks(active_track_ids)
        
        # 2. Color Analyzer
        if hasattr(self.color_analyzer, 'cleanup_stale_tracks'):
            cleaned_stats['color_analyzer'] = self.color_analyzer.cleanup_stale_tracks(active_track_ids)
        
        # 3. Shape Analyzer
        if hasattr(self.shape_analyzer, 'cleanup_stale_tracks'):
            cleaned_stats['shape_analyzer'] = self.shape_analyzer.cleanup_stale_tracks(active_track_ids)
        
        # 4. Track Manager (untuk IoU cache dan resources lain)
        if hasattr(self.track_manager, 'cleanup_stale_tracks'):
            cleaned_stats['track_manager'] = self.track_manager.cleanup_stale_tracks()
            
        # 5. Retention Metrics
        if hasattr(self.metrics, 'cleanup_stale_tracks'):
            cleaned_stats['metrics'] = self.metrics.cleanup_stale_tracks(active_track_ids)
        
        # Hitung total resources yang dibersihkan
        cleaned_stats['total'] = sum(cleaned_stats.values())
        
        # Force garbage collection
        gc.collect()
        
        # Logging jika jumlah resource yang dibersihkan signifikan
        if cleaned_stats['total'] > 50:
            print(f"[INFO] Cleanup: {cleaned_stats['total']} stale resources removed")
        
        return cleaned_stats
    
    def export_results(self, output_path: Path) -> Path:
        """Export tracking results"""
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write results to file
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for row in self.tracking_results:
                writer.writerow(row)
                
        return output_path

def save_fps_history(fps_history, output_dir):
    """Save FPS history to CSV file"""
    history_file = output_dir / 'fps_history.csv'
    with open(history_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_index', 'fps'])
        for i, fps in enumerate(fps_history):
            writer.writerow([i * 30, fps])  # 30 = fps_update_interval
    print(f"FPS history saved to {history_file}")

def process_image(image_path, weights_path, output_path=None, hybrid_mode=True):
    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Cannot read image {image_path}")
        return

    # Initialize components
    yolo = YOLODetector(weights_path)
    
    # Configure TrackGuard with hybrid mode
    config = {
        'confidence_threshold': 0.25,
        'iou_threshold': 0.5,
        'retention_threshold': 30,
        'visualization_enabled': True,
        'hybrid_mode': hybrid_mode,
        'adaptive_bbox': hybrid_mode,
        'occlusion_aware': hybrid_mode,
        'two_stage_threshold': hybrid_mode
    }
    
    # Initialize with proper components for hybrid mode
    bbox_handler = BBoxHandler(expansion_factor=1.1 if hybrid_mode else 1.0)
    color_analyzer = ColorAnalyzer()
    shape_analyzer = ShapeAnalyzer()
    
    trackguard = TrackGuard(
        config=config,
        bbox_handler=bbox_handler,
        color_analyzer=color_analyzer,
        shape_analyzer=shape_analyzer
    )

    # Get detections
    yolo_detections = yolo.detect(frame)
    enhanced_detections, metrics = trackguard.process_frame(frame, yolo_detections)

    # Draw detections
    for det in enhanced_detections:
        bbox = det['bbox']
        conf = det['confidence']
        class_id = det.get('class_id', 0)
        class_name = yolo.model.names[class_id]
        is_occluded = det.get('is_occluded', False)
        
        # Draw bbox
        x1, y1, x2, y2 = map(int, bbox)
        bbox_color = (0, 0, 255) if is_occluded else (0, 255, 0)
        cv2.rectangle(frame, 
                     (x1, y1), 
                     (x2, y2), 
                     bbox_color, 2)
        
        # Draw label with confidence
        label = f"{class_name} {conf:.3f}"
        cv2.putText(frame, 
                   label, 
                   (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, 
                   bbox_color, 
                   2)
        
        # Draw occlusion indicator if applicable
        if is_occluded:
            cv2.putText(frame,
                      "OCCLUDED",
                      (x1, y1-30),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      0.5,
                      (0, 0, 255),
                      2)

    # Show image
    cv2.imshow('TrackGuard Detection', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save if output path specified
    if output_path:
        cv2.imwrite(output_path, frame)
        print(f"Result saved to {output_path}")

def main():
    print("""
    🔥 TrackGuard - Enhanced YOLO Object Tracking 🔥
    =============================================
    """)

    parser = argparse.ArgumentParser(description='TrackGuard: Enhanced YOLO Tracking')
    parser.add_argument('--input', type=str, required=True, 
                       help='Input video file or camera index (e.g., fire_video.mp4 or 0 for webcam)')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to YOLO weights file (e.g., best.pt)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output video (optional)')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Enable visualization')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--output-csv', type=str, default='metrics.csv',
                       help='Path to save metrics CSV file (default: metrics.csv)')
    parser.add_argument('--window-width', type=int, default=1280,
                       help='Width of display window (default: 1280)')
    
    # Hybrid detection parameters
    parser.add_argument('--hybrid', action='store_true', default=True,
                        help='Enable hybrid detection mode for better instance detection (default: True)')
    parser.add_argument('--adaptive-bbox', action='store_true', default=True,
                        help='Enable adaptive bounding box expansion (default: True)')
    parser.add_argument('--occlusion-aware', action='store_true', default=True,
                        help='Enable occlusion-aware tracking (default: True)')
    parser.add_argument('--two-stage', action='store_true', default=True,
                        help='Enable two-stage detection threshold (default: True)')

    try:
        args = parser.parse_args()
    except:
        print("\n❌ Error: Missing required arguments!")
        print("\n📝 Example usage:")
        print("  python main.py --input fire_video.mp4 --weights best.pt")
        print("\n🔍 Optional arguments:")
        print("  --output result.mp4    # Save processed video")
        print("  --visualize           # Enable visualization (default: True)")
        print("  --conf 0.25           # Confidence threshold (default: 0.25)")
        print("\n🔧 Hybrid detection parameters:")
        print("  --hybrid              # Enable hybrid detection mode (default: True)")
        print("  --adaptive-bbox       # Enable adaptive bounding box expansion (default: True)")
        print("  --occlusion-aware     # Enable occlusion-aware tracking (default: True)")
        print("  --two-stage           # Enable two-stage detection threshold (default: True)")
        return

    # Validate input files
    if not args.input.isdigit() and not Path(args.input).exists():
        print(f"\n❌ Error: Input video not found: {args.input}")
        return

    if not Path(args.weights).exists():
        print(f"\n❌ Error: Weights file not found: {args.weights}")
        return

    # Check if input is image
    if args.input.endswith(('.jpg', '.jpeg', '.png')):
        process_image(args.input, args.weights, args.output, args.hybrid)
        return

    try:
        # Initialize components
        print("\n🚀 Initializing TrackGuard...")
        
        # Create config with hybrid detection settings if enabled
        config = {
            'confidence_threshold': args.conf,
            'iou_threshold': 0.5,
            'retention_threshold': 30,
            'visualization_enabled': args.visualize,
            # Hybrid detection parameters
            'hybrid_mode': args.hybrid,
            'adaptive_bbox': args.adaptive_bbox,
            'occlusion_aware': args.occlusion_aware,
            'two_stage_threshold': args.two_stage
        }
        
        # Configure detector threshold params if two-stage is enabled
        if args.two_stage:
            config.update({
                'high_det_thresh': 0.5,  # For high-confidence first stage
                'low_det_thresh': 0.25,  # For lower-confidence second stage
                'match_thresh': 0.4      # Standard matching threshold
            })
        
        # Initialize YOLO detector with confidence threshold
        yolo = YOLODetector(args.weights, conf_threshold=args.conf)
        
        # Initialize BBoxHandler with adaptive expansion if enabled
        bbox_handler = BBoxHandler(
            expansion_factor=1.1 if args.adaptive_bbox else 1.0
        )
        
        # Initialize analyzers
        color_analyzer = ColorAnalyzer()
        shape_analyzer = ShapeAnalyzer()
        
        # Initialize TrackGuard with all components and config
        trackguard = TrackGuard(
            config=config,
            bbox_handler=bbox_handler,
            color_analyzer=color_analyzer,
            shape_analyzer=shape_analyzer,
            visualization=args.visualize
        )
        
        # Setup video capture
        cap = cv2.VideoCapture(int(args.input) if args.input.isdigit() else args.input)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Get initial frame for dimensions
        ret, init_frame = cap.read()
        if not ret:
            print("❌ Error: Could not read video")
            return
            
        frame_height, frame_width = init_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Setup output writer
        writer = None
        if args.output:
            writer = cv2.VideoWriter(
                args.output,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (frame_width, frame_height)
            )

        # Create window for live view with proper aspect ratio
        if args.visualize:
            cv2.namedWindow('TrackGuard Live View', cv2.WINDOW_NORMAL)
            window_width = args.window_width
            window_height = int((window_width/frame_width) * frame_height)
            cv2.resizeWindow('TrackGuard Live View', window_width, window_height)
            print(f"\nDisplay window size: {window_width}x{window_height}")
            
        print("\n▶️ Processing video...")
        print("   Press 'q' to quit")
        
        # Process frames with progress bar
        start_time = time.time()
        pbar = tqdm(total=total_frames, desc="Progress", unit="frames")
        frame_count = 0
        
        # FPS calculation variables
        total_model_time = 0
        total_full_time = 0
        model_fps_history = []
        full_fps_history = []
        fps_report_interval = 50  # Report FPS every 50 frames
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Start timing for this frame
            frame_start_time = time.time()
            
            # Process frame - measure model time separately
            model_start_time = time.time()
            
            # YOLO detection and TrackGuard processing
            yolo_detections = yolo.detect(frame)
            enhanced_detections, metrics = trackguard.process_frame(frame, yolo_detections)
            
            # Calculate model time (detection + tracking only)
            model_end_time = time.time()
            model_time = model_end_time - model_start_time
            total_model_time += model_time
            model_fps = 1.0 / model_time if model_time > 0 else 0
            model_fps_history.append(model_fps)
            
            # Draw detections on frame
            if args.visualize:
                # Standard TrackGuard visualization
                for det in enhanced_detections:
                    bbox = det['bbox']
                    conf = det['confidence']
                    track_id = det.get('track_id', 0)
                    class_id = det.get('class_id', 0)
                    is_occluded = det.get('is_occluded', False)
                    
                    # Get class name from YOLO model
                    class_name = yolo.model.names[class_id]
                    
                    # Get track duration from metrics
                    track_duration = trackguard.metrics.get_track_duration(track_id)
                    duration_seconds = track_duration / fps
                    
                    # Generate color based on track_id
                    color = (int(hash(str(track_id)) % 255), 
                            int(hash(str(track_id*2)) % 255),
                            int(hash(str(track_id*3)) % 255))
                    
                    # Use red color for occluded objects if occlusion-aware mode is enabled
                    if args.occlusion_aware and is_occluded:
                        color = (0, 0, 255)  # Red for occluded
                    
                    # Draw bbox
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw top label with confidence
                    top_label = f"{class_name} ID:{track_id} Conf:{conf:.3f}"
                    (label_width, label_height), _ = cv2.getTextSize(
                        top_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame,
                                (x1, y1 - label_height - 10),
                                (x1 + label_width, y1),
                                color, -1)
                    cv2.putText(frame,
                               top_label,
                               (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               0.5, (255, 255, 255), 2)
                               
                    # Draw duration label
                    bottom_label = f"Duration: {duration_seconds:.1f}s"
                    cv2.putText(frame,
                               bottom_label,
                               (x1, y2 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               0.5,
                               color,
                               2)
                    
                    # Tambahkan indikator oklusi jika diaktifkan dan terdeteksi oklusi
                    if args.occlusion_aware and is_occluded:
                        cv2.putText(frame,
                                  "OCCLUDED",
                                  (x1, y1-25),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5,
                                  (0, 0, 255),
                                  2)
                        
                        # Draw diagonal lines untuk indikator oklusi
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.line(frame, (x1, y2), (x2, y1), (0, 0, 255), 2)
                
                # Add information overlay - background for text
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (300, 190), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                
                # Keep FPS history limited
                if len(model_fps_history) > 100:
                    model_fps_history = model_fps_history[-100:]
                
                # Display Model FPS (detection + tracking only)
                avg_model_fps = sum(model_fps_history[-30:]) / min(30, len(model_fps_history))
                cv2.putText(frame, 
                           f"Model FPS: {avg_model_fps:.1f}", 
                           (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, 
                           (0, 255, 0), 
                           2)
                
                # Calculate full frame time (including visualization)
                frame_end_time = time.time()
                full_time = frame_end_time - frame_start_time
                total_full_time += full_time
                full_fps = 1.0 / full_time if full_time > 0 else 0
                full_fps_history.append(full_fps)
                
                # Keep FPS history limited
                if len(full_fps_history) > 100:
                    full_fps_history = full_fps_history[-100:]
                
                # Display Full FPS (including visualization)
                avg_full_fps = sum(full_fps_history[-30:]) / min(30, len(full_fps_history))
                cv2.putText(frame, 
                           f"Total FPS: {avg_full_fps:.1f}", 
                           (20, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, 
                           (0, 255, 0), 
                           2)
                
                # Draw object counts
                total_objects = len(enhanced_detections)
                cv2.putText(frame,
                          f"Objects: {total_objects}", 
                          (20, 100),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.7,
                          (0, 255, 0),
                          2)
                
                # Display frame counter
                cv2.putText(frame,
                          f"Frame: {frame_count}/{total_frames}", 
                          (20, 130),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.7,
                          (0, 255, 0),
                          2)
                
                # Display processing times
                cv2.putText(frame, 
                          f"Model time: {model_time*1000:.1f} ms", 
                          (20, 160), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, 
                          (0, 255, 0), 
                          2)
                
                cv2.putText(frame, 
                          f"Total time: {full_time*1000:.1f} ms", 
                          (20, 190), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, 
                          (0, 255, 0), 
                          2)
                
                # Display hybrid mode status if enabled
                if args.hybrid:
                    hybrid_mode_text = "Hybrid Detection: ON"
                    if args.occlusion_aware and any(det.get('is_occluded', False) for det in enhanced_detections):
                        hybrid_mode_text += " (Occlusion Active)"
                    
                    cv2.putText(frame, 
                              hybrid_mode_text, 
                              (20, 220), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, 
                              (0, 255, 0), 
                              2)
            
                # Show frame with visualizations
                cv2.imshow('TrackGuard Live View', frame)
            
            # Write frame if output is enabled
            if writer is not None:
                writer.write(frame)
            
            # Report FPS every N frames
            if frame_count % fps_report_interval == 0 and frame_count > 0:
                avg_model_fps = sum(model_fps_history[-fps_report_interval:]) / min(fps_report_interval, len(model_fps_history))
                avg_full_fps = sum(full_fps_history[-fps_report_interval:]) / min(fps_report_interval, len(full_fps_history))
                elapsed_time = time.time() - start_time
                eta = (total_frames - frame_count) * elapsed_time / frame_count
                
                # Count occluded objects if occlusion aware mode is enabled
                occluded_count = 0
                if args.occlusion_aware:
                    occluded_count = sum(1 for det in enhanced_detections if det.get('is_occluded', False))
                
                print(f"Frame: {frame_count}/{total_frames} | " 
                      f"Model FPS: {avg_model_fps:.1f} | "
                      f"Total FPS: {avg_full_fps:.1f} | "
                      f"Objects: {len(enhanced_detections)}")
                
                if args.occlusion_aware:
                    print(f"Hybrid Mode: {'ON' if args.hybrid else 'OFF'} | "
                         f"Occlusions: {occluded_count}")
                
                print(f"ETA: {eta:.0f}s")
                
            frame_count += 1
            pbar.update(1)
            
            if args.visualize and cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⚠️ Processing interrupted by user")
                break

    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup and save results
        pbar.close()
        end_time = time.time()
        duration = end_time - start_time
        
        # Create output directories
        output_dir = Path('results')
        metrics_dir = output_dir / 'metrics'
        viz_dir = output_dir / 'visualizations'
        
        metrics_dir.mkdir(parents=True, exist_ok=True)
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FPS history if available
        if 'full_fps_history' in locals() and full_fps_history:
            save_fps_history(full_fps_history, output_dir)
        
        # Get final metrics
        metrics = trackguard.metrics.get_summary_metrics()
        
        # Calculate average FPS metrics
        if frame_count > 0:
            avg_model_fps = frame_count / total_model_time if total_model_time > 0 else 0
            avg_full_fps = frame_count / total_full_time if total_full_time > 0 else 0
            final_fps = frame_count / duration if duration > 0 else 0
        else:
            avg_model_fps = avg_full_fps = final_fps = 0
        
        # Calculate object count metrics
        total_tracking_instances = len(trackguard.tracking_results)
        try:
            # Ini menggunakan enhanced_metrics jika tersedia
            enhanced_metrics = trackguard.enhanced_metrics.get_summary()
            avg_objects_per_frame = enhanced_metrics['avg_objects_per_frame']
        except:
            # Fallback menggunakan total_bboxes dibagi frame_count
            if frame_count > 0:
                avg_objects_per_frame = metrics['total_bboxes'] / frame_count
            else:
                avg_objects_per_frame = 0
        
        # Print ringkasan tracking seperti di BoTSORT
        print(f"\n✅ Tracking completed successfully!")
        if args.output:
            print(f"  • Results saved to: {args.output}")
        else:
            print(f"  • Results saved to: metrics/")
        print(f"  • Processed {frame_count} frames")
        print(f"  • Detected {total_tracking_instances} tracking instances")
        print(f"  • Average {avg_objects_per_frame:.1f} objects per frame")
        
        # Print hybrid mode info if enabled
        if args.hybrid:
            print("\n🧩 Hybrid Detection Mode:")
            print(f"  • Adaptive BBox: {'Enabled' if args.adaptive_bbox else 'Disabled'}")
            print(f"  • Occlusion Aware: {'Enabled' if args.occlusion_aware else 'Disabled'}")
            print(f"  • Two-Stage Threshold: {'Enabled' if args.two_stage else 'Disabled'}")
        
        # Print summary with confidence metrics
        print("\n📊 Processing Summary:")
        print(f"  • Processed Frames: {frame_count}")
        print(f"  • Time Taken: {duration:.2f} seconds")
        
        # Handle division by zero cases
        if frame_count > 0:
            print(f"  • Model processing time: {total_model_time*1000/frame_count:.1f} ms per frame")
            print(f"  • Total processing time: {total_full_time*1000/frame_count:.1f} ms per frame")
        else:
            print(f"  • Model processing time: N/A (no frames processed)")
            print(f"  • Total processing time: N/A (no frames processed)")
            
        print(f"  • Average Model FPS: {avg_model_fps:.1f} (detection + tracking only)")
        print(f"  • Average Total FPS: {avg_full_fps:.1f} (including visualization)")
        print(f"  • Average End-to-End FPS: {final_fps:.1f} (including all overhead)")
        
        # Print confidence metrics
        print("\n📊 Confidence Metrics:")
        print(f"  • Average Confidence: {metrics['avg_confidence']:.3f}")
        print(f"  • Min Confidence: {metrics['min_confidence']:.3f}")
        print(f"  • Max Confidence: {metrics['max_confidence']:.3f}")

        # Print tracking metrics
        print("\n📈 Tracking Metrics:")
        print(f"  • Total Tracks: {metrics['total_tracks']}")
        print(f"  • Average Retention: {metrics['avg_retention']:.2f} frames")
        print(f"  • Max Retention: {metrics['max_retention']} frames")
        print(f"  • Total Bounding Boxes: {metrics['total_bboxes']}")
        print(f"  • ID Switches: {metrics['id_switches']}")
        print(f"  • Avg Duration per ID: {metrics['avg_duration_per_id']:.2f}s")
        
        # Export metrics to CSV
        csv_path = metrics_dir / 'metrics.csv' if not args.output_csv else Path(args.output_csv)
        trackguard.metrics.export_to_csv(csv_path)
        
        # Export visualizations if analyzer is available
        try:
            analyzer = TrackGuardAnalyzer()
            performance_fig = analyzer.create_performance_plots()
            performance_fig.savefig(viz_dir / 'performance_plots.png')
            roi_fig = analyzer.create_roi_visualization()
            roi_fig.savefig(viz_dir / 'roi_analysis.png')
        except Exception as viz_error:
            print(f"  • Could not create visualizations: {str(viz_error)}")
        
        print(f"\n📁 Results saved:")
        print(f"  • Metrics: {csv_path}")
        print(f"  • Visualizations: {viz_dir}")
        
        if writer is not None:
            writer.release()
            print(f"  • Output video: {args.output}")
            
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n✨ TrackGuard finished!")

if __name__ == "__main__":
    main()