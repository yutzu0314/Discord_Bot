"""
YOLOv8x Detector for High-Accuracy Pedestrian Detection in GBC-MOT POC
Uses centralized settings from utils.settings
"""

import torch
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import time
from pathlib import Path

# Try import ultralytics
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("⚠️ ultralytics not installed. Please run: pip install ultralytics")

class YOLOv8Detector:
    """
    YOLOv8x-based high-accuracy pedestrian detector
    Optimized untuk detection quality pada complex scenes
    """
    
    def __init__(self, **kwargs):

        """
        Initialize YOLOv8x detector using centralized settings
        """
        from utils.settings import SETTINGS
        
        # Get config from centralized settings
        config = SETTINGS.get_detector_config()
        
        # Override with any kwargs provided
        for key, value in kwargs.items():
            if key in config:
                config[key] = value
        
        # Apply configuration
        self.model_variant = config['model_variant']
        self.confidence_threshold = config['confidence_threshold']
        self.nms_threshold = config['nms_threshold']
        self.device = config['device']
        self.input_size = config['input_size']
        self.half_precision = config['half_precision']
        
        # Load YOLOv8x model
        self.model = self._load_model()
        
        # COCO person class ID (same as SSD)
        self.PERSON_CLASS_ID = 0  # Person class di COCO
        
        # Performance tracking
        self.detection_times = []
        self.total_detections = 0
        
        print(f"✓ YOLOv8x detector initialized on {self.device}")
        print(f"  Model: {self.model_variant}")
        print(f"  Input size: {self.input_size}")
        print(f"  Confidence: {self.confidence_threshold}")
        print(f"  Half precision: {self.half_precision}")
        print(f"  🎯 Using centralized settings")
        
    def _load_model(self) -> YOLO:
        """Load YOLOv8x model dengan optimasi"""
        try:
            # Load model
            model = YOLO(self.model_variant)
            
            # Optimasi model
            if self.device == 'cuda':
                model.to(self.device)
                
                # Enable half precision jika supported
                if self.half_precision:
                    try:
                        model.half()
                        print("✓ Half precision (FP16) enabled")
                    except Exception as e:
                        print(f"⚠️ Half precision failed: {e}")
                        self.half_precision = False
            
            # Set model ke eval mode
            model.model.eval()
            
            # Warmup dengan dummy inference (more iterations for larger model)
            dummy_input = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            for _ in range(5):  # More warmup for YOLOv8x
                _ = model(dummy_input, verbose=False)
            print("✓ YOLOv8x warmup completed")
            
            return model
            
        except Exception as e:
            print(f"✗ Error loading YOLOv8x model: {e}")
            raise
    
    def detect(self, image: np.ndarray, return_stats: bool = False) -> List[Dict]:
        """
        Detect pedestrians dalam image dengan YOLOv8x
        
        Args:
            image: Input image (H, W, C) dalam BGR format
            return_stats: Whether to return detection statistics
            
        Returns:
            List of detection dictionaries
        """
        start_time = time.time()
        
        # YOLOv8x inference
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            classes=[self.PERSON_CLASS_ID],  # Only detect persons
            verbose=False,
            imgsz=self.input_size
        )
        
        # Process hasil
        detections = self._process_results(results[0], image.shape[:2])
        
        # Track performance
        detection_time = time.time() - start_time
        self.detection_times.append(detection_time)
        self.total_detections += len(detections)
        
        if return_stats:
            stats = {
                'detection_time': detection_time,
                'num_detections': len(detections),
                'input_size': self.input_size,
                'model_variant': self.model_variant
            }
            return detections, stats
        
        return detections
    
    def _process_results(self, result, image_shape: Tuple[int, int]) -> List[Dict]:
        """
        Process YOLOv8x results ke format detection standard
        
        Args:
            result: YOLOv8x result object
            image_shape: (height, width) dari input image
            
        Returns:
            List of processed detections
        """
        detections = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        # Extract boxes, scores, classes
        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        for i in range(len(boxes)):
            # Ambil hanya person detections
            if classes[i] != self.PERSON_CLASS_ID:
                continue
            
            bbox = boxes[i]
            confidence = scores[i]
            
            # Extract coordinates
            x1, y1, x2, y2 = bbox.astype(int)
            
            # Clamp ke image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image_shape[1], x2)
            y2 = min(image_shape[0], y2)
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Calculate center dan size
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            width = x2 - x1
            height = y2 - y1
            
            # Basic size filtering (less aggressive for better recall)
            if width < 8 or height < 15:  # Relaxed size constraints
                continue
            
            detection = {
                'bbox': [x1, y1, x2, y2],
                'confidence': float(confidence),
                'center': [center_x, center_y],
                'size': [width, height],
                'aspect_ratio': height / width if width > 0 else 0
            }
            
            detections.append(detection)
        
        return detections
    
    def detect_batch(self, images: List[np.ndarray]) -> List[List[Dict]]:
        """
        Batch detection untuk multiple images (optimal untuk accuracy)
        
        Args:
            images: List of input images
            
        Returns:
            List of detection lists untuk each image
        """
        start_time = time.time()
        
        # YOLOv8x mendukung batch inference
        results = self.model(
            images,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            classes=[self.PERSON_CLASS_ID],
            verbose=False,
            imgsz=self.input_size
        )
        
        # Process semua results
        batch_detections = []
        for i, result in enumerate(results):
            image_shape = images[i].shape[:2]
            detections = self._process_results(result, image_shape)
            batch_detections.append(detections)
        
        # Track performance
        batch_time = time.time() - start_time
        total_detections = sum(len(dets) for dets in batch_detections)
        
        self.detection_times.append(batch_time)
        self.total_detections += total_detections
        
        return batch_detections
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict], 
                           show_fps: bool = True, show_stats: bool = True) -> np.ndarray:
        """
        Visualize detections dengan performance info
        
        Args:
            image: Input image
            detections: List of detections
            show_fps: Whether to show FPS info
            
        Returns:
            Image dengan drawn bounding boxes
        """
        vis_image = image.copy()
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            x1, y1, x2, y2 = bbox
            
            # Color berdasarkan confidence
            if confidence > 0.7:
                color = (0, 255, 0)      # Green - high confidence
            elif confidence > 0.5:
                color = (0, 255, 255)    # Yellow - medium confidence
            else:
                color = (0, 165, 255)    # Orange - low confidence
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence
            label = f"Person: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 5), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Show performance info
        if show_fps and self.detection_times:
            avg_time = np.mean(self.detection_times[-10:])  # Last 10 frames
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            perf_text = f"YOLOv8x: {len(detections)} dets, {fps:.1f} FPS"
            cv2.putText(vis_image, perf_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_image
    
    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics"""
        if not self.detection_times:
            return {
                'avg_detection_time': 0,
                'avg_fps': 0,
                'total_detections': 0,
                'model_info': self.get_model_info()
            }
        
        avg_time = np.mean(self.detection_times)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'avg_detection_time': avg_time,
            'avg_fps': avg_fps,
            'min_time': np.min(self.detection_times),
            'max_time': np.max(self.detection_times),
            'total_detections': self.total_detections,
            'total_inferences': len(self.detection_times),
            'detections_per_inference': self.total_detections / len(self.detection_times),
            'model_info': self.get_model_info()
        }
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        # Estimate model size
        model_size_mb = 68.0 if 'x' in self.model_variant else 6.0  # YOLOv8x size
        
        return {
            'model_name': f'YOLOv8x',
            'model_variant': self.model_variant,
            'device': self.device,
            'input_size': self.input_size,
            'confidence_threshold': self.confidence_threshold,
            'nms_threshold': self.nms_threshold,
            'half_precision': self.half_precision,
            'estimated_size_mb': model_size_mb,
            'person_class_id': self.PERSON_CLASS_ID,
            'optimization_level': 'high_accuracy'
        }
    
    def optimize_for_accuracy(self):
        """Apply accuracy optimizations"""
        print("🎯 Applying accuracy optimizations...")
        
        # Set optimal batch size
        if hasattr(self.model, 'model'):
            try:
                # Enable optimized operations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                print("✓ CUDNN optimizations enabled")
            except:
                pass
        
        # Additional warmup for larger model
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(5):  # More warmup iterations for YOLOv8x
            _ = self.detect(dummy_image)
        
        print("✓ YOLOv8x warmup completed")
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.detection_times = []
        self.total_detections = 0


# Backward compatibility - alias untuk SSDDetector
class SSDDetector(YOLOv8Detector):
    """Backward compatibility alias"""
    def __init__(self, **kwargs):
        # Map SSD parameters ke YOLOv8x parameters
        if 'confidence_threshold' not in kwargs:
            kwargs['confidence_threshold'] = 0.2  # Optimized for recall
        if 'nms_threshold' not in kwargs:
            kwargs['nms_threshold'] = 0.3  # Optimized for recall
            
        super().__init__(**kwargs)
        print("🔄 Using YOLOv8x as SSD replacement for better accuracy")


# Example usage dan testing
if __name__ == "__main__":
    # Test YOLOv8x detector
    detector = YOLOv8Detector(confidence_threshold=0.2, nms_threshold=0.3)
    
    # Print model info
    info = detector.get_model_info()
    print("YOLOv8x Detector Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Apply optimizations
    detector.optimize_for_accuracy()
    
    # Test dengan dummy image
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    print("\nTesting YOLOv8x detection accuracy...")
    
    # Benchmark speed
    times = []
    for i in range(10):
        start = time.time()
        detections = detector.detect(dummy_image)
        times.append(time.time() - start)
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    print(f"Average detection time: {avg_time:.4f}s")
    print(f"Estimated FPS: {fps:.1f}")
    print(f"Found {len(detections)} detections in test")
    
    # Performance stats
    perf_stats = detector.get_performance_stats()
    print(f"Performance stats: {perf_stats}")