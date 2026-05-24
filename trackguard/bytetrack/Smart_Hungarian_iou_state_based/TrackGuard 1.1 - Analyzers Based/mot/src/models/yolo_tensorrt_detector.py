# src/models/yolo_tensorrt_detector.py
from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict

class YOLOTensorRTDetector:
    """
    YOLOv8 TensorRT detector untuk TrackGuard
    Menggunakan Ultralytics API untuk inference dengan engine TensorRT
    """
    
    def __init__(self, engine_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Initialize YOLOv8 TensorRT detector
        
        Args:
            engine_path: Path ke TensorRT engine file
            conf_threshold: Threshold confidence untuk deteksi
            iou_threshold: Threshold IoU untuk NMS
        """
        self.engine_path = engine_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load model
        print(f"Loading TensorRT engine from {engine_path}")
        self.model = YOLO(engine_path)
        
        # Simpan model attributes
        self.model_type = "tensorrt"
        if hasattr(self.model, 'names'):
            self.class_names = self.model.names
        else:
            self.class_names = {}
        
        print(f"Model loaded successfully with {len(self.class_names)} classes")
        
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in a frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List[Dict]: List of detections in TrackGuard format
        """
        # Run inference
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        
        # Convert results to TrackGuard format
        detections = []
        
        if results and len(results) > 0:
            # Get first result (for single frame)
            result = results[0]
            
            # Get boxes
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Get box coordinates (xyxy format)
                xyxy = boxes.xyxy[i].cpu().numpy()
                
                # Get confidence
                conf = float(boxes.conf[i].cpu().numpy())
                
                # Get class ID
                cls_id = int(boxes.cls[i].cpu().numpy())
                
                # Add detection in TrackGuard format
                detection = {
                    'bbox': xyxy.tolist(),  # [x1, y1, x2, y2]
                    'confidence': conf,
                    'class_id': cls_id
                }
                
                detections.append(detection)
        
        return detections
    
    def cleanup(self):
        """Clean up resources"""
        # Free GPU memory
        if hasattr(self, 'model'):
            del self.model
            
        # Force CUDA cache cleanup
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        print("TensorRT detector resources released")