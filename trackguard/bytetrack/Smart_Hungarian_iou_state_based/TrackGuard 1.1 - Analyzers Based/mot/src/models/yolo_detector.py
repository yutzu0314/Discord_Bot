# src/models/yolo_detector.py

from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict
import sys

class YOLODetector:
    """YOLO model wrapper for object detection"""
    
    def __init__(self, weights_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.4, device: str = 'cuda'):
        """
        Initialize YOLO detector
        
        Args:
            weights_path: Path to YOLO weights file (YOLOv8)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for Non-Maximum Suppression
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        print(f"\n🔍 Loading YOLO model from {weights_path}")
        try:
            # Load YOLOv8 model
            self.model = YOLO(weights_path)
            print(f"✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            sys.exit(1)
        
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run YOLO detection on frame
        
        Args:
            frame: BGR image
            
        Returns:
            List[Dict]: List of detections, each with bbox, confidence, and class_id
        """
        # Run inference
        results = self.model.predict(frame, stream=True, conf=self.conf_threshold, iou=self.iou_threshold)
        
        # Process predictions
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'class_id': cls
                })
        
        # Debug info dengan progress indicator
        sys.stdout.write(f"\r🔍 Frame: Detected {len(detections)} objects (conf>{self.conf_threshold}, iou>{self.iou_threshold})")
        sys.stdout.flush()
        
        return detections

    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model
        
        Returns:
            Dict: Model information including type, classes, etc.
        """
        return {
            'model_type': self.model.type,
            'num_classes': len(self.model.names),
            'class_names': self.model.names,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'device': self.device
        }