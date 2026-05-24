"""
YOLO11x Adapter for ByteTrack
==============================

Adapter untuk mengkonversi output YOLO11x (Ultralytics) ke format yang bisa
digunakan oleh ByteTracker dari repo ByteTrack.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
import cv2


class YOLO11Adapter:
    """
    Adapter untuk YOLO11x model yang mengkonversi output ke format ByteTrack.
    """
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize YOLO11x adapter.
        
        Args:
            model_path: Path ke model YOLO11x (.pt file)
            device: Device untuk inference ('cuda' atau 'cpu')
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.model.to(device)
            self.device = device
            print(f"✓ YOLO11x model loaded from {model_path}")
        except ImportError:
            raise ImportError(
                "ultralytics tidak ditemukan. Install dengan: pip install ultralytics"
            )
        except Exception as e:
            raise RuntimeError(f"Error loading YOLO11x model: {e}")
    
    def preprocess(self, img: np.ndarray, img_size: Tuple[int, int] = (640, 640)):
        """
        Preprocess image untuk YOLO11x (sudah ditangani oleh model.predict).
        Method ini hanya untuk reference.
        """
        return img
    
    def detect(self, img: np.ndarray, conf_threshold: float = 0.01) -> np.ndarray:
        """
        Run detection dengan YOLO11x dan konversi ke format ByteTrack.
        
        Format output ByteTrack: [x1, y1, x2, y2, obj_conf, class_conf, class_id]
        Atau: [x1, y1, x2, y2, score] jika hanya 5 kolom
        
        Args:
            img: Input image (BGR format, numpy array)
            conf_threshold: Confidence threshold untuk filtering
            
        Returns:
            detections: numpy array shape (N, 7) atau (N, 5)
                       Format: [x1, y1, x2, y2, obj_conf, class_conf, class_id]
        """
        # YOLO11x expects RGB, tapi model.predict bisa handle BGR juga
        results = self.model.predict(
            img,
            conf=conf_threshold,
            iou=0.65,
            verbose=False,
            device=self.device
        )
        
        if len(results) == 0 or results[0].boxes is None:
            return np.empty((0, 7), dtype=np.float32)
        
        boxes = results[0].boxes
        
        # Extract data
        if boxes.xyxy is not None and len(boxes.xyxy) > 0:
            # Get boxes in xyxy format
            xyxy = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            
            # Get confidences
            conf = boxes.conf.cpu().numpy()  # Object confidence
            
            # Get class predictions
            cls = boxes.cls.cpu().numpy().astype(np.int32)  # Class IDs
            
            # IMPORTANT: ByteTrack calculates score as obj_conf * class_conf
            # YOLO11x's conf is already the final confidence score, so we need to avoid double-squaring
            # Solution: Set class_conf = 1.0 so final score = obj_conf * 1.0 = obj_conf = conf
            # Alternative: Set both to sqrt(conf) so sqrt(conf) * sqrt(conf) = conf
            # We'll use the simpler approach: obj_conf = conf, class_conf = 1.0
            
            # Combine: [x1, y1, x2, y2, obj_conf, class_conf, class_id]
            detections = np.zeros((len(xyxy), 7), dtype=np.float32)
            detections[:, :4] = xyxy
            detections[:, 4] = conf  # obj_conf: use YOLO11x confidence directly
            detections[:, 5] = 1.0  # class_conf: set to 1.0 so final score = obj_conf * 1.0 = conf
            detections[:, 6] = cls  # class_id
            
            return detections
        else:
            return np.empty((0, 7), dtype=np.float32)
    
    def detect_batch(self, imgs: List[np.ndarray], conf_threshold: float = 0.01) -> List[np.ndarray]:
        """
        Batch detection (optional, bisa digunakan untuk optimasi).
        """
        detections_list = []
        for img in imgs:
            det = self.detect(img, conf_threshold)
            detections_list.append(det)
        return detections_list
