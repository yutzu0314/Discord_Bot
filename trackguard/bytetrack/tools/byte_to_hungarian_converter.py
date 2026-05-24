"""
ByteTrack to Hungarian Evaluator Converter
==========================================

Converter untuk mengkonversi output ByteTracker ke format yang bisa diterima
oleh SimpleMOTACalculator di mot_evaluator_hungarian.py
"""

import numpy as np
from typing import List, Dict, Any


class ByteTrackToHungarianConverter:
    """
    Converter untuk mengkonversi ByteTracker output ke format evaluator.
    """
    
    @staticmethod
    def convert_tracks_to_evaluator_format(
        byte_tracks: List[Any],
        frame_id: int
    ) -> List[Dict]:
        """
        Convert ByteTracker STrack objects ke format evaluator.
        
        Evaluator mengharapkan format:
        - List of objects dengan attribute `current_detection` yang berisi:
          - `bbox`: [x1, y1, x2, y2]
          - `confidence`: float
        - Atau list of dicts dengan keys:
          - `track_id`: int
          - `bbox`: [x1, y1, x2, y2]
          - `confidence`: float
        
        Args:
            byte_tracks: List of STrack objects dari ByteTracker
            frame_id: Frame ID (optional, untuk logging)
            
        Returns:
            tracking_results: List of dicts dalam format evaluator
        """
        tracking_results = []
        
        for track in byte_tracks:
            # ByteTracker STrack memiliki:
            # - track.track_id
            # - track.tlbr (top-left bottom-right: [x1, y1, x2, y2])
            # - track.tlwh (top-left width height: [x1, y1, w, h])
            # - track.score
            # Note: ByteTrack tidak menyimpan class_id, jadi kita asumsikan class_id=1 (person)
            # karena MOT17 hanya track person
            
            # Convert tlbr ke format bbox [x1, y1, x2, y2]
            bbox_xyxy = track.tlbr.copy()  # Already in [x1, y1, x2, y2] format
            
            # Create dict dalam format evaluator
            track_dict = {
                'track_id': track.track_id,
                'bbox': bbox_xyxy.tolist(),  # [x1, y1, x2, y2]
                'bbox_xyxy': bbox_xyxy.tolist(),  # Alias for compatibility
                'confidence': float(track.score),
                'conf': float(track.score),  # Alias
                'class_id': 1,  # MOT17 hanya track person (class_id=1)
                'category_id': 1,  # Alias
                'frame_id': frame_id
            }
            
            # Create object-like structure dengan current_detection attribute
            # Evaluator kadang cek hasattr(track, 'current_detection')
            class TrackObject:
                def __init__(self, track_dict):
                    self.track_id = track_dict['track_id']
                    self.class_id = track_dict['class_id']
                    self.category_id = track_dict['category_id']
                    self.current_detection = {
                        'bbox': track_dict['bbox'],
                        'confidence': track_dict['confidence'],
                        'class_id': track_dict['class_id']
                    }
            
            track_obj = TrackObject(track_dict)
            tracking_results.append(track_obj)
        
        return tracking_results
    
    @staticmethod
    def convert_detections_to_evaluator_format(
        detections: np.ndarray,
        track_ids: np.ndarray = None
    ) -> List[Dict]:
        """
        Convert raw detections (array) ke format evaluator.
        
        Args:
            detections: numpy array shape (N, 4) dengan [x1, y1, x2, y2]
            track_ids: numpy array shape (N,) dengan track IDs (optional)
            
        Returns:
            tracking_results: List of dicts dalam format evaluator
        """
        tracking_results = []
        
        if detections is None or len(detections) == 0:
            return tracking_results
        
        for i, det in enumerate(detections):
            track_id = track_ids[i] if track_ids is not None else i
            
            track_dict = {
                'track_id': int(track_id),
                'bbox': det[:4].tolist(),  # [x1, y1, x2, y2]
                'confidence': float(det[4]) if len(det) > 4 else 0.8,
                'bbox_xyxy': det[:4].tolist()
            }
            
            # Create object-like structure
            class TrackObject:
                def __init__(self, track_dict):
                    self.track_id = track_dict['track_id']
                    self.current_detection = {
                        'bbox': track_dict['bbox'],
                        'confidence': track_dict['confidence']
                    }
            
            track_obj = TrackObject(track_dict)
            tracking_results.append(track_obj)
        
        return tracking_results
