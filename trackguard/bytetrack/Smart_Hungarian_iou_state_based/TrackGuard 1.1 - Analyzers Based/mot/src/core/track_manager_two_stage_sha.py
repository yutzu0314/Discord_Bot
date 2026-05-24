"""
Track Manager Two-Stage State-Based dengan Smart Hungarian TrackGuard
=====================================================================

State-based two-stage tracking:
- Stage 1: Active tracks (state='active') → Smart Hungarian matching
- Stage 2: Ghost tracks (state='ghost') → Ghost recovery matching

Menggunakan SmartHungarianTrackGuard untuk multi-modal matching (IoU + Color + Shape)
TIDAK menggunakan config JSON - semua tuning di smart_hungarian_trackguard.py
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import time
import logging
from collections import defaultdict

from src.core.smart_hungarian_trackguard import SmartHungarianTrackGuard
from src.analyzers.color_analyzer import ColorAnalyzer
from src.analyzers.shape_analyzer import ShapeAnalyzer

logger = logging.getLogger(__name__)


class KalmanFilter:
    """Simple Kalman filter untuk motion prediction"""
    
    def __init__(self):
        self.state = np.zeros(4)  # [x, y, vx, vy]
        self.covariance = np.eye(4) * 1000
        
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        self.Q = np.array([
            [0.1, 0, 0, 0],
            [0, 0.1, 0, 0],
            [0, 0, 8, 0],
            [0, 0, 0, 8]
        ])
        
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        self.R = np.array([
            [20, 0],
            [0, 20]
        ])
    
    def predict(self):
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        return self.state[:2]
    
    def update(self, measurement):
        y = measurement - self.H @ self.state
        S = self.H @ self.covariance @ self.H.T + self.R
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y
        self.covariance = (np.eye(4) - K @ self.H) @ self.covariance


class Track:
    """Track representation dengan state-based management"""
    
    def __init__(self, track_id: int, detection: Dict, frame_id: int):
        self.track_id = track_id
        self.state = 'active'
        
        # Current state
        self.current_detection = detection
        self.current_frame = frame_id
        
        # Track history
        self.history = [detection]
        self.frame_history = [frame_id]
        
        # Track statistics
        self.age = 1
        self.hits = 1
        self.misses = 0
        self.last_seen = frame_id
        
        # Ghost node specific
        self.ghost_start_frame = None
        self.ghost_predictions = []
        self.kalman_filter = KalmanFilter()
        self.ghost_confidence = 1.0
        self.initial_confidence = detection.get('confidence', 0.8)
        
        # Initialize Kalman filter
        center = self._get_center(detection['bbox'])
        self.kalman_filter.state[:2] = center
        
        # Motion model
        self.velocity = np.array([0.0, 0.0])
        
        # Track quality metrics
        self.avg_confidence = self.initial_confidence
        self.confidence_history = [self.initial_confidence]
        self.stability_score = 1.0
        
        # Track attributes
        self.bbox = detection['bbox']
        self.center = self._get_center(detection['bbox'])
        self.confidence = self.initial_confidence
        self.time_since_update = 0
        self.class_id = detection.get('class_id', 0)
    
    def _get_center(self, bbox):
        """Get center point dari bbox"""
        return np.array([
            (bbox[0] + bbox[2]) / 2.0,
            (bbox[1] + bbox[3]) / 2.0
        ])
    
    def update(self, detection: Dict, frame_id: int):
        """Update track with new detection"""
        # Update motion model
        if len(self.history) >= 2:
            prev_center = self._get_center(self.current_detection['bbox'])
            new_center = self._get_center(detection['bbox'])
            new_velocity = new_center - prev_center
            
            alpha = 0.7
            self.velocity = alpha * self.velocity + (1 - alpha) * new_velocity
        
        # Update Kalman filter
        center = self._get_center(detection['bbox'])
        self.kalman_filter.update(center)
        
        # Update state
        self.current_detection = detection
        self.current_frame = frame_id
        self.last_seen = frame_id
        
        # Update history
        self.history.append(detection)
        self.frame_history.append(frame_id)
        
        # Update confidence tracking
        current_conf = detection.get('confidence', 0.8)
        self.confidence_history.append(current_conf)
        self.avg_confidence = np.mean(self.confidence_history[-5:])
        
        # Update stability score
        self._update_stability_score()
        
        # Update statistics
        self.age += 1
        self.hits += 1
        self.misses = 0
        self.time_since_update = 0
        self.state = 'active'
        
        # Update attributes
        self.bbox = detection['bbox']
        self.center = self._get_center(detection['bbox'])
        self.confidence = current_conf
        
        # Reset ghost state
        if self.state in ['ghost', 'lost']:
            self.ghost_start_frame = None
            self.ghost_predictions = []
            self.ghost_confidence = 1.0
    
    def _update_stability_score(self):
        """Calculate track stability based on motion consistency"""
        if len(self.history) < 3:
            self.stability_score = 1.0
            return
        
        recent_positions = [self._get_center(det['bbox']) for det in self.history[-5:]]
        if len(recent_positions) < 3:
            return
        
        velocities = []
        for i in range(1, len(recent_positions)):
            vel = recent_positions[i] - recent_positions[i-1]
            velocities.append(vel)
        
        if len(velocities) >= 2:
            velocity_std = np.std(velocities, axis=0)
            velocity_consistency = 1.0 / (1.0 + np.mean(velocity_std))
            self.stability_score = 0.7 * velocity_consistency + 0.3 * self.avg_confidence
    
    def miss(self, frame_id: int, enable_ghost: bool = True):
        """Mark track as missed in current frame"""
        self.age += 1
        self.misses += 1
        self.time_since_update += 1
        self.current_frame = frame_id
        
        if enable_ghost and self.state == 'active':
            if self.hits >= 3 and self.stability_score > 0.5:
                self.state = 'ghost'
                self.ghost_start_frame = frame_id
                self.ghost_confidence = min(0.9, self.stability_score)
            else:
                self.state = 'lost'
        elif self.state == 'ghost':
            self.ghost_confidence *= 0.8
            
        # Termination conditions
        max_missing = 20 if self.hits >= 5 else 15
        if self.misses >= max_missing:
            self.state = 'terminated'
        elif not enable_ghost and self.state != 'ghost':
            self.state = 'lost'


class TrackManagerTwoStageSHA:
    """
    State-Based Two-Stage Track Manager dengan Smart Hungarian TrackGuard
    
    Two-Stage berdasarkan STATE:
    - Stage 1: Active tracks (state='active') → Smart Hungarian matching
    - Stage 2: Ghost tracks (state='ghost') → Ghost recovery matching
    """
    
    def __init__(self, 
                 color_analyzer: ColorAnalyzer,
                 shape_analyzer: ShapeAnalyzer,
                 max_age: int = 30,
                 min_hits: int = 3,
                 fps: int = 30,
                 enable_ghost: bool = True):
        """
        Initialize Track Manager
        
        Args:
            color_analyzer: Color analyzer instance
            shape_analyzer: Shape analyzer instance
            max_age: Maximum frames to keep unmatched tracks
            min_hits: Minimum hits needed for track confirmation
            fps: Frames per second
            enable_ghost: Enable ghost node mechanism
        """
        self.color_analyzer = color_analyzer
        self.shape_analyzer = shape_analyzer
        self.max_age = max_age
        self.min_hits = min_hits
        self.fps = fps
        self.enable_ghost = enable_ghost
        
        # Track storage
        self.tracks = {}
        self.next_track_id = 0
        
        # Initialize Smart Hungarian TrackGuard
        # Config hardcode di smart_hungarian_trackguard.py
        self.smart_hungarian = SmartHungarianTrackGuard()
        
        # Statistics
        self.frame_counter = 0
        self.total_tracks_created = 0
        self.total_associations = 0
        self.total_ghost_reidentifications = 0
        
        logger.info("TrackManagerTwoStageSHA initialized")
        logger.info(f"  Max age: {max_age}, Min hits: {min_hits}")
        logger.info(f"  Ghost nodes: {enable_ghost}")
    
    def update(self, frame: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        Update tracks dengan detections menggunakan state-based two-stage matching
        
        Args:
            frame: Current video frame
            detections: List of detections
            
        Returns:
            List of active tracks
        """
        self.frame_counter += 1
        
        if len(detections) == 0:
            return self._handle_empty_detections()
        
        # Get active and ghost tracks
        active_tracks = [t for t in self.tracks.values() if t.state in ['active', 'ghost']]
        
        if len(active_tracks) == 0:
            return self._handle_no_tracks(detections, frame)
        
        # === STAGE 1: Active tracks matching ===
        active_only_tracks = [t for t in active_tracks if t.state == 'active']
        stage1_associations = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = []
        
        if active_only_tracks:
            stage1_result = self._stage1_active_matching(
                active_only_tracks, detections, frame
            )
            stage1_associations = stage1_result['associations']
            unmatched_tracks = stage1_result['unmatched_tracks']
            
            # Update unmatched detections
            matched_det_indices = [assoc['detection_idx'] for assoc in stage1_associations]
            unmatched_detections = [i for i in range(len(detections)) 
                                  if i not in matched_det_indices]
        
        # === STAGE 2: Ghost tracks recovery ===
        ghost_tracks = [t for t in active_tracks if t.state == 'ghost']
        stage2_associations = []
        
        if ghost_tracks:
            stage2_result = self._stage2_ghost_matching(
                ghost_tracks, detections, frame, unmatched_detections
            )
            stage2_associations = stage2_result['associations']
            
            # Update unmatched detections
            for assoc in stage2_associations:
                if assoc['detection_idx'] in unmatched_detections:
                    unmatched_detections.remove(assoc['detection_idx'])
        
        # Merge associations
        all_associations = stage1_associations + stage2_associations
        
        # Update tracks
        for assoc in all_associations:
            track_id = assoc['track_id']
            det_idx = assoc['detection_idx']
            
            if track_id in self.tracks:
                track = self.tracks[track_id]
                track.update(detections[det_idx], self.frame_counter)
                self.total_associations += 1
            
            # Handle unmatched tracks
        for track in unmatched_tracks:
            track.miss(self.frame_counter, enable_ghost=self.enable_ghost)
            
            # Terminate if needed
            if track.state == 'terminated':
                del self.tracks[track.track_id]
        
        # Handle unmatched ghost tracks
        for track in ghost_tracks:
            if track.track_id not in [assoc['track_id'] for assoc in stage2_associations]:
                track.miss(self.frame_counter, enable_ghost=self.enable_ghost)
                if track.state == 'terminated':
                    del self.tracks[track.track_id]
        
        # Create new tracks dari unmatched detections
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            if detection.get('confidence', 0) >= 0.25:  # Min confidence threshold
                new_track = self._create_track(detection)
                self.tracks[new_track.track_id] = new_track
                self.total_tracks_created += 1
        
        # Return active tracks
        active_tracks = [t for t in self.tracks.values() if t.state == 'active']
        return self._tracks_to_dict_list(active_tracks)
    
    def _stage1_active_matching(self, 
                               active_tracks: List[Track], 
                               detections: List[Dict],
                               frame: np.ndarray) -> Dict:
        """
        Stage 1: Active tracks matching menggunakan Smart Hungarian TrackGuard
        """
        # Create multi-modal matrices
        iou_matrix, color_matrix, shape_matrix = self._create_multimodal_matrices(
            active_tracks, detections, frame
        )
        
        # Use Smart Hungarian TrackGuard untuk optimal assignment
        assignments, info = self.smart_hungarian.optimize_assignment_multimodal(
            iou_matrix, color_matrix, shape_matrix, active_tracks, detections
        )
        
        # Convert to association format
        associations = []
        matched_track_indices = set()
        
        for track_idx, det_idx in assignments:
            if track_idx < len(active_tracks):
                track = active_tracks[track_idx]
                associations.append({
                    'track_id': track.track_id,
                    'detection_idx': det_idx,
                    'cost': 1.0 - (iou_matrix[track_idx, det_idx] * 0.6 + 
                                  color_matrix[track_idx, det_idx] * 0.25 + 
                                  shape_matrix[track_idx, det_idx] * 0.15),
                    'type': 'stage1_active_smart_hungarian'
                })
                matched_track_indices.add(track_idx)
        
        # Unmatched tracks
        unmatched_tracks = [active_tracks[i] for i in range(len(active_tracks))
                          if i not in matched_track_indices]
        
        return {
            'associations': associations,
            'unmatched_tracks': unmatched_tracks
        }
    
    def _stage2_ghost_matching(self,
                              ghost_tracks: List[Track],
                              detections: List[Dict],
                              frame: np.ndarray,
                              available_detections: List[int]) -> Dict:
        """
        Stage 2: Ghost tracks recovery matching
        """
        if not available_detections:
            return {'associations': []}
        
        # Filter detections
        available_dets = [detections[i] for i in available_detections]
        
        # Create multi-modal matrices untuk ghost tracks
        iou_matrix, color_matrix, shape_matrix = self._create_multimodal_matrices(
            ghost_tracks, available_dets, frame
        )
        
        # Use Stage 2 Smart Hungarian (more permissive)
        assignments, info = self.smart_hungarian.optimize_assignment_multimodal_stage2(
            iou_matrix, color_matrix, shape_matrix, ghost_tracks, available_dets
        )
        
        # Convert to association format dengan mapping ke original detection indices
        associations = []
        
        for track_idx, local_det_idx in assignments:
            if track_idx < len(ghost_tracks) and local_det_idx < len(available_detections):
                track = ghost_tracks[track_idx]
                original_det_idx = available_detections[local_det_idx]
                
                associations.append({
                    'track_id': track.track_id,
                    'detection_idx': original_det_idx,
                    'cost': 1.0 - (iou_matrix[track_idx, local_det_idx] * 0.4 + 
                                  color_matrix[track_idx, local_det_idx] * 0.35 + 
                                  shape_matrix[track_idx, local_det_idx] * 0.25),
                    'type': 'stage2_ghost_recovery'
                })
                self.total_ghost_reidentifications += 1
        
        return {'associations': associations}
    
    def _create_multimodal_matrices(self,
                                   tracks: List[Track],
                                   detections: List[Dict],
                                   frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create multi-modal similarity matrices (IoU, Color, Shape)
        """
        num_tracks = len(tracks)
        num_detections = len(detections)
        
        if num_tracks == 0 or num_detections == 0:
            return (np.array([]).reshape(num_tracks, num_detections),
                   np.array([]).reshape(num_tracks, num_detections),
                   np.array([]).reshape(num_tracks, num_detections))
        
        # Initialize matrices
        iou_matrix = np.zeros((num_tracks, num_detections), dtype=np.float32)
        color_matrix = np.zeros((num_tracks, num_detections), dtype=np.float32)
        shape_matrix = np.zeros((num_tracks, num_detections), dtype=np.float32)
        
        # Calculate matrices
        for i, track in enumerate(tracks):
            for j, detection in enumerate(detections):
                # IoU
                iou_matrix[i, j] = self._calculate_iou(track.bbox, detection['bbox'])
                
                # Color similarity
                color_matrix[i, j] = self.color_analyzer.analyze(
                    frame, detection['bbox'], track.track_id
                )
                
                # Shape similarity
                shape_matrix[i, j] = self.shape_analyzer.analyze(
                    frame, detection['bbox'], track.track_id
                )
        
        return iou_matrix, color_matrix, shape_matrix

    def _calculate_iou(self, bbox1, bbox2) -> float:
        """Calculate Intersection over Union"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = bbox1_area + bbox2_area - intersection
        
        return intersection / (union + 1e-6)
    
    def _create_track(self, detection: Dict) -> Track:
        """Create new track from detection"""
        track = Track(self.next_track_id, detection, self.frame_counter)
        self.next_track_id += 1
        return track
    
    def _tracks_to_dict_list(self, tracks: List[Track]) -> List[Dict]:
        """Convert Track objects to dict list"""
        result = []
        for track in tracks:
            result.append({
                'track_id': track.track_id,
                'bbox': track.bbox,
                'confidence': track.confidence,
                'class_id': track.class_id,
                'hits': track.hits,
                'age': track.age,
                'state': track.state,
                'history': {
                    'bboxes': [det['bbox'] for det in track.history],
                    'confidences': track.confidence_history,
                    'timestamps': track.frame_history
                }
            })
        return result
    
    def _handle_empty_detections(self) -> List[Dict]:
        """Handle case when no detections"""
        for track in list(self.tracks.values()):
            track.miss(self.frame_counter, enable_ghost=self.enable_ghost)
            if track.state == 'terminated':
                del self.tracks[track.track_id]
        
        active_tracks = [t for t in self.tracks.values() if t.state == 'active']
        return self._tracks_to_dict_list(active_tracks)
    
    def _handle_no_tracks(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """Handle case when no tracks exist"""
        new_tracks = []
        for detection in detections:
            if detection.get('confidence', 0) >= 0.25:
                track = self._create_track(detection)
                self.tracks[track.track_id] = track
                self.total_tracks_created += 1
                new_tracks.append(track)
        
        return self._tracks_to_dict_list(new_tracks)