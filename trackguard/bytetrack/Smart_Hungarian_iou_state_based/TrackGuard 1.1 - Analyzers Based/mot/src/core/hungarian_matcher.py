import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment
import logging

class HungarianMatcher:
    """
    Hungarian Algorithm matcher untuk TrackGuard dengan multi-modal cost matrix
    Menggunakan IoU + Color + Shape untuk optimal assignment
    """
    
    def __init__(self, 
                 iou_weight: float = 0.6,
                 color_weight: float = 0.25,
                 shape_weight: float = 0.15,
                 max_cost_threshold: float = 0.7):
        """
        Initialize Hungarian matcher
        
        Args:
            iou_weight: Weight untuk IoU similarity
            color_weight: Weight untuk color similarity  
            shape_weight: Weight untuk shape similarity
            max_cost_threshold: Maximum cost threshold untuk valid assignment
        """
        self.iou_weight = iou_weight
        self.color_weight = color_weight
        self.shape_weight = shape_weight
        self.max_cost_threshold = max_cost_threshold
        
        # Validate weights
        total_weight = iou_weight + color_weight + shape_weight
        if abs(total_weight - 1.0) > 1e-6:
            # Normalize weights
            self.iou_weight = iou_weight / total_weight
            self.color_weight = color_weight / total_weight
            self.shape_weight = shape_weight / total_weight
        
        self.logger = logging.getLogger("TrackGuard.HungarianMatcher")
        
    def calculate_cost_matrix(self, 
                            tracks: List[Dict], 
                            detections: List[Dict],
                            frame: np.ndarray,
                            color_analyzer,
                            shape_analyzer,
                            bbox_calculator) -> np.ndarray:
        """
        Calculate cost matrix untuk Hungarian assignment
        
        Args:
            tracks: List of active tracks
            detections: List of detections
            frame: Current video frame
            color_analyzer: Color analyzer instance
            shape_analyzer: Shape analyzer instance
            bbox_calculator: Function untuk calculate IoU
            
        Returns:
            np.ndarray: Cost matrix (tracks x detections)
        """
        num_tracks = len(tracks)
        num_detections = len(detections)
        
        if num_tracks == 0 or num_detections == 0:
            return np.array([]).reshape(num_tracks, num_detections)
        
        # Initialize cost matrix dengan high cost (low similarity)
        cost_matrix = np.full((num_tracks, num_detections), 1.0, dtype=np.float32)
        
        for i, track in enumerate(tracks):
            for j, detection in enumerate(detections):
                # Calculate individual similarity scores
                similarity_scores = self._calculate_similarities(
                    track, detection, frame, color_analyzer, shape_analyzer, bbox_calculator
                )
                
                # Calculate combined similarity
                combined_similarity = (
                    self.iou_weight * similarity_scores['iou'] +
                    self.color_weight * similarity_scores['color'] +
                    self.shape_weight * similarity_scores['shape']
                )
                
                # Convert similarity to cost (1 - similarity)
                cost = 1.0 - combined_similarity
                cost_matrix[i, j] = cost
        
        return cost_matrix
    
    def calculate_adaptive_cost_matrix(self, 
                                     tracks: List[Dict], 
                                     detections: List[Dict],
                                     frame: np.ndarray,
                                     color_analyzer,
                                     shape_analyzer,
                                     bbox_calculator,
                                     occlusion_detector=None) -> np.ndarray:
        """
        Calculate adaptive cost matrix dengan context-aware weighting
        
        Args:
            tracks: List of active tracks
            detections: List of detections
            frame: Current video frame
            color_analyzer: Color analyzer instance
            shape_analyzer: Shape analyzer instance
            bbox_calculator: Function untuk calculate IoU
            occlusion_detector: Function untuk detect occlusion
            
        Returns:
            np.ndarray: Adaptive cost matrix
        """
        num_tracks = len(tracks)
        num_detections = len(detections)
        
        if num_tracks == 0 or num_detections == 0:
            return np.array([]).reshape(num_tracks, num_detections)
        
        cost_matrix = np.full((num_tracks, num_detections), 1.0, dtype=np.float32)
        
        for i, track in enumerate(tracks):
            # Determine track context untuk adaptive weighting
            is_occluded = occlusion_detector(track) if occlusion_detector else False
            is_static = self._is_static_track(track)
            is_edge = self._is_near_edge(track)
            
            # Adaptive weights berdasarkan track context
            weights = self._get_adaptive_weights(is_occluded, is_static, is_edge)
            
            for j, detection in enumerate(detections):
                # Calculate individual similarities
                similarity_scores = self._calculate_similarities(
                    track, detection, frame, color_analyzer, shape_analyzer, bbox_calculator
                )
                
                # Apply adaptive weighting
                combined_similarity = (
                    weights['iou'] * similarity_scores['iou'] +
                    weights['color'] * similarity_scores['color'] +
                    weights['shape'] * similarity_scores['shape']
                )
                
                # Convert to cost
                cost = 1.0 - combined_similarity
                cost_matrix[i, j] = cost
        
        return cost_matrix
    
    def _calculate_similarities(self, 
                              track: Dict, 
                              detection: Dict,
                              frame: np.ndarray,
                              color_analyzer,
                              shape_analyzer,
                              bbox_calculator) -> Dict[str, float]:
        """Calculate individual similarity scores"""
        similarities = {
            'iou': 0.0,
            'color': 0.0,
            'shape': 0.0
        }
        
        try:
            # IoU similarity
            iou = bbox_calculator(track['bbox'], detection['bbox'])
            similarities['iou'] = max(0.0, min(1.0, iou))
            
            # Color similarity
            if color_analyzer:
                color_score = color_analyzer.analyze(
                    frame, detection['bbox'], track.get('track_id', -1)
                )
                similarities['color'] = max(0.0, min(1.0, color_score))
            
            # Shape similarity
            if shape_analyzer:
                shape_score = shape_analyzer.analyze(
                    frame, detection['bbox'], track.get('track_id', -1)
                )
                similarities['shape'] = max(0.0, min(1.0, shape_score))
                
        except Exception as e:
            self.logger.warning(f"Error calculating similarities: {e}")
            # Return low similarities untuk safety
            similarities = {'iou': 0.0, 'color': 0.0, 'shape': 0.0}
        
        return similarities
    
    def _get_adaptive_weights(self, is_occluded: bool, is_static: bool, is_edge: bool) -> Dict[str, float]:
        """Get adaptive weights berdasarkan track context"""
        if is_occluded:
            # Boost appearance features saat occlusion
            weights = {
                'iou': 0.4,      # Reduced IoU weight
                'color': 0.35,   # Increased color weight
                'shape': 0.25    # Increased shape weight
            }
        elif is_static:
            # Static objects - moderate appearance reliance
            weights = {
                'iou': 0.55,
                'color': 0.275,
                'shape': 0.175
            }
        elif is_edge:
            # Edge objects - be more conservative
            weights = {
                'iou': 0.65,
                'color': 0.22,
                'shape': 0.13
            }
        else:
            # Normal case - default weights
            weights = {
                'iou': self.iou_weight,
                'color': self.color_weight,
                'shape': self.shape_weight
            }
        
        return weights
    
    def _is_static_track(self, track: Dict) -> bool:
        """Simple static detection berdasarkan movement history"""
        if 'history' not in track or 'bboxes' not in track['history']:
            return False
            
        history_bboxes = track['history']['bboxes']
        if len(history_bboxes) < 3:
            return False
        
        # Calculate movement in last few frames
        recent_bboxes = history_bboxes[-3:]
        movements = []
        
        for i in range(1, len(recent_bboxes)):
            prev_center = self._get_bbox_center(recent_bboxes[i-1])
            curr_center = self._get_bbox_center(recent_bboxes[i])
            
            movement = np.sqrt(
                (curr_center[0] - prev_center[0])**2 + 
                (curr_center[1] - prev_center[1])**2
            )
            movements.append(movement)
        
        # Static jika movement rata-rata < threshold
        avg_movement = np.mean(movements)
        return avg_movement < 10.0  # 10 pixels threshold
    
    def _is_near_edge(self, track: Dict, margin: int = 50) -> bool:
        """Check if track near frame edge"""
        bbox = track['bbox']
        # Assume standard frame size - bisa diubah jadi parameter
        frame_width, frame_height = 1920, 1080
        
        x1, y1, x2, y2 = bbox
        return (x1 < margin or y1 < margin or 
                x2 > frame_width - margin or y2 > frame_height - margin)
    
    def _get_bbox_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Get center point of bbox"""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    def solve_assignment(self, cost_matrix: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Solve assignment problem menggunakan Hungarian algorithm
        
        Args:
            cost_matrix: Cost matrix (tracks x detections)
            
        Returns:
            Tuple containing:
            - matches: List of (track_idx, detection_idx) pairs
            - unmatched_tracks: List of unmatched track indices
            - unmatched_detections: List of unmatched detection indices
        """
        if cost_matrix.size == 0:
            return [], [], []
        
        num_tracks, num_detections = cost_matrix.shape
        
        # Apply Hungarian algorithm
        try:
            track_indices, detection_indices = linear_sum_assignment(cost_matrix)
        except Exception as e:
            self.logger.error(f"Hungarian algorithm failed: {e}")
            # Fallback ke empty assignment
            return [], list(range(num_tracks)), list(range(num_detections))
        
        # Filter matches berdasarkan cost threshold
        matches = []
        matched_tracks = set()
        matched_detections = set()
        
        for track_idx, det_idx in zip(track_indices, detection_indices):
            cost = cost_matrix[track_idx, det_idx]
            
            # Only accept match jika cost < threshold (high similarity)
            if cost < self.max_cost_threshold:
                matches.append((track_idx, det_idx))
                matched_tracks.add(track_idx)
                matched_detections.add(det_idx)
        
        # Find unmatched tracks and detections
        unmatched_tracks = [i for i in range(num_tracks) if i not in matched_tracks]
        unmatched_detections = [i for i in range(num_detections) if i not in matched_detections]
        
        return matches, unmatched_tracks, unmatched_detections
    
    def match_tracks_to_detections(self,
                                 tracks: List[Dict],
                                 detections: List[Dict],
                                 frame: np.ndarray,
                                 color_analyzer,
                                 shape_analyzer,
                                 bbox_calculator,
                                 use_adaptive: bool = True,
                                 occlusion_detector=None) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Main function untuk match tracks ke detections menggunakan Hungarian algorithm
        
        Args:
            tracks: List of active tracks
            detections: List of detections
            frame: Current video frame
            color_analyzer: Color analyzer instance
            shape_analyzer: Shape analyzer instance
            bbox_calculator: Function untuk calculate IoU
            use_adaptive: Whether to use adaptive weighting
            occlusion_detector: Function untuk detect occlusion
            
        Returns:
            Tuple containing matches, unmatched_tracks, unmatched_detections
        """
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Calculate cost matrix
        if use_adaptive:
            cost_matrix = self.calculate_adaptive_cost_matrix(
                tracks, detections, frame, color_analyzer, 
                shape_analyzer, bbox_calculator, occlusion_detector
            )
        else:
            cost_matrix = self.calculate_cost_matrix(
                tracks, detections, frame, color_analyzer,
                shape_analyzer, bbox_calculator
            )
        
        # Solve assignment
        matches, unmatched_tracks, unmatched_detections = self.solve_assignment(cost_matrix)
        
        # Log results untuk debugging
        self.logger.debug(f"Hungarian matching: {len(matches)} matches, "
                         f"{len(unmatched_tracks)} unmatched tracks, "
                         f"{len(unmatched_detections)} unmatched detections")
        
        return matches, unmatched_tracks, unmatched_detections
    
    def update_weights(self, iou_weight: float, color_weight: float, shape_weight: float):
        """Update weights untuk matching"""
        total = iou_weight + color_weight + shape_weight
        self.iou_weight = iou_weight / total
        self.color_weight = color_weight / total  
        self.shape_weight = shape_weight / total
        
    def get_current_weights(self) -> Dict[str, float]:
        """Get current weights"""
        return {
            'iou': self.iou_weight,
            'color': self.color_weight,
            'shape': self.shape_weight
        }