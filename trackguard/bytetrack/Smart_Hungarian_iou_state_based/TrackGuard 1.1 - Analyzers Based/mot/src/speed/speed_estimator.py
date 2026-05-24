import numpy as np
import math
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import time

from .calculators.pixel_displacement import PixelDisplacementCalculator
from .calculators.trajectory_fitting import TrajectoryFittingCalculator
from .fusion.confidence_fusion import ConfidenceFusion

class SpeedEstimator:
    """
    Confidence-Weighted Speed Fusion (CWSF) Main Orchestrator
    
    Integrates with TrackGuard to provide real-time speed estimation
    using multiple calculation methods and confidence-based fusion.
    """
    
    def __init__(self, 
                 pixels_per_meter: float = 20.0,
                 fps: float = 30.0,
                 smoothing_window: int = 5,
                 speed_unit: str = "kmh"):
        """
        Initialize CWSF Speed Estimator
        
        Args:
            pixels_per_meter: Camera calibration (pixels per meter)
            fps: Video frames per second
            smoothing_window: Number of frames for temporal smoothing
            speed_unit: Output unit ("ms" for m/s, "kmh" for km/h)
        """
        self.pixels_per_meter = pixels_per_meter
        self.fps = fps
        self.smoothing_window = smoothing_window
        self.speed_unit = speed_unit.lower()
        
        # Initialize calculators
        self.pixel_calculator = PixelDisplacementCalculator(pixels_per_meter, fps)
        self.trajectory_calculator = TrajectoryFittingCalculator(pixels_per_meter, fps)
        self.confidence_fusion = ConfidenceFusion()
        
        # Track history for temporal smoothing
        self.speed_history = defaultdict(lambda: deque(maxlen=smoothing_window))
        self.bbox_history = defaultdict(lambda: deque(maxlen=2))  # Current and previous
        self.confidence_history = defaultdict(lambda: deque(maxlen=smoothing_window))
        
        # Speed statistics
        self.speed_stats = defaultdict(lambda: {
            'current_speed': 0.0,
            'max_speed': 0.0,
            'avg_speed': 0.0,
            'total_distance': 0.0,
            'direction': 0.0
        })
        
        # Configuration for CWSF fusion (now using actual ConfidenceFusion)
        self.frame_counter = 0  # Track frame numbers
        
    def estimate_speed(self, track_id: int, track_data: Dict, 
                      trackguard_confidence: float = 1.0) -> Dict:
        """
        Main speed estimation method - integrates with TrackGuard
        
        Args:
            track_id: Track identifier from TrackGuard
            track_data: Track data including bbox and history
            trackguard_confidence: Confidence score from TrackGuard
            
        Returns:
            Dict: Speed estimation results
        """
        current_bbox = track_data['bbox']
        self.frame_counter += 1
        
        # Store current bbox in history
        self.bbox_history[track_id].append(current_bbox)
        self.confidence_history[track_id].append(trackguard_confidence)
        
        # Need at least 2 bboxes to calculate speed
        if len(self.bbox_history[track_id]) < 2:
            return self._create_speed_result(track_id, 0.0, trackguard_confidence)
        
        # Calculate speed using multiple methods
        speed_estimates = self._calculate_multiple_speeds(track_id, current_bbox)
        
        # Calculate additional quality factors
        stability_factor = self._calculate_tracking_stability(track_id)
        occlusion_factor = self._estimate_occlusion_factor(track_id, current_bbox)
        trajectory_confidence = self.trajectory_calculator.get_trajectory_confidence(track_id)
        speed_consistency = self._calculate_speed_consistency(track_id)
        
        # Apply advanced confidence-weighted fusion
        adaptive_weights = self.confidence_fusion.calculate_adaptive_weights(
            trackguard_confidence=trackguard_confidence,
            stability_factor=stability_factor,
            occlusion_factor=occlusion_factor,
            trajectory_confidence=trajectory_confidence,
            speed_consistency=speed_consistency
        )
        
        # Get method confidences
        method_confidences = {
            'pixel': self.pixel_calculator.get_speed_confidence(
                current_bbox, list(self.bbox_history[track_id])[-2]
            ),
            'trajectory': trajectory_confidence,
            'optical_flow': 0.5  # Placeholder until optical flow is implemented
        }
        
        # Fuse speeds using confidence-weighted fusion
        fused_speed = self.confidence_fusion.fuse_speed_estimates(
            speed_estimates, adaptive_weights, method_confidences
        )
        
        # Apply temporal smoothing
        smoothed_speed = self._apply_temporal_smoothing(track_id, fused_speed)
        
        # Update statistics
        self._update_speed_statistics(track_id, smoothed_speed, current_bbox, 
                                    list(self.bbox_history[track_id])[-2])
        
        # Create comprehensive result with fusion details
        result = self._create_speed_result(track_id, smoothed_speed, trackguard_confidence)
        
        # Add detailed fusion information
        result.update({
            'speed_estimates': speed_estimates,
            'adaptive_weights': adaptive_weights,
            'method_confidences': method_confidences,
            'stability_factor': stability_factor,
            'occlusion_factor': occlusion_factor,
            'trajectory_confidence': trajectory_confidence,
            'fusion_confidence': self.confidence_fusion.calculate_fusion_confidence(
                speed_estimates, adaptive_weights, method_confidences
            )
        })
        
        return result
    
    def batch_estimate_speeds(self, tracks_data: List[Dict]) -> Dict[int, Dict]:
        """
        Estimate speeds for multiple tracks
        
        Args:
            tracks_data: List of track data dictionaries
            
        Returns:
            Dict[int, Dict]: Speed results for each track_id
        """
        results = {}
        
        for track in tracks_data:
            track_id = track['track_id']
            confidence = track.get('confidence', 1.0)
            
            speed_result = self.estimate_speed(track_id, track, confidence)
            results[track_id] = speed_result
            
        return results
    
    def _calculate_multiple_speeds(self, track_id: int, 
                                  current_bbox: List[float]) -> Dict[str, float]:
        """
        Calculate speed using multiple methods
        
        Args:
            track_id: Track identifier
            current_bbox: Current frame bbox
            
        Returns:
            Dict[str, float]: Speed estimates from different methods
        """
        speeds = {}
        previous_bbox = list(self.bbox_history[track_id])[-2]
        
        # Method 1: Pixel displacement (always available)
        if self.speed_unit == "kmh":
            speeds['pixel'] = self.pixel_calculator.calculate_speed_kmh(current_bbox, previous_bbox)
        else:
            speeds['pixel'] = self.pixel_calculator.calculate_speed(current_bbox, previous_bbox)
        
        # Method 2: Trajectory fitting (now implemented!)
        if self.speed_unit == "kmh":
            speeds['trajectory'] = self.trajectory_calculator.calculate_speed_kmh(
                track_id, current_bbox, self.frame_counter
            )
        else:
            speeds['trajectory'] = self.trajectory_calculator.calculate_speed(
                track_id, current_bbox, self.frame_counter
            )
        
        # Method 3: Optical flow (still placeholder)
        speeds['optical_flow'] = speeds['pixel']  # Use pixel as fallback for now
        
        return speeds
    
    def _calculate_tracking_stability(self, track_id: int) -> float:
        """
        Calculate tracking stability based on confidence history
        
        Args:
            track_id: Track identifier
            
        Returns:
            float: Stability factor (0-1)
        """
        if track_id not in self.confidence_history:
            return 0.0
        
        confidences = list(self.confidence_history[track_id])
        if len(confidences) < 2:
            return 1.0
        
        # Calculate coefficient of variation
        mean_conf = np.mean(confidences)
        if mean_conf == 0:
            return 0.0
        
        std_conf = np.std(confidences)
        cv = std_conf / mean_conf
        
        # Convert to stability score (lower CV = higher stability)
        stability = math.exp(-cv * 2)
        return max(0.0, min(1.0, stability))
    
    def _estimate_occlusion_factor(self, track_id: int, current_bbox: List[float]) -> float:
        """
        Estimate occlusion factor based on bbox changes
        
        Args:
            track_id: Track identifier
            current_bbox: Current bounding box
            
        Returns:
            float: Occlusion factor (0-1, 0=no occlusion)
        """
        if track_id not in self.bbox_history or len(self.bbox_history[track_id]) < 2:
            return 0.0
        
        previous_bbox = list(self.bbox_history[track_id])[-2]
        
        # Calculate area change
        current_area = (current_bbox[2] - current_bbox[0]) * (current_bbox[3] - current_bbox[1])
        previous_area = (previous_bbox[2] - previous_bbox[0]) * (previous_bbox[3] - previous_bbox[1])
        
        if previous_area == 0:
            return 0.0
        
        area_ratio = current_area / previous_area
        
        # Sudden area reduction indicates potential occlusion
        if area_ratio < 0.7:
            occlusion_factor = (0.7 - area_ratio) / 0.7
        else:
            occlusion_factor = 0.0
        
        return max(0.0, min(1.0, occlusion_factor))
    
    def _calculate_speed_consistency(self, track_id: int) -> float:
        """
        Calculate speed consistency over recent history
        
        Args:
            track_id: Track identifier
            
        Returns:
            float: Speed consistency (0-1)
        """
        if track_id not in self.speed_history or len(self.speed_history[track_id]) < 3:
            return 1.0
        
        speeds = list(self.speed_history[track_id])
        
        # Calculate coefficient of variation
        mean_speed = np.mean(speeds)
        if mean_speed == 0:
            return 1.0 if all(s == 0 for s in speeds) else 0.0
        
        std_speed = np.std(speeds)
        cv = std_speed / mean_speed
        
        # Convert to consistency score
        consistency = math.exp(-cv)
        return max(0.0, min(1.0, consistency))
    
    def _apply_temporal_smoothing(self, track_id: int, current_speed: float) -> float:
        """
        Apply temporal smoothing to reduce speed jitter
        
        Args:
            track_id: Track identifier
            current_speed: Current frame speed estimate
            
        Returns:
            float: Temporally smoothed speed
        """
        # Add current speed to history
        self.speed_history[track_id].append(current_speed)
        
        # Calculate smoothed speed using moving average
        speeds = list(self.speed_history[track_id])
        
        if len(speeds) == 1:
            return current_speed
        
        # Weighted moving average - more weight to recent speeds
        weights = np.linspace(0.5, 1.0, len(speeds))
        weighted_speed = np.average(speeds, weights=weights)
        
        return float(weighted_speed)
    
    def _update_speed_statistics(self, track_id: int, speed: float, 
                               current_bbox: List[float], previous_bbox: List[float]):
        """
        Update speed statistics for track
        
        Args:
            track_id: Track identifier
            speed: Current speed
            current_bbox: Current bbox
            previous_bbox: Previous bbox
        """
        stats = self.speed_stats[track_id]
        
        # Update current speed
        stats['current_speed'] = speed
        
        # Update max speed
        stats['max_speed'] = max(stats['max_speed'], speed)
        
        # Update average speed
        speeds = list(self.speed_history[track_id])
        stats['avg_speed'] = np.mean(speeds) if speeds else 0.0
        
        # Update total distance traveled
        distance_increment = speed / self.fps  # Distance in this frame
        if self.speed_unit == "kmh":
            distance_increment = distance_increment / 3.6  # Convert km/h to m/s for distance
        stats['total_distance'] += distance_increment
        
        # Update direction
        stats['direction'] = self.pixel_calculator.calculate_direction(current_bbox, previous_bbox)
    
    def _create_speed_result(self, track_id: int, speed: float, 
                           confidence: float) -> Dict:
        """
        Create standardized speed result dictionary
        
        Args:
            track_id: Track identifier
            speed: Estimated speed
            confidence: TrackGuard confidence
            
        Returns:
            Dict: Speed result
        """
        stats = self.speed_stats[track_id]
        
        return {
            'track_id': track_id,
            'speed': speed,
            'speed_unit': self.speed_unit,
            'confidence': confidence,
            'max_speed': stats['max_speed'],
            'avg_speed': stats['avg_speed'],
            'total_distance': stats['total_distance'],
            'direction_degrees': stats['direction'],
            'method': 'CWSF'  # Confidence-Weighted Speed Fusion
        }
    
    def get_track_summary(self, track_id: int) -> Optional[Dict]:
        """
        Get complete summary for a track
        
        Args:
            track_id: Track identifier
            
        Returns:
            Optional[Dict]: Track summary or None if track not found
        """
        if track_id not in self.speed_stats:
            return None
            
        stats = self.speed_stats[track_id]
        speeds = list(self.speed_history[track_id])
        
        return {
            'track_id': track_id,
            'current_speed': stats['current_speed'],
            'max_speed': stats['max_speed'],
            'avg_speed': stats['avg_speed'],
            'total_distance': stats['total_distance'],
            'direction_degrees': stats['direction'],
            'speed_history': speeds,
            'frames_tracked': len(speeds),
            'speed_unit': self.speed_unit
        }
    
    def cleanup_stale_tracks(self, active_track_ids: List[int]) -> int:
        """
        Clean up data for inactive tracks
        
        Args:
            active_track_ids: List of currently active track IDs
            
        Returns:
            int: Number of tracks cleaned up
        """
        # Find stale tracks
        all_tracks = set(self.speed_history.keys())
        active_tracks = set(active_track_ids)
        stale_tracks = all_tracks - active_tracks
        
        # Clean up stale tracks from all components
        for track_id in stale_tracks:
            if track_id in self.speed_history:
                del self.speed_history[track_id]
            if track_id in self.bbox_history:
                del self.bbox_history[track_id]
            if track_id in self.confidence_history:
                del self.confidence_history[track_id]
            if track_id in self.speed_stats:
                del self.speed_stats[track_id]
        
        # Clean up trajectory calculator
        trajectory_cleaned = self.trajectory_calculator.cleanup_stale_tracks(active_track_ids)
        
        return len(stale_tracks) + trajectory_cleaned
    
    def update_calibration(self, pixels_per_meter: float, fps: float = None):
        """
        Update camera calibration parameters
        
        Args:
            pixels_per_meter: New calibration parameter
            fps: New fps value (optional)
        """
        self.pixels_per_meter = pixels_per_meter
        if fps is not None:
            self.fps = fps
            
        # Update calculators
        self.pixel_calculator.update_calibration(pixels_per_meter, fps)
        self.trajectory_calculator.update_calibration(pixels_per_meter, fps)