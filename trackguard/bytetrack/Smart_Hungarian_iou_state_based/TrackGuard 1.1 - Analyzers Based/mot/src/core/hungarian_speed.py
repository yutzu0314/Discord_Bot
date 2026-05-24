"""
Hungarian Speed Calculator Module
================================

Integrated speed calculation yang memanfaatkan Smart Hungarian tracking quality
untuk confidence-weighted speed estimation dengan physics constraints.

Author: Research Team
Version: 1.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import math
import time
from dataclasses import dataclass
import logging

@dataclass
class SpeedConfig:
    """Configuration untuk Hungarian Speed Calculator"""
    pixels_per_meter: float = 20.0
    fps: float = 30.0
    speed_unit: str = 'kmh'  # 'kmh' atau 'ms'
    
    # Physics constraints (m/s²)
    max_acceleration: Dict[str, float] = None
    
    # Confidence weighting parameters
    min_confidence_threshold: float = 0.3
    confidence_weight_exp: float = 2.0  # Exponential untuk confidence weighting
    
    # Temporal smoothing
    smoothing_window: int = 5
    temporal_weight_decay: float = 0.8
    
    # Adaptive pixel ratio parameters
    use_adaptive_pixel_ratio: bool = True
    min_pixel_ratio: float = 15.0  # pixels/meter at bottom (close)
    max_pixel_ratio: float = 35.0  # pixels/meter at top (far)
    
    def __post_init__(self):
        if self.max_acceleration is None:
            self.max_acceleration = {
                'person': 5.0,      # m/s²
                'car': 3.0,         # m/s²
                'truck': 2.5,       # m/s²
                'bus': 2.0,         # m/s²
                'motorcycle': 4.0,  # m/s²
                'bicycle': 4.0,     # m/s²
                'default': 3.0      # m/s²
            }

@dataclass
class SpeedResult:
    """Result dari speed calculation"""
    track_id: int
    speed: float
    confidence: float
    direction_degrees: float
    quality_score: float
    method_used: str
    raw_speed: float
    smoothed_speed: float
    physics_constrained: bool

class HungarianSpeedCalculator:
    """
    Speed calculator yang terintegrasi dengan Smart Hungarian tracking
    untuk confidence-weighted speed estimation
    """
    
    def __init__(self, config: SpeedConfig = None):
        """
        Initialize Hungarian Speed Calculator
        
        Args:
            config: SpeedConfig instance
        """
        self.config = config or SpeedConfig()
        self.logger = logging.getLogger("TrackGuard.HungarianSpeed")
        
        # Speed history untuk temporal smoothing
        self.speed_history: Dict[int, List[Dict]] = {}
        
        # Performance tracking
        self.calculation_stats = {
            'total_calculations': 0,
            'confidence_weighted': 0,
            'physics_constrained': 0,
            'kalman_based': 0,
            'fallback_used': 0
        }
        
        # Frame interval
        self.frame_interval = 1.0 / self.config.fps
        
        self.logger.info(f"HungarianSpeed initialized: {self.config.speed_unit}, "
                        f"adaptive_pixel={self.config.use_adaptive_pixel_ratio}")
    
    def update_speeds(self, 
                     tracked_objects: List[Dict], 
                     hungarian_quality: Dict = None,
                     motion_predictor = None,
                     frame_shape: Tuple[int, int] = None) -> Dict[int, SpeedResult]:
        """
        Update speed calculations untuk all tracked objects
        
        Args:
            tracked_objects: List track results dari Smart Hungarian
            hungarian_quality: Quality metrics dari Hungarian assignment
            motion_predictor: MotionPredictor instance untuk Kalman velocities
            frame_shape: (height, width) untuk adaptive pixel ratio
            
        Returns:
            Dict[track_id, SpeedResult]: Speed results untuk setiap track
        """
        if not tracked_objects:
            return {}
            
        speed_results = {}
        current_time = time.time()
        
        for track in tracked_objects:
            track_id = track['track_id']
            
            try:
                speed_result = self._calculate_track_speed(
                    track, hungarian_quality, motion_predictor, frame_shape, current_time
                )
                
                if speed_result:
                    speed_results[track_id] = speed_result
                    self._update_speed_history(track_id, speed_result, current_time)
                    
            except Exception as e:
                self.logger.warning(f"Speed calculation failed for track {track_id}: {e}")
                continue
        
        # Cleanup old histories
        self._cleanup_old_histories(current_time)
        
        return speed_results
    
    def _calculate_track_speed(self, 
                              track: Dict, 
                              hungarian_quality: Dict, 
                              motion_predictor,
                              frame_shape: Tuple[int, int],
                              current_time: float) -> Optional[SpeedResult]:
        """
        Calculate speed untuk single track dengan multiple methods
        
        Args:
            track: Track data dari Smart Hungarian
            hungarian_quality: Quality metrics
            motion_predictor: MotionPredictor instance
            frame_shape: Frame dimensions
            current_time: Current timestamp
            
        Returns:
            SpeedResult atau None jika calculation gagal
        """
        track_id = track['track_id']
        bbox = track['bbox']
        track_confidence = track.get('confidence', 0.5)
        
        # Get track history
        history = track.get('history', {})
        bboxes = history.get('bboxes', [])
        timestamps = history.get('timestamps', [])
        
        # Minimum requirements untuk speed calculation
        if len(bboxes) < 2 or len(timestamps) < 2:
            return None
        
        # Method 1: Try Kalman velocity (highest priority)
        kalman_speed = self._try_kalman_speed(track_id, motion_predictor, frame_shape)
        
        # Method 2: Multi-frame temporal calculation
        temporal_speed = self._calculate_temporal_speed(track, frame_shape)
        
        # Method 3: Simple frame-to-frame calculation (fallback)
        simple_speed = self._calculate_simple_speed(track, frame_shape)
        
        # Select best method based on availability dan quality
        raw_speed, method_used = self._select_best_method(
            kalman_speed, temporal_speed, simple_speed, track_confidence
        )
        
        if raw_speed is None:
            return None
        
        # Apply confidence weighting
        confidence_weighted_speed = self._apply_confidence_weighting(
            raw_speed, track_confidence, hungarian_quality, track_id
        )
        
        # Apply physics constraints
        constrained_speed, physics_applied = self._apply_physics_constraints(
            confidence_weighted_speed, track_id, track.get('class_id', 0)
        )
        
        # Apply temporal smoothing
        smoothed_speed = self._apply_temporal_smoothing(
            constrained_speed, track_id, current_time
        )
        
        # Calculate direction
        direction = self._calculate_direction(track)
        
        # Quality assessment
        quality_score = self._assess_speed_quality(
            track, hungarian_quality, method_used, physics_applied
        )
        
        # Update statistics
        self.calculation_stats['total_calculations'] += 1
        if method_used == 'kalman':
            self.calculation_stats['kalman_based'] += 1
        elif method_used == 'fallback':
            self.calculation_stats['fallback_used'] += 1
        if physics_applied:
            self.calculation_stats['physics_constrained'] += 1
        
        return SpeedResult(
            track_id=track_id,
            speed=smoothed_speed,
            confidence=track_confidence,
            direction_degrees=direction,
            quality_score=quality_score,
            method_used=method_used,
            raw_speed=raw_speed,
            smoothed_speed=smoothed_speed,
            physics_constrained=physics_applied
        )
    
    def _try_kalman_speed(self, track_id: int, motion_predictor, frame_shape: Tuple[int, int]) -> Optional[float]:
        """Try mendapatkan speed dari Kalman filter velocity"""
        if not motion_predictor:
            return None
            
        try:
            velocity = motion_predictor.get_track_velocity(track_id)
            if velocity is not None:
                # Calculate velocity magnitude dalam pixels/frame
                velocity_magnitude = np.sqrt(velocity[0]**2 + velocity[1]**2)
                
                # Convert ke real-world speed
                if self.config.use_adaptive_pixel_ratio and frame_shape:
                    # Use middle of frame sebagai reference untuk Kalman velocity
                    mid_y = frame_shape[0] // 2
                    pixel_ratio = self._get_adaptive_pixel_ratio(mid_y, frame_shape[0])
                else:
                    pixel_ratio = self.config.pixels_per_meter
                
                # Convert pixels/frame ke meters/second
                speed_ms = (velocity_magnitude / pixel_ratio) / self.frame_interval
                
                # Convert ke unit yang diinginkan
                if self.config.speed_unit == 'kmh':
                    return speed_ms * 3.6
                else:
                    return speed_ms
                    
        except Exception as e:
            self.logger.debug(f"Kalman velocity failed for track {track_id}: {e}")
            
        return None
    
    def _calculate_temporal_speed(self, track: Dict, frame_shape: Tuple[int, int]) -> Optional[float]:
        """Calculate speed menggunakan multiple frames dengan weighting"""
        history = track.get('history', {})
        bboxes = history.get('bboxes', [])
        timestamps = history.get('timestamps', [])
        
        if len(bboxes) < 3 or len(timestamps) < 3:
            return None
        
        # Use recent frames untuk calculation
        recent_frames = min(self.config.smoothing_window, len(bboxes))
        recent_bboxes = bboxes[-recent_frames:]
        recent_timestamps = timestamps[-recent_frames:]
        
        total_distance = 0
        total_time = 0
        weighted_speeds = []
        
        for i in range(1, len(recent_bboxes)):
            # Calculate distance
            center1 = self._get_bbox_center(recent_bboxes[i-1])
            center2 = self._get_bbox_center(recent_bboxes[i])
            
            pixel_distance = math.sqrt(
                (center2[0] - center1[0])**2 + (center2[1] - center1[1])**2
            )
            
            # Time difference
            time_diff = (recent_timestamps[i] - recent_timestamps[i-1]) * self.frame_interval
            
            if time_diff > 0:
                # Adaptive pixel ratio berdasarkan posisi
                if self.config.use_adaptive_pixel_ratio and frame_shape:
                    pixel_ratio = self._get_adaptive_pixel_ratio(center2[1], frame_shape[0])
                else:
                    pixel_ratio = self.config.pixels_per_meter
                
                # Speed calculation
                speed_ms = (pixel_distance / pixel_ratio) / time_diff
                
                # Weight berdasarkan recency
                weight = self.config.temporal_weight_decay ** (len(recent_bboxes) - i - 1)
                weighted_speeds.append((speed_ms, weight))
                
                total_distance += pixel_distance
                total_time += time_diff
        
        if not weighted_speeds:
            return None
        
        # Weighted average
        total_weight = sum(weight for _, weight in weighted_speeds)
        if total_weight > 0:
            avg_speed_ms = sum(speed * weight for speed, weight in weighted_speeds) / total_weight
        else:
            avg_speed_ms = sum(speed for speed, _ in weighted_speeds) / len(weighted_speeds)
        
        # Convert ke unit yang diinginkan
        if self.config.speed_unit == 'kmh':
            return avg_speed_ms * 3.6
        else:
            return avg_speed_ms
    
    def _calculate_simple_speed(self, track: Dict, frame_shape: Tuple[int, int]) -> Optional[float]:
        """Simple frame-to-frame speed calculation sebagai fallback"""
        history = track.get('history', {})
        bboxes = history.get('bboxes', [])
        
        if len(bboxes) < 2:
            return None
        
        # Use last two frames
        center1 = self._get_bbox_center(bboxes[-2])
        center2 = self._get_bbox_center(bboxes[-1])
        
        pixel_distance = math.sqrt(
            (center2[0] - center1[0])**2 + (center2[1] - center1[1])**2
        )
        
        # Adaptive pixel ratio
        if self.config.use_adaptive_pixel_ratio and frame_shape:
            pixel_ratio = self._get_adaptive_pixel_ratio(center2[1], frame_shape[0])
        else:
            pixel_ratio = self.config.pixels_per_meter
        
        # Speed calculation
        speed_ms = (pixel_distance / pixel_ratio) / self.frame_interval
        
        # Convert ke unit yang diinginkan
        if self.config.speed_unit == 'kmh':
            return speed_ms * 3.6
        else:
            return speed_ms
    
    def _select_best_method(self, kalman_speed: Optional[float], 
                           temporal_speed: Optional[float], 
                           simple_speed: Optional[float],
                           track_confidence: float) -> Tuple[Optional[float], str]:
        """Select best speed calculation method"""
        
        # Priority: Kalman > Temporal > Simple
        if kalman_speed is not None and track_confidence >= 0.5:
            return kalman_speed, 'kalman'
        elif temporal_speed is not None:
            return temporal_speed, 'temporal'
        elif simple_speed is not None:
            return simple_speed, 'fallback'
        else:
            return None, 'none'
    
    def _apply_confidence_weighting(self, raw_speed: float, 
                                   track_confidence: float,
                                   hungarian_quality: Dict,
                                   track_id: int) -> float:
        """Apply confidence weighting ke speed estimate"""
        
        # Base confidence dari track
        confidence_factor = max(0.1, min(1.0, track_confidence))
        
        # Hungarian quality bonus jika tersedia
        if hungarian_quality and track_id in hungarian_quality.get('track_qualities', {}):
            hungarian_conf = hungarian_quality['track_qualities'][track_id]
            # Combine track confidence dengan Hungarian assignment quality
            confidence_factor = 0.7 * confidence_factor + 0.3 * hungarian_conf
        
        # Apply exponential weighting
        weight = confidence_factor ** self.config.confidence_weight_exp
        
        # Conservative scaling untuk low confidence
        if confidence_factor < self.config.min_confidence_threshold:
            weight *= 0.5
        
        self.calculation_stats['confidence_weighted'] += 1
        
        return raw_speed * weight
    
    def _apply_physics_constraints(self, speed: float, track_id: int, class_id: int) -> Tuple[float, bool]:
        """Apply physics constraints berdasarkan object type"""
        
        # Get previous speed jika ada
        if track_id in self.speed_history and self.speed_history[track_id]:
            prev_speed = self.speed_history[track_id][-1]['speed']
        else:
            return speed, False  # No previous speed, no constraint needed
        
        # Determine object type
        object_type = self._get_object_type(class_id)
        max_accel = self.config.max_acceleration.get(object_type, 
                                                   self.config.max_acceleration['default'])
        
        # Convert acceleration limit ke speed unit
        if self.config.speed_unit == 'kmh':
            max_accel_per_frame = (max_accel * 3.6) * self.frame_interval
        else:
            max_accel_per_frame = max_accel * self.frame_interval
        
        # Check acceleration limit
        speed_change = abs(speed - prev_speed)
        
        if speed_change > max_accel_per_frame:
            # Limit speed change to physics constraints
            direction = 1 if speed > prev_speed else -1
            constrained_speed = prev_speed + direction * max_accel_per_frame
            return constrained_speed, True
        
        return speed, False
    
    def _apply_temporal_smoothing(self, speed: float, track_id: int, current_time: float) -> float:
        """Apply temporal smoothing menggunakan history"""
        
        if track_id not in self.speed_history:
            return speed
        
        recent_speeds = [entry['speed'] for entry in self.speed_history[track_id][-self.config.smoothing_window:]]
        
        if not recent_speeds:
            return speed
        
        # Exponential moving average
        alpha = 0.3  # Smoothing factor
        smoothed = speed
        
        for prev_speed in reversed(recent_speeds):
            smoothed = alpha * smoothed + (1 - alpha) * prev_speed
            alpha *= 0.8  # Decay factor untuk older values
        
        return smoothed
    
    def _get_adaptive_pixel_ratio(self, y_position: float, frame_height: float) -> float:
        """Calculate adaptive pixel/meter ratio berdasarkan vertical position"""
        
        # Normalize y position (0 = top, 1 = bottom)
        normalized_y = y_position / frame_height
        
        # Linear interpolation
        pixel_ratio = (self.config.min_pixel_ratio + 
                      (self.config.max_pixel_ratio - self.config.min_pixel_ratio) * normalized_y)
        
        return pixel_ratio
    
    def _get_bbox_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Get center point dari bounding box"""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    def _calculate_direction(self, track: Dict) -> float:
        """Calculate movement direction dalam degrees"""
        history = track.get('history', {})
        bboxes = history.get('bboxes', [])
        
        if len(bboxes) < 2:
            return 0.0
        
        center1 = self._get_bbox_center(bboxes[-2])
        center2 = self._get_bbox_center(bboxes[-1])
        
        dx = center2[0] - center1[0]
        dy = center2[1] - center1[1]
        
        # Calculate angle dalam degrees (0 = right, 90 = down)
        angle = math.degrees(math.atan2(dy, dx))
        
        # Normalize ke 0-360
        if angle < 0:
            angle += 360
            
        return angle
    
    def _assess_speed_quality(self, track: Dict, hungarian_quality: Dict, 
                             method_used: str, physics_applied: bool) -> float:
        """Assess quality dari speed estimate"""
        
        base_quality = 0.5
        
        # Method-based quality
        if method_used == 'kalman':
            base_quality = 0.9
        elif method_used == 'temporal':
            base_quality = 0.7
        else:  # fallback
            base_quality = 0.4
        
        # Track confidence bonus
        track_conf = track.get('confidence', 0.5)
        confidence_bonus = track_conf * 0.3
        
        # Track stability bonus
        hits = track.get('hits', 1)
        stability_bonus = min(hits / 10.0, 0.2)
        
        # Physics constraint penalty (indicates unrealistic raw estimate)
        physics_penalty = 0.1 if physics_applied else 0.0
        
        # Hungarian quality bonus
        hungarian_bonus = 0.0
        if hungarian_quality and track['track_id'] in hungarian_quality.get('track_qualities', {}):
            hungarian_conf = hungarian_quality['track_qualities'][track['track_id']]
            hungarian_bonus = hungarian_conf * 0.15
        
        final_quality = base_quality + confidence_bonus + stability_bonus + hungarian_bonus - physics_penalty
        
        return max(0.0, min(1.0, final_quality))
    
    def _get_object_type(self, class_id: int) -> str:
        """Map class_id ke object type untuk physics constraints"""
        
        # Common COCO/YOLO class mappings
        class_mapping = {
            0: 'person',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck',
            1: 'bicycle'
        }
        
        return class_mapping.get(class_id, 'default')
    
    def _update_speed_history(self, track_id: int, speed_result: SpeedResult, current_time: float):
        """Update speed history untuk track"""
        
        if track_id not in self.speed_history:
            self.speed_history[track_id] = []
        
        self.speed_history[track_id].append({
            'speed': speed_result.speed,
            'quality': speed_result.quality_score,
            'timestamp': current_time,
            'method': speed_result.method_used
        })
        
        # Limit history size
        max_history = self.config.smoothing_window * 2
        if len(self.speed_history[track_id]) > max_history:
            self.speed_history[track_id] = self.speed_history[track_id][-max_history:]
    
    def _cleanup_old_histories(self, current_time: float, max_age: float = 30.0):
        """Cleanup old speed histories"""
        
        expired_tracks = []
        
        for track_id, history in self.speed_history.items():
            if history and current_time - history[-1]['timestamp'] > max_age:
                expired_tracks.append(track_id)
        
        for track_id in expired_tracks:
            del self.speed_history[track_id]
    
    def get_statistics(self) -> Dict:
        """Get calculation statistics"""
        
        total = self.calculation_stats['total_calculations']
        if total == 0:
            return self.calculation_stats.copy()
        
        stats = self.calculation_stats.copy()
        stats['percentages'] = {
            'confidence_weighted_pct': (stats['confidence_weighted'] / total) * 100,
            'physics_constrained_pct': (stats['physics_constrained'] / total) * 100,
            'kalman_based_pct': (stats['kalman_based'] / total) * 100,
            'fallback_used_pct': (stats['fallback_used'] / total) * 100
        }
        
        return stats
    
    def reset_statistics(self):
        """Reset calculation statistics"""
        
        self.calculation_stats = {
            'total_calculations': 0,
            'confidence_weighted': 0,
            'physics_constrained': 0,
            'kalman_based': 0,
            'fallback_used': 0
        }
    
    def update_config(self, new_config: SpeedConfig):
        """Update configuration"""
        
        self.config = new_config
        self.frame_interval = 1.0 / self.config.fps
        
        self.logger.info(f"HungarianSpeed config updated: {self.config.speed_unit}, "
                        f"adaptive_pixel={self.config.use_adaptive_pixel_ratio}")

# Utility functions untuk easy integration

def create_default_speed_config(pixels_per_meter: float = 20.0, 
                               fps: float = 30.0, 
                               speed_unit: str = 'kmh') -> SpeedConfig:
    """Create default speed configuration"""
    
    return SpeedConfig(
        pixels_per_meter=pixels_per_meter,
        fps=fps,
        speed_unit=speed_unit,
        use_adaptive_pixel_ratio=True,
        smoothing_window=5,
        confidence_weight_exp=2.0
    )

def create_speed_calculator(config: SpeedConfig = None) -> HungarianSpeedCalculator:
    """Factory function untuk create speed calculator"""
    
    if config is None:
        config = create_default_speed_config()
    
    return HungarianSpeedCalculator(config)