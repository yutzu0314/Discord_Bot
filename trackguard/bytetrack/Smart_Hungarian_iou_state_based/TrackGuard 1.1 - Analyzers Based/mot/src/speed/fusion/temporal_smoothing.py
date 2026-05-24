import numpy as np
from typing import Dict, List, Optional, Tuple
import math
from collections import defaultdict, deque
from scipy import signal

class TemporalSmoothing:
    """
    Advanced temporal smoothing for speed estimates
    
    Applies sophisticated filtering techniques to reduce speed jitter and noise
    while preserving genuine speed changes and accelerations.
    """
    
    def __init__(self, 
                 window_size: int = 7,
                 adaptive_smoothing: bool = True,
                 outlier_detection: bool = True):
        """
        Initialize temporal smoothing system
        
        Args:
            window_size: Size of smoothing window
            adaptive_smoothing: Enable adaptive smoothing based on track quality
            outlier_detection: Enable outlier detection and filtering
        """
        self.window_size = window_size
        self.adaptive_smoothing = adaptive_smoothing
        self.outlier_detection = outlier_detection
        
        # Track-specific data storage
        self.speed_history = defaultdict(lambda: deque(maxlen=window_size))
        self.confidence_history = defaultdict(lambda: deque(maxlen=window_size))
        self.timestamp_history = defaultdict(lambda: deque(maxlen=window_size))
        self.smoothed_history = defaultdict(lambda: deque(maxlen=window_size))
        
        # Smoothing parameters
        self.smoothing_params = {
            'alpha_base': 0.3,      # Base smoothing factor
            'alpha_max': 0.8,       # Maximum smoothing factor
            'alpha_min': 0.1,       # Minimum smoothing factor
            'outlier_threshold': 2.5,  # Z-score threshold for outlier detection
            'acceleration_threshold': 10.0  # m/s² threshold for genuine acceleration
        }
    
    def smooth_speed(self, 
                    track_id: int,
                    current_speed: float,
                    confidence: float = 1.0,
                    timestamp: Optional[float] = None) -> float:
        """
        Apply temporal smoothing to speed estimate
        
        Args:
            track_id: Track identifier
            current_speed: Raw speed estimate to smooth
            confidence: Confidence in current speed estimate
            timestamp: Current timestamp (optional)
            
        Returns:
            float: Smoothed speed estimate
        """
        # Store current data
        self.speed_history[track_id].append(current_speed)
        self.confidence_history[track_id].append(confidence)
        
        if timestamp is not None:
            self.timestamp_history[track_id].append(timestamp)
        
        # Need at least one data point
        if len(self.speed_history[track_id]) < 1:
            return current_speed
        
        # Apply different smoothing strategies based on data availability
        if len(self.speed_history[track_id]) == 1:
            # First measurement - no smoothing possible
            smoothed_speed = current_speed
            
        elif len(self.speed_history[track_id]) <= 3:
            # Limited data - simple exponential smoothing
            smoothed_speed = self._simple_exponential_smoothing(track_id, current_speed, confidence)
            
        else:
            # Sufficient data - advanced smoothing
            smoothed_speed = self._advanced_smoothing(track_id, current_speed, confidence)
        
        # Store smoothed result
        self.smoothed_history[track_id].append(smoothed_speed)
        
        return max(0.0, smoothed_speed)  # Ensure non-negative speed
    
    def _simple_exponential_smoothing(self, 
                                    track_id: int, 
                                    current_speed: float, 
                                    confidence: float) -> float:
        """
        Simple exponential smoothing for tracks with limited history
        
        Args:
            track_id: Track identifier
            current_speed: Current speed estimate
            confidence: Confidence in current estimate
            
        Returns:
            float: Smoothed speed
        """
        speeds = list(self.speed_history[track_id])
        
        if len(speeds) < 2:
            return current_speed
        
        previous_speed = speeds[-2]
        
        # Adaptive alpha based on confidence
        if self.adaptive_smoothing:
            alpha = self._calculate_adaptive_alpha(confidence, track_id)
        else:
            alpha = self.smoothing_params['alpha_base']
        
        # Exponential smoothing
        smoothed_speed = alpha * current_speed + (1 - alpha) * previous_speed
        
        return smoothed_speed
    
    def _advanced_smoothing(self, 
                          track_id: int, 
                          current_speed: float, 
                          confidence: float) -> float:
        """
        Advanced smoothing using multiple techniques
        
        Args:
            track_id: Track identifier
            current_speed: Current speed estimate
            confidence: Confidence in current estimate
            
        Returns:
            float: Smoothed speed
        """
        speeds = list(self.speed_history[track_id])
        confidences = list(self.confidence_history[track_id])
        
        # Step 1: Outlier detection and handling
        if self.outlier_detection:
            current_speed = self._handle_outliers(track_id, current_speed, speeds)
        
        # Step 2: Determine smoothing strategy based on speed pattern
        speed_pattern = self._analyze_speed_pattern(speeds)
        
        if speed_pattern == 'acceleration':
            # Reduced smoothing for genuine acceleration
            smoothed_speed = self._acceleration_aware_smoothing(track_id, current_speed, confidence)
            
        elif speed_pattern == 'stable':
            # Strong smoothing for stable speeds
            smoothed_speed = self._stable_speed_smoothing(track_id, current_speed, confidence)
            
        else:  # 'variable' or unknown pattern
            # Adaptive smoothing
            smoothed_speed = self._adaptive_smoothing_advanced(track_id, current_speed, confidence)
        
        return smoothed_speed
    
    def _handle_outliers(self, 
                        track_id: int, 
                        current_speed: float, 
                        speeds: List[float]) -> float:
        """
        Detect and handle outlier speed measurements
        
        Args:
            track_id: Track identifier
            current_speed: Current speed measurement
            speeds: Historical speed measurements
            
        Returns:
            float: Outlier-corrected speed
        """
        if len(speeds) < 3:
            return current_speed
        
        # Calculate z-score for current measurement
        recent_speeds = speeds[-5:]  # Use last 5 measurements
        mean_speed = np.mean(recent_speeds)
        std_speed = np.std(recent_speeds)
        
        if std_speed > 0:
            z_score = abs(current_speed - mean_speed) / std_speed
            
            if z_score > self.smoothing_params['outlier_threshold']:
                # Potential outlier detected
                
                # Check if this could be genuine acceleration
                if len(speeds) >= 2:
                    acceleration = abs(current_speed - speeds[-1])
                    
                    if acceleration < self.smoothing_params['acceleration_threshold']:
                        # Likely an outlier, not genuine acceleration
                        # Replace with predicted value
                        corrected_speed = self._predict_speed(speeds)
                        return corrected_speed
        
        return current_speed  # Not an outlier or genuine acceleration
    
    def _predict_speed(self, speeds: List[float]) -> float:
        """
        Predict expected speed based on recent trend
        
        Args:
            speeds: Historical speed measurements
            
        Returns:
            float: Predicted speed
        """
        if len(speeds) < 2:
            return speeds[-1] if speeds else 0.0
        
        if len(speeds) == 2:
            # Linear extrapolation
            return 2 * speeds[-1] - speeds[-2]
        
        # Use last 3 points for quadratic extrapolation
        recent_speeds = speeds[-3:]
        
        # Simple trend analysis
        trend1 = recent_speeds[-1] - recent_speeds[-2]
        trend2 = recent_speeds[-2] - recent_speeds[-3]
        
        # Predicted change
        predicted_change = (trend1 + trend2) / 2.0
        predicted_speed = recent_speeds[-1] + predicted_change
        
        return max(0.0, predicted_speed)
    
    def _analyze_speed_pattern(self, speeds: List[float]) -> str:
        """
        Analyze speed pattern to determine appropriate smoothing strategy
        
        Args:
            speeds: Historical speed measurements
            
        Returns:
            str: Pattern type ('stable', 'acceleration', 'variable')
        """
        if len(speeds) < 4:
            return 'variable'
        
        recent_speeds = speeds[-4:]
        
        # Calculate speed changes
        changes = [recent_speeds[i] - recent_speeds[i-1] for i in range(1, len(recent_speeds))]
        
        # Analyze pattern
        mean_change = np.mean(changes)
        std_change = np.std(changes)
        
        # Stable pattern: low variation in speed
        if std_change < 1.0 and abs(mean_change) < 0.5:
            return 'stable'
        
        # Acceleration pattern: consistent direction of change
        if abs(mean_change) > 1.0 and std_change < abs(mean_change) * 0.5:
            return 'acceleration'
        
        # Variable pattern: inconsistent changes
        return 'variable'
    
    def _acceleration_aware_smoothing(self, 
                                    track_id: int, 
                                    current_speed: float, 
                                    confidence: float) -> float:
        """
        Smoothing that preserves genuine accelerations
        
        Args:
            track_id: Track identifier
            current_speed: Current speed
            confidence: Confidence score
            
        Returns:
            float: Smoothed speed
        """
        speeds = list(self.speed_history[track_id])
        
        # Use reduced smoothing to preserve acceleration
        alpha = max(0.6, confidence * 0.8)  # Higher alpha = less smoothing
        
        if len(speeds) >= 2:
            # Weight current measurement more heavily during acceleration
            previous_speed = speeds[-2]
            smoothed_speed = alpha * current_speed + (1 - alpha) * previous_speed
        else:
            smoothed_speed = current_speed
        
        return smoothed_speed
    
    def _stable_speed_smoothing(self, 
                              track_id: int, 
                              current_speed: float, 
                              confidence: float) -> float:
        """
        Strong smoothing for stable speed patterns
        
        Args:
            track_id: Track identifier
            current_speed: Current speed
            confidence: Confidence score
            
        Returns:
            float: Smoothed speed
        """
        speeds = list(self.speed_history[track_id])
        confidences = list(self.confidence_history[track_id])
        
        # Use weighted moving average with higher weights for stable measurements
        weights = []
        for i, conf in enumerate(confidences):
            # Higher weight for more recent and more confident measurements
            recency_weight = (i + 1) / len(confidences)  # 0 to 1
            weight = conf * recency_weight
            weights.append(weight)
        
        # Normalize weights
        if sum(weights) > 0:
            weights = [w / sum(weights) for w in weights]
            smoothed_speed = sum(s * w for s, w in zip(speeds, weights))
        else:
            smoothed_speed = np.mean(speeds)
        
        return smoothed_speed
    
    def _adaptive_smoothing_advanced(self, 
                                   track_id: int, 
                                   current_speed: float, 
                                   confidence: float) -> float:
        """
        Advanced adaptive smoothing for variable speed patterns
        
        Args:
            track_id: Track identifier
            current_speed: Current speed
            confidence: Confidence score
            
        Returns:
            float: Smoothed speed
        """
        speeds = list(self.speed_history[track_id])
        confidences = list(self.confidence_history[track_id])
        
        # Calculate adaptive alpha
        alpha = self._calculate_adaptive_alpha(confidence, track_id)
        
        # Apply Kalman-like filtering
        if len(speeds) >= 2:
            # Estimate measurement noise based on recent variations
            recent_variations = [abs(speeds[i] - speeds[i-1]) for i in range(1, len(speeds))]
            measurement_noise = np.mean(recent_variations) if recent_variations else 1.0
            
            # Adjust alpha based on measurement noise
            noise_factor = 1.0 / (1.0 + measurement_noise)
            adjusted_alpha = alpha * noise_factor
            
            # Apply smoothing
            previous_smoothed = self.smoothed_history[track_id][-1] if self.smoothed_history[track_id] else speeds[-2]
            smoothed_speed = adjusted_alpha * current_speed + (1 - adjusted_alpha) * previous_smoothed
        else:
            smoothed_speed = current_speed
        
        return smoothed_speed
    
    def _calculate_adaptive_alpha(self, confidence: float, track_id: int) -> float:
        """
        Calculate adaptive smoothing factor
        
        Args:
            confidence: Current confidence score
            track_id: Track identifier
            
        Returns:
            float: Adaptive alpha value
        """
        if not self.adaptive_smoothing:
            return self.smoothing_params['alpha_base']
        
        # Base alpha from confidence
        base_alpha = self.smoothing_params['alpha_min'] + \
                    (self.smoothing_params['alpha_max'] - self.smoothing_params['alpha_min']) * confidence
        
        # Adjust based on confidence stability
        confidences = list(self.confidence_history[track_id])
        if len(confidences) > 1:
            conf_stability = 1.0 - np.std(confidences[-3:]) if len(confidences) >= 3 else 1.0
            base_alpha *= conf_stability
        
        # Clamp to valid range
        return max(self.smoothing_params['alpha_min'], 
                  min(self.smoothing_params['alpha_max'], base_alpha))
    
    def get_smoothing_statistics(self, track_id: int) -> Dict:
        """
        Get smoothing statistics for a track
        
        Args:
            track_id: Track identifier
            
        Returns:
            Dict: Smoothing statistics
        """
        speeds = list(self.speed_history[track_id])
        smoothed = list(self.smoothed_history[track_id])
        
        if not speeds:
            return {'track_id': track_id, 'no_data': True}
        
        # Calculate noise reduction
        if len(speeds) > 1 and len(smoothed) > 1:
            raw_variance = np.var(speeds)
            smoothed_variance = np.var(smoothed)
            noise_reduction = (raw_variance - smoothed_variance) / raw_variance if raw_variance > 0 else 0
        else:
            noise_reduction = 0
        
        return {
            'track_id': track_id,
            'data_points': len(speeds),
            'raw_speed_std': np.std(speeds),
            'smoothed_speed_std': np.std(smoothed) if smoothed else 0,
            'noise_reduction': noise_reduction,
            'current_pattern': self._analyze_speed_pattern(speeds) if len(speeds) >= 4 else 'insufficient_data'
        }
    
    def cleanup_stale_tracks(self, active_track_ids: List[int]) -> int:
        """
        Clean up smoothing data for inactive tracks
        
        Args:
            active_track_ids: List of currently active track IDs
            
        Returns:
            int: Number of tracks cleaned up
        """
        all_tracks = set(self.speed_history.keys())
        active_tracks = set(active_track_ids)
        stale_tracks = all_tracks - active_tracks
        
        for track_id in stale_tracks:
            if track_id in self.speed_history:
                del self.speed_history[track_id]
            if track_id in self.confidence_history:
                del self.confidence_history[track_id]
            if track_id in self.timestamp_history:
                del self.timestamp_history[track_id]
            if track_id in self.smoothed_history:
                del self.smoothed_history[track_id]
        
        return len(stale_tracks)
    
    def update_parameters(self, new_params: Dict):
        """
        Update smoothing parameters
        
        Args:
            new_params: Dictionary of parameter updates
        """
        for key, value in new_params.items():
            if key in self.smoothing_params:
                self.smoothing_params[key] = value